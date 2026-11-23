"""Shared request reservations plus a conservative account-balance stop guard.

The default mode uses verified prices and input-token bounds. An explicitly
selected account_balance mode follows the user's balance-monitor policy;
it records usage without inventing a monetary estimate or claiming a hard
upper bound on charges not yet reported by the provider.
"""
from contextlib import contextmanager
from datetime import datetime
from decimal import Decimal, ROUND_CEILING
import json
from pathlib import Path
import re
import sqlite3
import time
import uuid


class BudgetPaused(RuntimeError):
    pass


class BudgetBusy(RuntimeError):
    pass


def money(value):
    d = Decimal(str(value))
    if not d.is_finite() or d < 0:
        raise ValueError('Money must be finite and nonnegative')
    return int((d * 1_000_000).to_integral_value(rounding=ROUND_CEILING))


def balance_snapshot(path):
    with Path(path).open('rb') as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 16384))
        tail = f.read().decode('utf-8', errors='replace')
    rows = re.findall(r'\[([^\]]+)\] OK Key=(\d+) 剩余额度=(-?[\d.]+) 累计消费=([\d.]+)', tail)
    if not rows:
        raise ValueError('No valid recent balance record')
    stamp, key, remain, spent = rows[-1]
    return {'timestamp': datetime.strptime(stamp, '%Y-%m-%d %H:%M:%S %z').timestamp(),
            'key_id': key, 'remaining': remain, 'total_spent': spent}


class BudgetGuard:
    def __init__(self, config_path):
        self.path = Path(config_path).resolve()
        self.config = json.loads(self.path.read_text())
        self.root = self.path.parent
        self.limit = money(self.config['limit_cny'])
        self.mode = self.config.get('mode', 'priced_reservations')
        if self.mode not in ('priced_reservations', 'account_balance'):
            raise ValueError('Unknown budget monitoring mode')
        self.conversion = Decimal(str(self.config['balance_unit_cny']))
        if not self.conversion.is_finite() or self.conversion <= 0:
            raise ValueError('Verified balance currency conversion required')
        self.prices = self.config['prices']
        if not self.prices and self.mode == 'priced_reservations':
            raise ValueError('Verified model prices required before generation')
        for p in self.prices.values():
            money(p['input_cny_per_million']); money(p['output_cny_per_million'])
            if type(p['max_input_tokens']) is not int or p['max_input_tokens'] < 1:
                raise ValueError('Model input-token upper bound required')
        self.database = self.root/'ledger.sqlite3'
        self.pause_path = self.root/'PAUSED.json'
        with self.transaction() as db:
            db.execute('CREATE TABLE IF NOT EXISTS requests (id TEXT PRIMARY KEY, '
                       'created REAL, model TEXT, reserved INTEGER, charged INTEGER, '
                       'state TEXT, usage TEXT, owner TEXT)')

    @contextmanager
    def transaction(self):
        db = sqlite3.connect(self.database, timeout=60)
        try:
            db.execute('BEGIN IMMEDIATE')
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def pause(self, reason):
        # Immutable first cause, shared by pilot, fallback, batch and evaluation.
        try:
            with self.pause_path.open('x') as f:
                json.dump({'state': 'paused_budget', 'reason': reason, 'at': time.time(),
                           'checkpoints_retained': True}, f, indent=2)
        except FileExistsError:
            pass
        raise BudgetPaused(reason)

    def account_delta(self):
        try:
            now = balance_snapshot(self.config['balance_file'])
        except (ValueError, OSError):
            return self.pause('Balance monitor unreadable')
        old = self.config['baseline']
        if now['key_id'] != str(old['key_id']):
            return self.pause('Balance monitor key changed')
        age = time.time() - now['timestamp']
        if not -10 <= age <= self.config.get('max_monitor_age_seconds', 120):
            return self.pause('Balance monitor stale; new requests disabled')
        if Decimal(old['remaining']) > 0 and Decimal(now['remaining']) <= 0:
            return self.pause('API key quota exhausted; new requests disabled')
        delta = max(Decimal(now['total_spent'])-Decimal(old['total_spent']),
                    Decimal(old['remaining'])-Decimal(now['remaining']), Decimal(0))
        return money(delta * self.conversion)

    def totals(self, db):
        rows = db.execute('SELECT state, SUM(reserved), SUM(charged), COUNT(*) '
                          'FROM requests GROUP BY state').fetchall()
        charged = sum((c or 0) for _, _, c, _ in rows)
        pending = sum((r or 0) for s, r, _, _ in rows if s == 'pending')
        return charged, pending, {s: n for s, _, _, n in rows}

    def check(self):
        if self.pause_path.exists():
            raise BudgetPaused('Task budget is paused; explicit resume required')
        account = self.account_delta()
        with self.transaction() as db:
            charged, pending, counts = self.totals(db)
        if max(account, charged) >= self.limit:
            return self.pause('500 CNY task limit reached' if self.limit == money(500)
                              else 'Configured task limit reached')
        return {'mode':self.mode, 'limit_cny': self.limit/1e6, 'account_delta_cny': account/1e6,
                'task_charge_upper_bound_cny': charged/1e6 if self.mode == 'priced_reservations' else None,
                'pending_reservations_cny': pending/1e6, 'requests': counts,
                'warning':account >= money(self.config.get('warning_cny',self.config['limit_cny'])),
                'unreported_provider_charges_bounded':self.mode == 'priced_reservations',
                'account_delta_includes_other_activity': True, 'at': time.time()}

    def reserve(self, payload, owner):
        if self.pause_path.exists():
            raise BudgetPaused('Task budget is paused')
        model = payload['model']
        if model not in self.prices and self.mode == 'priced_reservations':
            return self.pause('Unpriced model; request blocked')
        output = payload.get('max_completion_tokens', payload.get('max_tokens'))
        if type(output) is not int or output < 1 or payload.get('n', 1) != 1:
            return self.pause('Unbounded completion; request blocked')
        reserved = 0
        if self.mode == 'priced_reservations':
            price = self.prices[model]
            reserved = money((Decimal(price['max_input_tokens'])*Decimal(str(price['input_cny_per_million']))
                              + Decimal(output)*Decimal(str(price['output_cny_per_million'])))/1_000_000)
        account = self.account_delta()
        ident = uuid.uuid4().hex
        with self.transaction() as db:
            charged, pending, _ = self.totals(db)
            if max(account, charged) >= self.limit:
                return self.pause('Configured task limit reached')
            if max(account, charged) + pending + reserved > self.limit:
                if pending:
                    raise BudgetBusy('Waiting for in-flight reservations to settle')
                return self.pause('Insufficient remaining budget for a bounded request')
            db.execute('INSERT INTO requests VALUES (?,?,?,?,?,?,?,?)',
                       (ident, time.time(), model, reserved, 0, 'pending', None, str(owner)))
        return ident

    def settle(self, ident, usage=None):
        exceeded = False
        with self.transaction() as db:
            row = db.execute('SELECT model,reserved,state FROM requests WHERE id=?', (ident,)).fetchone()
            if row is None or row[2] != 'pending':
                raise ValueError('Unknown or already settled reservation')
            model, reserved, _ = row
            if self.mode == 'account_balance':
                db.execute('UPDATE requests SET state=?,usage=? WHERE id=?',
                           ('usage_recorded' if usage else 'usage_unavailable',
                            json.dumps(usage) if usage else None, ident))
                return
            p = self.prices[model]
            inp = (usage or {}).get('prompt_tokens')
            out = (usage or {}).get('completion_tokens')
            valid = all(type(v) is int and v >= 0 for v in (inp, out))
            # Prompt includes cached input; bill all at the full input price.
            # completion_tokens includes reasoning; never subtract reasoning.
            charged = money((Decimal(inp)*Decimal(str(p['input_cny_per_million'])) +
                             Decimal(out)*Decimal(str(p['output_cny_per_million'])))/1_000_000) if valid else reserved
            exceeded = charged > reserved or (valid and inp > p['max_input_tokens'])
            db.execute('UPDATE requests SET charged=?,state=?,usage=? WHERE id=?',
                       (charged, 'settled' if valid else 'unknown_charge',
                        json.dumps(usage) if usage else None, ident))
        if exceeded:
            self.pause('Observed usage exceeded configured price/token bound')

