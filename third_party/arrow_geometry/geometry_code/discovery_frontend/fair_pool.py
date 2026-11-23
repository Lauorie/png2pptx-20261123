"""Shared bounded API task pool, round-robin among queued image queues."""
from collections import deque
from concurrent.futures import Future
import threading

class FairPool:
    def __init__(self,workers):
        self.queues={};self.order=deque();self.cv=threading.Condition();self.closed=False
        self.workers=[threading.Thread(target=self._work,daemon=True) for _ in range(workers)]
        for t in self.workers:t.start()
    def submit(self,key,fn,*args,**kwargs):
        f=Future()
        with self.cv:
            if self.closed:raise RuntimeError('pool closed')
            if key not in self.queues:self.queues[key]=deque();self.order.append(key)
            self.queues[key].append((f,fn,args,kwargs));self.cv.notify()
        return f
    def _work(self):
        while True:
            with self.cv:
                self.cv.wait_for(lambda:self.order or self.closed)
                if not self.order:return
                key=self.order.popleft();job=self.queues[key].popleft()
                if self.queues[key]:self.order.append(key)
                else:del self.queues[key]
            f,fn,args,kwargs=job
            if f.set_running_or_notify_cancel():
                try:f.set_result(fn(*args,**kwargs))
                except BaseException as exc:f.set_exception(exc)
    def shutdown(self):
        with self.cv:self.closed=True;self.cv.notify_all()
        for t in self.workers:t.join()
