"""One MuPDF lock even when legacy code imports native under two module names."""
import threading
PDF_LOCK=threading.RLock()
