"""ModelWorker: run zoo models in a child process and recycle it when NPU memory runs out.

libnoe 3.1.2 only frees NPU memory when the process exits (~1.5 GB of graphs per process lifetime).
Apps that switch between many models (like the demo) do inference through a worker:

    w = ModelWorker()
    res, vis = w.call("yolov8_l", img, conf=0.3)            # vis = m.draw(img, res), or None with draw=False
    w.call("sam2", img, method="segment", points=[(x, y, 1)])  # any method of the model

On an out-of-memory load the child is replaced (all its models are dropped) and the call retried once.
Calls are serialized; arguments and results travel by pickle (numpy arrays are fine).
"""
import multiprocessing as mp
import threading
import traceback

_OOM = ("ALLOC", "create_job")


def _serve(conn):
    import zoo
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            return
        name, method, img, kw, draw = msg
        try:
            m = zoo.get(name)
            res = m(img, **kw) if method is None else getattr(m, method)(img, **kw)
            vis = m.draw(img, res) if draw and method is None else None
            conn.send(("ok", res, vis, zoo.loaded()))
        except RuntimeError as e:
            kind = "oom" if any(k in str(e) for k in _OOM) else "err"
            conn.send((kind, f"{type(e).__name__}: {e}", traceback.format_exc(), zoo.loaded()))
        except Exception as e:
            conn.send(("err", f"{type(e).__name__}: {e}", traceback.format_exc(), zoo.loaded()))


class ModelWorker:
    def __init__(self):
        self._mp = mp.get_context("spawn")
        self._lock = threading.Lock()
        self._proc = self._conn = None
        self.loaded = []
        self.restarts = 0

    def _start(self):
        self._conn, child = self._mp.Pipe()
        self._proc = self._mp.Process(target=_serve, args=(child,), daemon=True)
        self._proc.start()
        child.close()

    def restart(self):
        if self._proc is not None:
            self._proc.kill()
            self._proc.join(5)
        self._proc = self._conn = None
        self.loaded = []
        self.restarts += 1

    def call(self, name, img, method=None, draw=True, **kw):
        with self._lock:
            for attempt in (0, 1):
                if self._proc is None or not self._proc.is_alive():
                    self._start()
                try:
                    self._conn.send((name, method, img, kw, draw))
                    r = self._conn.recv()
                except (EOFError, BrokenPipeError, ConnectionResetError):  # child crashed
                    self.restart()
                    if attempt:
                        raise RuntimeError(f"worker crashed running {name}")
                    continue
                self.loaded = r[-1]
                if r[0] == "ok":
                    return r[1], r[2]
                if r[0] == "oom" and attempt == 0 and len(self.loaded) > 0:
                    self.restart()  # free all NPU memory, retry in a fresh process
                    continue
                raise RuntimeError(r[1])
