"""Thin wrapper over libnoe (CIX NOE UMD 3.1.2) for one compiled .cix graph.

Handles quantize/dequantize per tensor descriptor and a pool of reusable jobs so
several threads can submit inferences.
"""
import queue
import threading

import numpy as np
import libnoe

_CTX = None
_CTX_LOCK = threading.Lock()


def _context():
    """One NOE context per process, shared by all graphs."""
    global _CTX
    with _CTX_LOCK:
        if _CTX is None:
            npu = libnoe.NPU()
            if npu.noe_init_context() != 0:
                raise RuntimeError("noe_init_context failed (is /dev/aipu passed in and the aipu module loaded?)")
            _CTX = npu
        return _CTX


def _np_dtype(dt):
    T = libnoe.noe_data_type_t
    return {T.NOE_DATA_TYPE_S8: np.int8, T.NOE_DATA_TYPE_U8: np.uint8, T.NOE_DATA_TYPE_S16: np.int16,
            T.NOE_DATA_TYPE_U16: np.uint16, T.NOE_DATA_TYPE_S32: np.int32, T.NOE_DATA_TYPE_U32: np.uint32,
            T.NOE_DATA_TYPE_F16: np.float16, T.NOE_DATA_TYPE_F32: np.float32}[dt]


class Tensor:
    def __init__(self, desc):
        self.desc = desc
        self.dtype = _np_dtype(desc.data_type)
        self.size = desc.size
        self.count = desc.size // np.dtype(self.dtype).itemsize
        self.scale, self.zp = desc.scale, desc.zero_point

    def quant(self, x):
        x = np.asarray(x)
        if np.issubdtype(self.dtype, np.floating):
            return x.astype(self.dtype)
        if x.dtype == self.dtype:  # already in device format
            return x
        info = np.iinfo(self.dtype)
        return np.clip(np.round(x.astype(np.float32) * self.scale - self.zp), info.min, info.max).astype(self.dtype)

    def dequant(self, raw):
        a = np.frombuffer(raw, dtype=self.dtype)
        if np.issubdtype(self.dtype, np.floating):
            return a.astype(np.float32)
        return (a.astype(np.float32) + self.zp) / self.scale

    def __repr__(self):
        return f"<{np.dtype(self.dtype).name} x{self.count} scale={self.scale:.5g} zp={self.zp}>"


class Graph:
    """A loaded .cix with `jobs` reusable jobs. run([inputs...]) -> [flat float32 outputs...]."""

    def __init__(self, cix_path, jobs=1):
        self.npu = _context()
        ret, self.graph = self.npu.noe_load_graph(cix_path)
        if ret != 0:
            raise RuntimeError(f"noe_load_graph({cix_path}) failed: {ret}")
        I, O = libnoe.NOE_TENSOR_TYPE_INPUT, libnoe.NOE_TENSOR_TYPE_OUTPUT
        self.ins = [Tensor(self.npu.noe_get_tensor_descriptor(self.graph, I, i)) for i in range(self._count(I))]
        self.outs = [Tensor(self.npu.noe_get_tensor_descriptor(self.graph, O, i)) for i in range(self._count(O))]
        self.jobs = queue.Queue()
        for _ in range(jobs):
            ret, job = self.npu.noe_create_job(self.graph, libnoe.noe_create_job_cfg_t())
            if ret != 0:
                raise RuntimeError("noe_create_job failed")
            self.jobs.put(job)

    def close(self):
        """Free the jobs and unload the graph. The object is unusable afterwards.
        NOTE: on UMD 3.1.2 this does not return NPU memory (noe_unload_graph returns SUCCESS but the
        buffers stay allocated until the process exits; noe_deinit_context frees them, but loading a
        different graph after a re-init segfaults). Use zoo.worker.ModelWorker to recycle memory."""
        while not self.jobs.empty():
            self.npu.noe_clean_job(self.jobs.get())
        if self.graph is not None:
            self.npu.noe_unload_graph(self.graph)
            self.graph = None

    def _count(self, t):
        ret, n = self.npu.noe_get_tensor_count(self.graph, t)
        if ret != 0:
            raise RuntimeError("noe_get_tensor_count failed")
        return n

    def run(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        assert len(inputs) == len(self.ins), f"{len(inputs)} inputs given, graph wants {len(self.ins)}"
        job = self.jobs.get()
        try:
            for i, (x, t) in enumerate(zip(inputs, self.ins)):
                b = t.quant(x).tobytes()
                if len(b) != t.size:
                    raise ValueError(f"input {i}: {len(b)} bytes != {t.size} ({t})")
                self.npu.noe_load_tensor(job, i, b)
            ret = self.npu.noe_job_infer_sync(job, -1)
            if ret != 0:
                raise RuntimeError(f"noe_job_infer_sync failed: {ret}")
            outs = []
            for j, t in enumerate(self.outs):
                ret, raw = self.npu.noe_get_tensor(job, libnoe.NOE_TENSOR_TYPE_OUTPUT, j)
                if ret != 0:
                    raise RuntimeError("noe_get_tensor failed")
                outs.append(t.dequant(raw))
            return outs
        finally:
            self.jobs.put(job)
