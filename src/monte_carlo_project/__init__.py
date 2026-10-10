import os, ctypes
if "NUMBA_ENABLE_CUDASIM" not in os.environ:
    try:
        ctypes.CDLL("libcuda.so.1")          # driver NVIDIA présent ?
    except OSError:
        os.environ["NUMBA_ENABLE_CUDASIM"] = "1"