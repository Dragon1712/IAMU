"""Linux ctypes adapter for ERL-DDE's original BACAP C++ simulation.

Adapted from upstream env.py at:
https://github.com/EvoNexusX/2024LiERLDDE/blob/main/env.py

Maintains the original Bay_Env interface and 5-tuple step() return contract.
The original C++ reset is NOT a full environment restart: make a new Bay_Env
object for each independent episode.
"""

import ctypes
from pathlib import Path

import numpy as np


STATE_NUMBER = 30
STATE_COLUMNS = 8
ACTION_DIM = 5


class Array2D(ctypes.Structure):
    _fields_ = [("data", (ctypes.c_double * STATE_COLUMNS) * STATE_NUMBER)]


class StepResult(ctypes.Structure):
    _fields_ = [
        ("state", Array2D),
        ("reward", ctypes.c_double),
        ("done", ctypes.c_bool),
    ]


def _check_abi():
    if ctypes.sizeof(Array2D) != 1920:
        raise RuntimeError(f"Unexpected Array2D size: {ctypes.sizeof(Array2D)}")
    if ctypes.sizeof(StepResult) != 1936:
        raise RuntimeError(f"Unexpected StepResult size: {ctypes.sizeof(StepResult)}")
    if StepResult.reward.offset != 1920 or StepResult.done.offset != 1928:
        raise RuntimeError("Incompatible C++/ctypes struct offsets")


class Bay_Env:
    def __init__(self, file):
        _check_abi()
        self.state_number = STATE_NUMBER
        self.observation_space = np.zeros((STATE_NUMBER, STATE_COLUMNS))
        self.action_space = ACTION_DIM
        self.Array2D = Array2D
        self.StepResult = StepResult
        self.env = None

        case_path = Path(file).expanduser().resolve()
        if not case_path.is_file():
            raise FileNotFoundError(f"ERL-DDE case does not exist: {case_path}")
        if case_path.stat().st_size == 0:
            raise ValueError(f"ERL-DDE case is empty: {case_path}")

        library_path = Path(__file__).resolve().parent / "libBACAP_env.so"
        if not library_path.is_file():
            raise FileNotFoundError(
                f"Linux C++ library missing: {library_path}. Run bash build_linux.sh"
            )
        self.dll = ctypes.CDLL(str(library_path))

        self.create_env = self.dll.create_env
        self.create_env.argtypes = [ctypes.c_char_p]
        self.create_env.restype = ctypes.c_void_p

        self.delete_env = self.dll.delete_env
        self.delete_env.argtypes = [ctypes.c_void_p]
        self.delete_env.restype = None

        self.step_func = self.dll.step
        self.step_func.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(StepResult),
        ]
        self.step_func.restype = None

        self.reset_func = self.dll.reset
        self.reset_func.argtypes = [ctypes.c_void_p, ctypes.POINTER(Array2D)]
        self.reset_func.restype = None

        self.env = self.create_env(str(case_path).encode("utf-8"))
        if not self.env:
            raise RuntimeError(f"C++ create_env returned null for {case_path}")

    def reset(self):
        if self.env is None:
            raise RuntimeError("Environment has been closed")
        arr = self.Array2D()
        self.reset_func(self.env, ctypes.byref(arr))
        return np.ctypeslib.as_array(arr.data).copy().reshape(STATE_NUMBER, STATE_COLUMNS)

    def step(self, action):
        if self.env is None:
            raise RuntimeError("Environment has been closed")
        values = np.asarray(action, dtype=np.float64).reshape(-1)
        if values.size != ACTION_DIM:
            raise ValueError(f"Expected {ACTION_DIM} action components, got {values.size}")
        if not np.all(np.isfinite(values)):
            raise ValueError("Non-finite action components")
        action_arr = (ctypes.c_double * ACTION_DIM)(*map(float, values))
        result = self.StepResult()
        self.step_func(self.env, action_arr, ctypes.byref(result))
        state = np.ctypeslib.as_array(result.state.data).copy().reshape(
            STATE_NUMBER, STATE_COLUMNS
        )
        return state, float(result.reward), bool(result.done), None, None

    def close(self):
        env = getattr(self, "env", None)
        if env:
            self.delete_env(env)
            self.env = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
