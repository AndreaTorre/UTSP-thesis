"""Strumentazione tempi per fase (per la tesi).
Solo stdlib a livello di modulo: nessun import di progetto, per evitare cicli.
"""
import time, json, functools

_PHASE_TIMES = {}

def _add(name, seconds):
    _PHASE_TIMES[name] = _PHASE_TIMES.get(name, 0.0) + float(seconds)

def timed(name):
    def deco(fn):
        @functools.wraps(fn)
        def wrap(*a, **k):
            t = time.time()
            try:
                return fn(*a, **k)
            finally:
                _add(name, time.time() - t)
        return wrap
    return deco

def ls_add(label, seconds):
    _add("ls_train" if "train" in str(label) else "ls_test", seconds)

def timing_dump(subdir="report", filename="phase_times.json"):
    from common import out_path  # lazy: evita import circolare
    try:
        with open(out_path(filename, subdir), "w") as f:
            json.dump(_PHASE_TIMES, f, indent=2)
        print(f"  [tempi per fase] {_PHASE_TIMES}")
    except Exception as e:
        print(f"  [tempi per fase] dump saltato: {e}")
    _PHASE_TIMES.clear()
