"""Call tracer for the XG-NID pipeline (debug / study aid, not part of training).

Records, in execution order, every Python function that runs inside the given
directories (by default the GNN4ID project), with a compact summary of its
arguments and of its return value, so you can read *what runs after what*
without stepping through the debugger by hand.

Two ways to use it, both provided by Debug/trace_pipeline.py:

  1. VS Code debugger (F5)  -> breakpoints, step in/out. The tracer is turned
     OFF automatically in that case (pydevd owns sys.settrace; a second tracer
     would fight with it).
  2. Plain run              -> the tracer writes a *.trace log next to the run
     log, plus two summary tables:
        - functions in order of first call  (the "step by step" view)
        - functions by call count / time    (where the time actually goes)

Implementation notes
--------------------
* Uses sys.setprofile (not settrace): one event per call/return instead of one
  per line, so the overhead is ~2-4x instead of ~100x, and it does not clash
  with a line-level debugger as badly.
* Only frames whose file lives under one of `include_dirs` are logged, so
  pandas/torch internals do not drown the log. Add nfstream / torch_geometric
  with --trace-deps when you want to see inside them.
* Each function is logged in full only for its first `max_calls_per_func`
  calls; after that it is only counted. That keeps the log readable when a
  function runs 20 000 times (e.g. My_Custom.on_update, one call per packet).
"""

import os
import sys
import threading
import time
from collections import OrderedDict

_MAX_STR = 48
_SELF_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep


# ---------------------------------------------------------------------------
# value / argument summaries (duck-typed: never imports pandas / torch itself)
# ---------------------------------------------------------------------------
def summarize(v):
    """Short, allocation-free-ish description of a value for the trace log."""
    try:
        if v is None or isinstance(v, (bool, int, float)):
            return repr(v)
        if isinstance(v, str):
            return repr(v if len(v) <= _MAX_STR else v[:_MAX_STR] + "...")
        if isinstance(v, (bytes, bytearray)):
            return "%s(len=%d)" % (type(v).__name__, len(v))

        t = type(v)
        name = t.__name__
        mod = getattr(t, "__module__", "") or ""

        if mod.startswith("pandas"):
            if name == "DataFrame":
                return "DataFrame(shape=%s)" % (tuple(v.shape),)
            if name == "Series":
                return "Series(len=%d)" % len(v)
        if mod.startswith("numpy") and name == "ndarray":
            return "ndarray(shape=%s, dtype=%s)" % (tuple(v.shape), v.dtype)
        if mod.startswith("torch") and name in ("Tensor", "Parameter"):
            return "Tensor(shape=%s, dtype=%s)" % (tuple(v.shape), str(v.dtype).replace("torch.", ""))
        if name in ("HeteroData", "Data", "Batch", "DataLoader"):
            return name
        if isinstance(v, dict):
            return "dict(len=%d)" % len(v)
        if isinstance(v, (list, tuple, set, frozenset)):
            return "%s(len=%d)" % (name, len(v))
        if mod.startswith("nfstream"):
            return name
        return "<%s>" % name
    except Exception:
        return "<?>"


def _qualname(frame):
    """Best-effort Class.method name (co_qualname only exists on py3.11+)."""
    code = frame.f_code
    fn = code.co_name
    qual = getattr(code, "co_qualname", None)
    if qual:
        return qual.replace("<module>", "module")
    loc = frame.f_locals
    obj = loc.get("self")
    if obj is not None:
        try:
            return "%s.%s" % (type(obj).__name__, fn)
        except Exception:
            pass
    cls = loc.get("cls")
    if cls is not None and isinstance(cls, type):
        return "%s.%s" % (cls.__name__, fn)
    return fn


def _args_of(frame, max_args=6):
    code = frame.f_code
    names = code.co_varnames[:code.co_argcount]
    loc = frame.f_locals
    parts = []
    for nm in names:
        if nm in ("self", "cls"):
            continue
        if nm in loc:
            parts.append("%s=%s" % (nm, summarize(loc[nm])))
        if len(parts) >= max_args:
            parts.append("...")
            break
    return ", ".join(parts)


class _Local(threading.local):
    def __init__(self):
        self.stack = []          # frames we decided to log/count
        self.depth = 0


class CallTracer:
    """sys.setprofile based call/return recorder. Use as a context manager."""

    def __init__(self, include_dirs, out=None, echo=False,
                 max_calls_per_func=3, max_depth=40, log_returns=True,
                 skip_names=("<listcomp>", "<genexpr>", "<lambda>", "<dictcomp>")):
        self.include_dirs = tuple(os.path.realpath(p) + os.sep for p in include_dirs)
        self.out = out
        self.echo = echo
        self.max_calls_per_func = max_calls_per_func
        self.max_depth = max_depth
        self.log_returns = log_returns
        self.skip_names = set(skip_names)

        self._file_ok = {}        # filename -> bool (cache, realpath is slow)
        self._stats = OrderedDict()  # key -> dict(calls, time, file, line, name)
        self._local = _Local()
        self._n = 0
        self._t0 = None
        self._active = False

    # -- plumbing ----------------------------------------------------------
    def _keep_file(self, filename):
        ok = self._file_ok.get(filename)
        if ok is None:
            if filename.startswith("<"):
                ok = False           # <frozen importlib._bootstrap>, <string>, ...
            else:
                try:
                    rp = os.path.realpath(filename)
                except Exception:
                    rp = filename
                # Debug/ is this scaffolding, not the pipeline: never trace it.
                ok = rp.startswith(self.include_dirs) and not rp.startswith(_SELF_DIR)
            self._file_ok[filename] = ok
        return ok

    def note(self, msg):
        """Write a banner into the trace log (used for stage boundaries)."""
        self._emit("")
        self._emit("### %s" % msg)

    def _emit(self, line):
        if self.out is not None:
            self.out.write(line + "\n")
        if self.echo:
            sys.__stdout__.write(line + "\n")

    def _profile(self, frame, event, arg):
        if event == "call":
            code = frame.f_code
            if code.co_name in self.skip_names or not self._keep_file(code.co_filename):
                return
            key = (code.co_filename, code.co_firstlineno)
            st = self._stats.get(key)
            if st is None:
                st = {"calls": 0, "time": 0.0, "name": _qualname(frame),
                      "file": code.co_filename, "line": code.co_firstlineno,
                      "order": len(self._stats) + 1}
                self._stats[key] = st
            st["calls"] += 1
            loc = self._local
            loc.stack.append((frame, key, time.perf_counter()))
            if st["calls"] <= self.max_calls_per_func and loc.depth <= self.max_depth:
                self._n += 1
                self._emit("[%05d] t=%7.3fs %s-> %s(%s)   %s:%d" % (
                    self._n, time.perf_counter() - self._t0, "| " * loc.depth,
                    st["name"], _args_of(frame),
                    os.path.relpath(code.co_filename, self.include_dirs[0].rstrip(os.sep))
                    if code.co_filename.startswith(self.include_dirs[0]) else code.co_filename,
                    frame.f_lineno))
            elif st["calls"] == self.max_calls_per_func + 1:
                self._emit("[     ] %s.. %s called again; further calls only counted"
                           % ("| " * loc.depth, st["name"]))
            loc.depth += 1

        elif event == "return":
            loc = self._local
            if not loc.stack or loc.stack[-1][0] is not frame:
                return
            _f, key, t_start = loc.stack.pop()
            loc.depth -= 1
            st = self._stats[key]
            dt = time.perf_counter() - t_start
            st["time"] += dt
            if self.log_returns and st["calls"] <= self.max_calls_per_func \
                    and loc.depth <= self.max_depth:
                self._n += 1
                self._emit("[%05d] t=%7.3fs %s<- %s = %s   (%.1f ms)" % (
                    self._n, time.perf_counter() - self._t0, "| " * loc.depth,
                    st["name"], summarize(arg), dt * 1000.0))

    # -- api ---------------------------------------------------------------
    def start(self):
        if sys.gettrace() is not None:
            self._emit("[tracer] a debugger owns sys.settrace -> auto-trace disabled, "
                       "use breakpoints instead")
            return self
        self._t0 = time.perf_counter()
        self._active = True
        threading.setprofile(self._profile)
        sys.setprofile(self._profile)
        return self

    def stop(self):
        if not self._active:
            return
        sys.setprofile(None)
        threading.setprofile(None)
        self._active = False

    def summary(self, top=40):
        """Two tables: order of first call, and the hot functions."""
        if not self._stats:
            return "[tracer] nothing recorded"
        lines = []
        lines.append("")
        lines.append("=" * 100)
        lines.append("FUNCTIONS IN ORDER OF FIRST CALL  (this is the pipeline's step-by-step)")
        lines.append("=" * 100)
        lines.append("%4s  %-52s %8s %10s  %s" % ("#", "function", "calls", "total_s", "file:line"))
        for st in sorted(self._stats.values(), key=lambda s: s["order"]):
            lines.append("%4d  %-52s %8d %10.3f  %s:%d" % (
                st["order"], st["name"][:52], st["calls"], st["time"],
                os.path.basename(st["file"]), st["line"]))
        lines.append("")
        lines.append("=" * 100)
        lines.append("HOTTEST FUNCTIONS (cumulative wall time, includes callees)")
        lines.append("=" * 100)
        hot = sorted(self._stats.values(), key=lambda s: -s["time"])[:top]
        for st in hot:
            lines.append("%10.3fs %8d calls  %-52s %s:%d" % (
                st["time"], st["calls"], st["name"][:52],
                os.path.basename(st["file"]), st["line"]))
        return "\n".join(lines)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False


class Tee:
    """Duplicate a stream into a file, so console output lands in the run log."""

    def __init__(self, stream, fh):
        self.stream = stream
        self.fh = fh

    def write(self, data):
        self.stream.write(data)
        try:
            self.fh.write(data)
        except Exception:
            pass
        return len(data)

    def flush(self):
        self.stream.flush()
        try:
            self.fh.flush()
        except Exception:
            pass

    def isatty(self):
        return getattr(self.stream, "isatty", lambda: False)()

    def fileno(self):
        return self.stream.fileno()
