"""Auto-break: dừng debugger ngay khi vào từng hàm chính của pipeline,
KHÔNG cần tự đặt breakpoint nào. Bật bằng `--break-at all` (xem .vscode/launch.json).

Hai cơ chế, tự chọn:

1. **pydevd** (khi đang chạy dưới VS Code / debugpy): đăng ký breakpoint thật ở
   dòng đầu tiên trong thân mỗi hàm qua `PyDevdAPI.add_breakpoint` — đúng API mà
   debugpy dùng khi bạn click chuột vào lề trái. Debugger dừng **bên trong** hàm,
   F5 để đi tiếp sang hàm kế. `--break-max-hits N` thành hit condition
   `@HIT@ <= N` nên `on_update` (hàng chục nghìn lần gọi) chỉ dừng N lần đầu.
   Lưu ý: nếu trong lúc debug bạn tự thêm/bớt breakpoint trong CÙNG file, VS Code
   gửi lại `setBreakpoints` cho file đó và pydevd xoá hết breakpoint cũ của file
   (kể cả của auto-break) -> restart phiên debug để đăng ký lại.

2. **wrapper** (khi không có debugger, hoặc pydevd đổi API): bọc hàm lại, mỗi lần
   vào thì in ra tên hàm + tham số. Có `XGNID_BREAK_PDB=1` thì dừng bằng pdb.
   Wrapper phải được cài TRƯỚC khi stage làm `from Utility.Functions import ...`,
   nên `trace_pipeline.main()` gọi install() ngay đầu.
"""

import dis
import functools
import importlib
import itertools
import os

# (stage, "module:hàm" | "module:Class.method") - theo đúng thứ tự chạy của pipeline
TARGETS = [
    ("extract",  "Utility.Feature_extractor_flow_packet_combined:My_Custom.on_init"),
    ("extract",  "Utility.Feature_extractor_flow_packet_combined:My_Custom.on_update"),
    ("split",    "Utility.Functions:split_csv"),
    ("features", "Utility.Additional_Features:additional_features"),
    ("features", "Utility.Additional_Features:_rolling_sum"),
    ("features", "Utility.Additional_Features:_rolling_mean"),
    ("features", "Utility.Additional_Features:_rolling_unique"),
    ("combine",  "Utility.Functions:Combining_classes"),
    ("graphs",   "Utility.Functions:NIDSDataset.process"),
    ("graphs",   "Utility.Functions:NIDSDataset._get_flow_node_features"),
    ("graphs",   "Utility.Functions:NIDSDataset._get_packet_node_features"),
    ("graphs",   "Utility.Functions:NIDSDataset._get_contain_edge_index"),
    ("graphs",   "Utility.Functions:NIDSDataset._get_link_edge_index"),
    ("graphs",   "Utility.Functions:NIDSDataset._get_contain_edge_features"),
    ("graphs",   "Utility.Functions:NIDSDataset._get_link_edge_features"),
    ("graphs",   "Utility.Functions:NIDSDataset.get"),
    ("model",    "Utility.Model:HeteroGNN.forward"),
    ("model",    "Utility.Training:train"),
    ("model",    "Utility.Training:test"),
    ("model",    "Utility.Training:test_cm"),
    ("model",    "Utility.Training:calculate_metrics"),
]


def _resolve(stages, names, log):
    """TARGETS -> [(spec, owner, attr, function)] cho các stage/tên được chọn."""
    out = []
    for stage, spec in TARGETS:
        if stages and stage not in stages:
            continue
        if names and not any(n.lower() in spec.lower() for n in names):
            continue
        mod_name, dotted = spec.split(":")
        try:
            mod = importlib.import_module(mod_name)
        except Exception as exc:
            log("[auto-break] bỏ qua %s (import lỗi: %s)" % (spec, exc))
            continue
        if "." in dotted:
            cls_name, attr = dotted.split(".", 1)
            owner = getattr(mod, cls_name, None)
        else:
            owner, attr = mod, dotted
        fn = getattr(owner, attr, None) if owner is not None else None
        if fn is None:
            log("[auto-break] không thấy %s" % spec)
            continue
        out.append((spec, owner, attr, fn))
    return out


def _first_body_line(code):
    """Dòng thực thi đầu tiên trong thân hàm (bỏ qua dòng `def` và docstring)."""
    lines = sorted({ln for _, ln in dis.findlinestarts(code)
                    if ln and ln > code.co_firstlineno})
    return lines[0] if lines else code.co_firstlineno


def _install_pydevd(resolved, max_hits, log):
    import pydevd
    from _pydevd_bundle.pydevd_api import PyDevdAPI

    py_db = pydevd.get_global_debugger()
    if py_db is None:
        raise RuntimeError("chưa có debugger nào attach")

    api = PyDevdAPI()
    hit_condition = "@HIT@ <= %d" % max_hits if max_hits and max_hits > 0 else None
    suspend_policy = "ALL" if getattr(py_db, "multi_threads_single_notification", False) else "NONE"
    done = []
    for i, (spec, _owner, _attr, fn) in enumerate(resolved):
        code = getattr(fn, "__code__", None)
        if code is None:
            log("[auto-break] %s không có __code__, bỏ qua" % spec)
            continue
        line = _first_body_line(code)
        api.add_breakpoint(py_db, code.co_filename, "python-line", 990000 + i, line,
                           None,             # condition
                           "None",           # func_name (pydevd dùng chuỗi "None")
                           None,             # expression
                           suspend_policy,
                           hit_condition,
                           False,            # is_logpoint
                           adjust_line=True)
        done.append("%s  ->  %s:%d" % (spec, os.path.basename(code.co_filename), line))
    if not done:
        raise RuntimeError("không đăng ký được breakpoint nào")
    return done


# trạng thái của chế độ pause="input" (người dùng gõ q để tắt, s để bỏ qua 1 hàm)
_STATE = {"off": False, "skip": set()}


def _suspend_fallback(label, pause, log):
    """Dừng khi KHÔNG có debugger. pause: 'off' | 'input' | 'pdb'."""
    if pause == "pdb" or os.environ.get("XGNID_BREAK_PDB"):
        import pdb
        pdb.set_trace()
        return
    if pause != "input" or _STATE["off"] or label in _STATE["skip"]:
        return
    try:
        ans = input("      [Enter]=chạy tiếp  s=bỏ qua hàm này  q=tắt auto-break > ").strip().lower()
    except Exception as exc:          # nbclient / stdin bị chặn -> không dừng được
        _STATE["off"] = True
        log("      (không đọc được stdin: %s -> tắt chế độ chờ)" % type(exc).__name__)
        return
    if ans == "q":
        _STATE["off"] = True
        log("      -> tắt auto-break, chạy thẳng đến hết")
    elif ans == "s":
        _STATE["skip"].add(label)
        log("      -> bỏ qua các lần gọi sau của hàm này")


def _wrap(label, fn, max_hits, log, pause):
    counter = itertools.count(1)

    @functools.wraps(fn)
    def wrapper(*a, **kw):
        n = next(counter)
        if max_hits <= 0 or n <= max_hits:
            from Debug.tracer import summarize
            skip_self = 1 if (a and hasattr(type(a[0]), fn.__name__)) else 0
            args = ", ".join(summarize(x) for x in a[skip_self:skip_self + 4])
            log("[auto-break] %s   hit %d   args(%s)" % (label, n, args))
            _suspend_fallback(label, pause, log)
        return fn(*a, **kw)

    wrapper.__xgnid_autobreak__ = True
    return wrapper


def _install_wrappers(resolved, max_hits, log, pause):
    done = []
    for spec, owner, attr, fn in resolved:
        if getattr(fn, "__xgnid_autobreak__", False):
            continue
        setattr(owner, attr, _wrap(spec, fn, max_hits, log, pause))
        done.append(spec)
    return done


def install(stages=None, names=None, max_hits=1, log=print, mode="auto", pause="off"):
    """Cài auto-break. Trả về (danh sách đã cài, mode thực tế dùng).

    pause chỉ có tác dụng khi KHÔNG có debugger (Run Cell thường / chạy terminal):
      'off'   - chỉ in tên hàm rồi chạy tiếp
      'input' - dừng lại chờ bấm Enter mới chạy tiếp  <-- "chỉ chạy khi tôi cho phép"
      'pdb'   - mở prompt pdb ngay tại chỗ
    Khi có debugger (Debug Cell / F5) thì luôn dùng breakpoint thật của pydevd.
    """
    _STATE["off"] = False
    _STATE["skip"] = set()
    resolved = _resolve(stages, names, log)
    if not resolved:
        return [], "none"
    if mode in ("auto", "pydevd"):
        try:
            return _install_pydevd(resolved, max_hits, log), "pydevd"
        except Exception as exc:
            if mode == "pydevd":
                raise
            log("[auto-break] pydevd breakpoint không dùng được (%s)" % exc)
            log("[auto-break] -> chuyển sang wrapper (pause=%r)" % pause)
    return _install_wrappers(resolved, max_hits, log, pause), "wrapper-" + pause
