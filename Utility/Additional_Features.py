"""
Explainable Feature Extractor for XG-NID (paper Section 3.1.2 / Algorithm 1 / Table 1).

Reproduces the authors' upstream implementation (GNN4ID commit 551d1f1) exactly
by default -- 28 rolling-window columns computed per destination IP and per
(src_ip, dst_ip) pair over the previous ``window`` *flows* (rows sorted by
``bidirectional_first_seen_ms``), plus ``packet_size_variation`` and the 7
one-hot columns -- and adds three things the original lacked:

* ``window_unit``: ``'flows'`` (authors: a count of flow rows, default 350),
  ``'time'`` (a time window such as ``'60s'``, the semantics of paper
  Algorithm 1 "rolling time window W"), or ``'packets'`` (the smallest suffix
  of previous flows whose ``bidirectional_packets`` add up to at least N).
* ``count_mode``: ``'author'`` keeps the authors' mixed counting (UDP/TCP/ICMP/
  HTTP/DNS/vulnerable-port columns count *flows*, flag columns count
  *packets*); ``'packets'`` weights the per-flow booleans by
  ``bidirectional_packets`` so every "count" column counts packets as Table 1
  says; ``'flows'`` turns the flag columns into flow counts.
* ``schema``: ``'author82'`` (28 columns, Drive/checkpoint compatible) or
  ``'table1'`` (the 15 destination-grouped columns renamed to paper Table 1 +
  ``packet_size_variation``).

Where this sits in the pipeline (and why that is leak-free)
-----------------------------------------------------------
Run on the FULL, time-ordered raw NFStream CSV of a pcap, BEFORE the train/test
split.  Every feature at row i is a function of rows j <= i (same file, same
destination or pair); no label is used and no future row is used.  A temporal
split (test = later flows) therefore cannot leak: a test row only depends on
its own past, which is exactly what a deployed sensor sees, and no train row
depends on any test row.  Computing the features AFTER the split -- as the
June-2026 revision did -- is what breaks things: undersampling the train pool
first thins the sequence, so a 350-row window spans far more real traffic in
train than in test and the same column gets two different units, and every
per-split file starts with an artificial empty history.
"""

import os
import warnings

import numpy as np
import pandas as pd
from pandas.api.indexers import BaseIndexer

from Utility.Schema import (
    ROLLING_28, TABLE1_MAP, TIME_COL, ONEHOT_7,
    DEFAULT_HTTP_PORTS, DEFAULT_DNS_PORTS, DEFAULT_VULNERABLE_PORTS,
)

__all__ = ["additional_features", "compute_rolling_features", "ROLLING_SPEC"]

# (column, grouping, input, aggregation) -- the authors' order and inputs.
# grouping: 'dst' = per dst_ip, 'pair' = per src_ip-dst_ip pair.
ROLLING_SPEC = [
    ("Rolling_UDP_Requests_SourceDestination", "pair", "is_udp", "sum"),
    ("Rolling_UDP_Requests_Destination", "dst", "is_udp", "sum"),
    ("Rolling_TCP_Requests_SourceDestination", "pair", "is_tcp", "sum"),
    ("Rolling_TCP_Requests_Destination", "dst", "is_tcp", "sum"),
    ("Rolling_ACK_Packets_SourceDestination", "pair", "ack", "sum"),
    ("Rolling_ACK_Packets_Destination", "dst", "ack", "sum"),
    ("Rolling_FIN_Packets_SourceDestination", "pair", "fin", "sum"),
    ("Rolling_FIN_Packets_Destination", "dst", "fin", "sum"),
    ("Rolling_rst_Packets_SourceDestination", "pair", "rst", "sum"),
    ("Rolling_rst_Packets_Destination", "dst", "rst", "sum"),
    ("Rolling_psh_Packets_SourceDestination", "pair", "psh", "sum"),
    ("Rolling_psh_Packets_Destination", "dst", "psh", "sum"),
    ("Rolling_SYN_Packets_SourceDestination", "pair", "syn", "sum"),
    ("Rolling_SYN_Packets_Destination", "dst", "syn", "sum"),
    # authors use dst_port here although the paper text says "source ports"
    ("Unique_Ports_In_SourceDestinationIP", "pair", "dst_port", "nunique"),
    ("Rolling_ICMP_Requests_SourceDestination", "pair", "is_icmp", "sum"),
    ("Rolling_ICMP_Requests_Destination", "dst", "is_icmp", "sum"),
    ("Rolling_http_port_SourceDestination", "pair", "is_http_dst", "sum"),
    ("Rolling_http_port_Destination", "dst", "is_http_dst", "sum"),
    ("Rolling_Duration_Destination", "dst", "duration", "mean"),
    ("Rolling_Duration_SourceDestination", "pair", "duration", "mean"),
    ("Rolling_DNS_request_SourceDestination", "pair", "is_dns_dst", "sum"),
    ("Rolling_DNS_request_Destination", "dst", "is_dns_dst", "sum"),
    ("Rolling_DNS_request_SourceDestination2", "pair", "is_dns_src", "sum"),
    ("Rolling_DNS_request_Destination2", "dst", "is_dns_src", "sum"),
    # authors group the vulnerable-port count by PAIR (paper says destination)
    ("Rolling_vulnerable_port", "pair", "is_vuln_dst", "sum"),
    ("Rolling_packets_destination", "dst", "src2dst_packets", "sum"),
    ("Rolling_bipackets_destination", "dst", "bidirectional_packets", "sum"),
]
assert [c for c, _, _, _ in ROLLING_SPEC] == ROLLING_28

_FLAG_INPUTS = ("ack", "fin", "rst", "psh", "syn")
_BOOL_INPUTS = ("is_udp", "is_tcp", "is_icmp", "is_http_dst", "is_dns_dst",
                "is_dns_src", "is_vuln_dst")

_REQUIRED = [TIME_COL, "src_ip", "dst_ip", "src_port", "dst_port", "protocol",
             "expiration_id", "bidirectional_packets", "src2dst_packets",
             "bidirectional_duration_ms", "bidirectional_ack_packets",
             "bidirectional_fin_packets", "bidirectional_rst_packets",
             "bidirectional_psh_packets", "bidirectional_syn_packets",
             "src2dst_min_ps", "src2dst_max_ps", "dst2src_min_ps", "dst2src_max_ps"]


# ---------------------------------------------------------------------------
# window engine
# ---------------------------------------------------------------------------
class _Bounds(BaseIndexer):
    """Precomputed [start, end) bounds per row; pandas runs its own kernels."""

    def get_window_bounds(self, num_values=0, min_periods=None, center=None,
                          closed=None, step=None):
        return self.start, self.end


def _parse_window(window, window_unit):
    if window_unit == "flows":
        if isinstance(window, str):
            raise ValueError("window_unit='flows' needs an int window (rows), got %r" % (window,))
        w = int(window)
        if w < 1:
            raise ValueError("window must be >= 1")
        return w
    if window_unit == "packets":
        if isinstance(window, str):
            raise ValueError("window_unit='packets' needs an int window (packets), got %r" % (window,))
        n = int(window)
        if n < 1:
            raise ValueError("window must be >= 1")
        return n
    if window_unit == "time":
        if isinstance(window, str):
            ms = int(pd.Timedelta(window).total_seconds() * 1000)
        else:
            ms = int(window) * 1000  # a bare number is taken as seconds
        if ms <= 0:
            raise ValueError("time window must be > 0")
        return ms
    raise ValueError("window_unit must be 'flows', 'time' or 'packets', got %r" % (window_unit,))


def _group_order(gid):
    """Order rows so groups are contiguous and time order is kept inside a group."""
    n = len(gid)
    order = np.lexsort((np.arange(n), gid))
    g = gid[order]
    first = np.r_[0, np.flatnonzero(np.diff(g)) + 1]
    sizes = np.diff(np.r_[first, n])
    gstart = np.repeat(first, sizes)
    return order, gstart, first


def _window_bounds(gstart, first, t_ms, pkts, window_unit, window, include_current):
    """[start, end) per row (arrays are already in group order)."""
    n = len(gstart)
    pos = np.arange(n)
    end = pos + 1 if include_current else pos
    if window_unit == "flows":
        start = np.maximum(gstart, pos - int(window) + 1)
    elif window_unit == "time":
        W = int(window)
        start = np.empty(n, dtype=np.int64)
        bounds = np.r_[first, n]
        for g0, g1 in zip(bounds[:-1], bounds[1:]):
            t = t_ms[g0:g1]
            # pandas offset semantics: window (t_i - W, t_i]
            start[g0:g1] = g0 + np.searchsorted(t, t - W, side="right")
    else:  # packets
        N = int(window)
        start = np.empty(n, dtype=np.int64)
        bounds = np.r_[first, n]
        for g0, g1 in zip(bounds[:-1], bounds[1:]):
            cc = np.r_[0, np.cumsum(pkts[g0:g1])]
            i = np.arange(g1 - g0)
            # largest j with cc[j] <= cc[i+1] - N  -> window [j, i] holds >= N packets
            j = np.searchsorted(cc, cc[i + 1] - N, side="right") - 1
            start[g0:g1] = g0 + np.clip(j, 0, i)
    start = np.minimum(start, end)
    return start.astype(np.int64), end.astype(np.int64)


def _roll(values, start, end, how):
    r = pd.Series(values).rolling(_Bounds(start=start, end=end), min_periods=1)
    if how == "sum":
        out = r.sum()
    elif how == "mean":
        out = r.mean()
    elif how == "nunique":
        out = r.apply(lambda a: len(set(a)), raw=True)
    else:
        raise ValueError(how)
    return out.to_numpy()


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def compute_rolling_features(df, window=350, window_unit="flows", count_mode="author",
                             schema="author82", include_current=True,
                             http_ports=DEFAULT_HTTP_PORTS,
                             vulnerable_ports=DEFAULT_VULNERABLE_PORTS,
                             dns_ports=DEFAULT_DNS_PORTS,
                             exp_id=(0, -1), proto_list=(1, 2, 6, 17, 58), onehot=True):
    """Return a copy of ``df`` (sorted by time, like the authors' output) with
    ``packet_size_variation``, the 28 rolling columns and the 7 one-hot columns
    appended in the authors' order.  Pure, in-memory."""
    if count_mode not in ("author", "packets", "flows"):
        raise ValueError("count_mode must be 'author', 'packets' or 'flows'")
    if schema not in ("author82", "table1"):
        raise ValueError("schema must be 'author82' or 'table1'")
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise KeyError("CSV lacks columns needed for the rolling features: %s" % missing)
    W = _parse_window(window, window_unit)

    # Same sort call as the authors (pandas default kind) so that, given the same
    # raw CSV, tied timestamps are ordered identically and the output is
    # bit-for-bit the Drive CSV.  A stable sort would differ on ties.
    data = df.sort_values(by=TIME_COL).reset_index(drop=True)
    n = len(data)

    # ---- inputs -------------------------------------------------------------
    proto = data["protocol"].to_numpy()
    dst_port = data["dst_port"].to_numpy()
    src_port = data["src_port"].to_numpy()
    pkts = data["bidirectional_packets"].to_numpy(dtype=np.float64)
    inputs = {
        "is_udp": (proto == 17).astype(np.float64),
        "is_tcp": (proto == 6).astype(np.float64),
        "is_icmp": (proto == 1).astype(np.float64),
        "is_http_dst": np.isin(dst_port, list(http_ports)).astype(np.float64),
        "is_dns_dst": np.isin(dst_port, list(dns_ports)).astype(np.float64),
        "is_dns_src": np.isin(src_port, list(dns_ports)).astype(np.float64),
        "is_vuln_dst": np.isin(dst_port, list(vulnerable_ports)).astype(np.float64),
        "ack": data["bidirectional_ack_packets"].to_numpy(dtype=np.float64),
        "fin": data["bidirectional_fin_packets"].to_numpy(dtype=np.float64),
        "rst": data["bidirectional_rst_packets"].to_numpy(dtype=np.float64),
        "psh": data["bidirectional_psh_packets"].to_numpy(dtype=np.float64),
        "syn": data["bidirectional_syn_packets"].to_numpy(dtype=np.float64),
        "duration": data["bidirectional_duration_ms"].to_numpy(dtype=np.float64),
        "src2dst_packets": data["src2dst_packets"].to_numpy(dtype=np.float64),
        "bidirectional_packets": pkts,
        "dst_port": dst_port.astype(np.float64),
    }
    if count_mode == "packets":
        for k in _BOOL_INPUTS:
            inputs[k] = inputs[k] * pkts
    elif count_mode == "flows":
        for k in _FLAG_INPUTS:
            inputs[k] = (inputs[k] > 0).astype(np.float64)

    # ---- grouping keys ---------------------------------------------------------
    t_ms = data[TIME_COL].to_numpy(dtype=np.int64)
    gid = {
        "dst": pd.factorize(data["dst_ip"].astype(str))[0],
        "pair": pd.factorize(data["src_ip"].astype(str) + "-" + data["dst_ip"].astype(str))[0],
    }
    order, inv, bounds = {}, {}, {}
    for g in ("dst", "pair"):
        o, gstart, first = _group_order(gid[g])
        order[g] = o
        inv[g] = np.empty_like(o)
        inv[g][o] = np.arange(n)
        bounds[g] = _window_bounds(gstart, first, t_ms[o], pkts[o], window_unit, W, include_current)

    # ---- features, in the authors' assignment order ---------------------------
    data["packet_size_variation"] = data[
        ["src2dst_min_ps", "src2dst_max_ps", "dst2src_min_ps", "dst2src_max_ps"]
    ].std(axis=1)
    for col, g, src, how in ROLLING_SPEC:
        vals = _roll(inputs[src][order[g]], bounds[g][0], bounds[g][1], how)
        if not include_current:
            vals = np.nan_to_num(vals, nan=0.0)
        data[col] = vals[inv[g]]

    if onehot:
        data["expiration_id"] = pd.Categorical(data["expiration_id"], categories=list(exp_id))
        data["protocol"] = pd.Categorical(data["protocol"], categories=list(proto_list))
        data = pd.get_dummies(data, prefix=["Exp", "proto"],
                              columns=["expiration_id", "protocol"], dtype=int)

    if schema == "table1":
        drop = [c for c in ROLLING_28 if c not in TABLE1_MAP]
        data = data.drop(columns=drop).rename(columns=TABLE1_MAP)
    return data


def _already_enriched(columns):
    cols = set(columns)
    return bool(cols & set(ROLLING_28)) or bool(cols & set(TABLE1_MAP.values())) \
        or "proto_1" in cols


def _undo_enrichment(data, exp_id=(0, -1), proto_list=(1, 2, 6, 17, 58)):
    """Reverse a previous run so the features can be recomputed (force=True)."""
    data = data.copy()
    if "protocol" not in data.columns:
        pcols = [c for c in data.columns if c.startswith("proto_")]
        if pcols:
            codes = np.array([int(c.split("_", 1)[1]) for c in pcols])
            data["protocol"] = codes[data[pcols].to_numpy().argmax(axis=1)]
            data = data.drop(columns=pcols)
    if "expiration_id" not in data.columns:
        ecols = [c for c in data.columns if c.startswith("Exp_")]
        if ecols:
            codes = np.array([int(c.split("_", 1)[1]) for c in ecols])
            data["expiration_id"] = codes[data[ecols].to_numpy().argmax(axis=1)]
            data = data.drop(columns=ecols)
    gone = [c for c in list(ROLLING_28) + list(TABLE1_MAP.values()) + ["packet_size_variation"]
            if c in data.columns]
    return data.drop(columns=gone)


def additional_features(file_name, window=350, window_unit="flows", count_mode="author",
                        schema="author82", out_file=None, force=False, include_current=True,
                        http_ports=None, vulnerable_ports=None, dns_ports=None,
                        exp_id=(0, -1), proto_list=(1, 2, 6, 17, 58),
                        window_size=None, **legacy_kwargs):
    """File wrapper around :func:`compute_rolling_features`.

    Reads the NFStream CSV ``file_name`` and writes the enriched CSV to
    ``out_file`` (recommended) or, when ``out_file`` is None, back onto
    ``file_name`` (the authors' in-place behaviour; a warning is printed).

    ``window_size`` is accepted as a deprecated alias of ``window``.  Returns the
    output path, or '' on a read error (authors' convention).  If the file was
    already enriched the call is a no-op unless ``force=True``.
    """
    if window_size is not None:
        window = window_size
    if legacy_kwargs:
        warnings.warn("additional_features: ignoring unknown kwargs %s" % sorted(legacy_kwargs))
    http_ports = DEFAULT_HTTP_PORTS if http_ports is None else http_ports
    vulnerable_ports = DEFAULT_VULNERABLE_PORTS if vulnerable_ports is None else vulnerable_ports
    dns_ports = DEFAULT_DNS_PORTS if dns_ports is None else dns_ports

    try:
        data = pd.read_csv(file_name, low_memory=False)
    except Exception as exc:
        print("file reading error: %s (%s)" % (file_name, exc))
        return ""
    if TIME_COL not in data.columns:
        print(" CSV does not contain %s for initial sorting: %s" % (TIME_COL, file_name))
        return ""

    target = out_file or file_name
    if _already_enriched(data.columns):
        if not force:
            warnings.warn("%s already contains rolling/one-hot columns; skipping "
                          "(use force=True to recompute)" % os.path.basename(file_name))
            if target != file_name:
                os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
                data.to_csv(target, index=False)
            return target
        data = _undo_enrichment(data, exp_id, proto_list)

    data = compute_rolling_features(
        data, window=window, window_unit=window_unit, count_mode=count_mode,
        schema=schema, include_current=include_current, http_ports=http_ports,
        vulnerable_ports=vulnerable_ports, dns_ports=dns_ports,
        exp_id=exp_id, proto_list=proto_list, onehot=True)

    if out_file is None:
        warnings.warn("additional_features: overwriting %s in place; pass out_file= "
                      "to keep the raw CSV" % os.path.basename(file_name))
    else:
        os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
    data.to_csv(target, index=False)
    return target
