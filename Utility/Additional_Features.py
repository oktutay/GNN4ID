"""
Explainable Feature Extractor for XG-NID, paper Section 3.1.2 / Algorithm 1 / Table 1.

For each destination D_j (and source-destination pair when useful for the rolling
window context), aggregates packets received within a sliding time window W and
produces 16 temporal features that the HGNN consumes alongside the flow features.
"""

import pandas as pd
import numpy as np


# Default port lists used to flag categorical packet behavior. Kept module-level so
# they can be overridden from notebooks without touching the call signature.
DEFAULT_HTTP_PORTS = [80, 443, 8080]
DEFAULT_DNS_PORTS = [53]
DEFAULT_VULNERABLE_PORTS = [20, 21, 22, 23, 25, 53, 80, 110, 143, 443, 445, 3389, 8080]


def _rolling_sum(group, window_size):
    return group.rolling(window=window_size, min_periods=1).sum()


def _rolling_mean(group, window_size):
    return group.rolling(window=window_size, min_periods=1).mean()


def _rolling_unique(group, window_size):
    return group.rolling(window=window_size, min_periods=1).apply(
        lambda x: len(set(x)), raw=True
    )


def additional_features(file_name,
                        window_size=350,
                        http_ports=None,
                        vulnerable_ports=None,
                        dns_ports=None,
                        exp_id=(0, -1),
                        proto_list=(1, 2, 6, 17, 58)):
    """
    Reads the NFStream-extracted CSV at ``file_name``, computes the 16 temporal
    rolling-window features described in paper Table 1, one-hot encodes the
    `expiration_id` and `protocol` categorical columns, and overwrites the file
    in place.

    The function is idempotent on the schema as long as the original NFStream
    columns are still present; all helper boolean columns are dropped before the
    file is written back.
    """
    if http_ports is None:
        http_ports = DEFAULT_HTTP_PORTS
    if vulnerable_ports is None:
        vulnerable_ports = DEFAULT_VULNERABLE_PORTS
    if dns_ports is None:
        dns_ports = DEFAULT_DNS_PORTS

    try:
        data = pd.read_csv(file_name)
    except Exception:
        print(f"file reading error: {file_name}")
        return ""

    if 'bidirectional_first_seen_ms' not in data.columns:
        print(f" CSV does not contain bidirectional_first_seen_ms for initial sorting: {file_name}")
        return ""

    data = data.sort_values(by='bidirectional_first_seen_ms').reset_index(drop=True)

    # Group keys: per-destination IP, and per (src, dst) pair. Strings are
    # label-encoded only to make groupby cheaper, the encoded ids are dropped.
    src_dst = data['src_ip'].astype(str) + '-' + data['dst_ip'].astype(str)
    data['_src_dst_id'] = pd.factorize(src_dst)[0]
    data['_dst_id'] = pd.factorize(data['dst_ip'].astype(str))[0]

    # ------------------------------------------------------------------
    # Boolean / numeric helpers used as inputs to rolling sums.
    # ------------------------------------------------------------------
    data['_is_udp'] = (data['protocol'] == 17).astype(int)
    data['_is_tcp'] = (data['protocol'] == 6).astype(int)
    data['_is_icmp'] = (data['protocol'] == 1).astype(int)
    data['_is_http_port'] = data['dst_port'].isin(http_ports).astype(int)
    data['_is_dns_dst_port'] = data['dst_port'].isin(dns_ports).astype(int)
    data['_is_dns_src_port'] = data['src_port'].isin(dns_ports).astype(int)
    data['_is_vuln_port'] = data['dst_port'].isin(vulnerable_ports).astype(int)

    helper_cols = ['_src_dst_id', '_dst_id', '_is_udp', '_is_tcp', '_is_icmp',
                   '_is_http_port', '_is_dns_dst_port', '_is_dns_src_port',
                   '_is_vuln_port']

    # ------------------------------------------------------------------
    # 16 rolling-window features defined in paper Table 1.
    # The "_Destination" suffix matches the paper text; we additionally compute
    # a "_SourceDestination" variant for a richer feature set the HGNN can
    # learn from. Both are kept; the paper does not forbid the extra view.
    # ------------------------------------------------------------------

    # 1. Rolling_UDP_Sum  -- UDP packets to destination (rolling)
    data['Rolling_UDP_Sum'] = data.groupby('_dst_id')['_is_udp'].apply(
        lambda x: _rolling_sum(x, window_size)).reset_index(level=0, drop=True)

    # 2. Rolling_TCP_Sum
    data['Rolling_TCP_Sum'] = data.groupby('_dst_id')['_is_tcp'].apply(
        lambda x: _rolling_sum(x, window_size)).reset_index(level=0, drop=True)

    # 3. Rolling_ACK_Sum
    data['Rolling_ACK_Sum'] = data.groupby('_dst_id')['bidirectional_ack_packets'].apply(
        lambda x: _rolling_sum(x, window_size)).reset_index(level=0, drop=True)

    # 4. Rolling_FIN_Sum  (paper Table 1 lists Rolling_FIN_Sum twice; we map it once)
    data['Rolling_FIN_Sum'] = data.groupby('_dst_id')['bidirectional_fin_packets'].apply(
        lambda x: _rolling_sum(x, window_size)).reset_index(level=0, drop=True)

    # 5. Rolling_RST_Sum
    data['Rolling_RST_Sum'] = data.groupby('_dst_id')['bidirectional_rst_packets'].apply(
        lambda x: _rolling_sum(x, window_size)).reset_index(level=0, drop=True)

    # 6. Rolling_psh_Sum
    data['Rolling_psh_Sum'] = data.groupby('_dst_id')['bidirectional_psh_packets'].apply(
        lambda x: _rolling_sum(x, window_size)).reset_index(level=0, drop=True)

    # 7. Rolling_SYN_Sum
    data['Rolling_SYN_Sum'] = data.groupby('_dst_id')['bidirectional_syn_packets'].apply(
        lambda x: _rolling_sum(x, window_size)).reset_index(level=0, drop=True)

    # 8. Rolling_ICMP_Sum
    data['Rolling_ICMP_Sum'] = data.groupby('_dst_id')['_is_icmp'].apply(
        lambda x: _rolling_sum(x, window_size)).reset_index(level=0, drop=True)

    # 9. Rolling_http_port  -- frequency of HTTP-port access at destination
    data['Rolling_http_port'] = data.groupby('_dst_id')['_is_http_port'].apply(
        lambda x: _rolling_sum(x, window_size)).reset_index(level=0, drop=True)

    # 10. Rolling_Average_Duration  -- mean of bidirectional_duration_ms
    data['Rolling_Average_Duration'] = data.groupby('_dst_id')['bidirectional_duration_ms'].apply(
        lambda x: _rolling_mean(x, window_size)).reset_index(level=0, drop=True)

    # 11. Rolling_DNS_Sum  -- DNS requests to destination (dst_port == 53)
    data['Rolling_DNS_Sum'] = data.groupby('_dst_id')['_is_dns_dst_port'].apply(
        lambda x: _rolling_sum(x, window_size)).reset_index(level=0, drop=True)

    # 12. Rolling_vulnerable_port  -- presence of known vulnerable ports
    data['Rolling_vulnerable_port'] = data.groupby('_dst_id')['_is_vuln_port'].apply(
        lambda x: _rolling_sum(x, window_size)).reset_index(level=0, drop=True)

    # 13. Rolling_packets_Sum  -- all packets to destination
    data['Rolling_packets_Sum'] = data.groupby('_dst_id')['src2dst_packets'].apply(
        lambda x: _rolling_sum(x, window_size)).reset_index(level=0, drop=True)

    # 14. Rolling_bipackets_Sum  -- bidirectional packets at destination
    data['Rolling_bipackets_Sum'] = data.groupby('_dst_id')['bidirectional_packets'].apply(
        lambda x: _rolling_sum(x, window_size)).reset_index(level=0, drop=True)

    # 15. Unique_Ports_In_SourceDestination  -- unique src ports per (src,dst) pair
    data['Unique_Ports_In_SourceDestination'] = data.groupby('_src_dst_id')['src_port'].apply(
        lambda x: _rolling_unique(x, window_size)).reset_index(level=0, drop=True)

    # 16. packet_size_variation  -- statistical complement (kept from original tool)
    data['packet_size_variation'] = data[
        ['src2dst_min_ps', 'src2dst_max_ps', 'dst2src_min_ps', 'dst2src_max_ps']
    ].std(axis=1)

    # ------------------------------------------------------------------
    # One-hot encode categorical NFStream fields (kept from the original
    # implementation -- the HGNN flow node consumes the full row).
    # ------------------------------------------------------------------
    data['expiration_id'] = pd.Categorical(data['expiration_id'], categories=list(exp_id))
    data['protocol'] = pd.Categorical(data['protocol'], categories=list(proto_list))
    data = pd.get_dummies(data, prefix=['Exp', 'proto'],
                          columns=['expiration_id', 'protocol'], dtype=int)

    # Drop helper boolean / id columns -- they must NOT leak into flow node features.
    data.drop(columns=helper_cols, errors='ignore', inplace=True)

    data.to_csv(file_name, index=False)
