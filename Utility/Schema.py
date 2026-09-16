"""Single source of truth for the CIC-IoT2023 / XG-NID column schema.

Everything that must stay byte-compatible with the authors' Google-Drive CSVs
(``df_class_8_train.csv`` / ``df_class_8_test.csv``, 97 columns) and with the
checkpoints trained on them (82 flow-node features) is defined here, once.

Layout of the authors' class-8 CSV (verified on the Drive files):

    NFSTREAM_NUMERIC_46 + UDPS_14 + ['packet_size_variation'] + ROLLING_28 + ONEHOT_7 + ['Label']

and the flow node consumed by NIDSDataset is everything except UDPS_14 and Label:

    FLOW_FEATURE_NAMES_82 = NFSTREAM_NUMERIC_46 + ['packet_size_variation'] + ROLLING_28 + ONEHOT_7

The paper's "76 flow-level features" are the raw NFStream output minus ``id``
(77 - 1); the "14 packet-level features" are the 14 ``udps.*`` lists.  After the
notebook drops 29 identifier / redundant columns and adds 7 one-hots + 29
explainable features the flow node has 82 dimensions.
"""

# ---------------------------------------------------------------------------
# NFStream columns (statistical_analysis=True, n_dissections=0) that survive the
# notebook's 29-column drop, in NFStream's own order.
# ---------------------------------------------------------------------------
NFSTREAM_NUMERIC_46 = [
    "src2dst_duration_ms", "src2dst_packets", "src2dst_bytes",
    "dst2src_duration_ms", "dst2src_packets", "dst2src_bytes",
    "bidirectional_min_ps", "bidirectional_mean_ps",
    "bidirectional_stddev_ps", "bidirectional_max_ps",
    "src2dst_min_ps", "src2dst_mean_ps", "src2dst_stddev_ps", "src2dst_max_ps",
    "dst2src_min_ps", "dst2src_mean_ps", "dst2src_stddev_ps", "dst2src_max_ps",
    "bidirectional_min_piat_ms", "bidirectional_mean_piat_ms",
    "bidirectional_stddev_piat_ms", "bidirectional_max_piat_ms",
    "src2dst_min_piat_ms", "src2dst_mean_piat_ms",
    "src2dst_stddev_piat_ms", "src2dst_max_piat_ms",
    "dst2src_min_piat_ms", "dst2src_mean_piat_ms",
    "dst2src_stddev_piat_ms", "dst2src_max_piat_ms",
    "src2dst_syn_packets", "src2dst_cwr_packets", "src2dst_ece_packets",
    "src2dst_urg_packets", "src2dst_ack_packets", "src2dst_psh_packets",
    "src2dst_rst_packets", "src2dst_fin_packets",
    "dst2src_syn_packets", "dst2src_cwr_packets", "dst2src_ece_packets",
    "dst2src_urg_packets", "dst2src_ack_packets", "dst2src_psh_packets",
    "dst2src_rst_packets", "dst2src_fin_packets",
]

# The 14 packet-level lists produced by My_Custom (Feature_extractor_*.py).
UDPS_14 = [
    "udps.payload_data", "udps.delta_time", "udps.packet_direction",
    "udps.ip_size", "udps.transport_size", "udps.payload_size",
    "udps.syn", "udps.cwr", "udps.ece", "udps.urg",
    "udps.ack", "udps.psh", "udps.rst", "udps.fin",
]

# The authors' 28 rolling-window columns, in the order Additional_Features.py
# (upstream 551d1f1) assigns them.  Two groupings: *_SourceDestination (per
# src_ip-dst_ip pair) and *_Destination (per dst_ip).
ROLLING_28 = [
    "Rolling_UDP_Requests_SourceDestination", "Rolling_UDP_Requests_Destination",
    "Rolling_TCP_Requests_SourceDestination", "Rolling_TCP_Requests_Destination",
    "Rolling_ACK_Packets_SourceDestination", "Rolling_ACK_Packets_Destination",
    "Rolling_FIN_Packets_SourceDestination", "Rolling_FIN_Packets_Destination",
    "Rolling_rst_Packets_SourceDestination", "Rolling_rst_Packets_Destination",
    "Rolling_psh_Packets_SourceDestination", "Rolling_psh_Packets_Destination",
    "Rolling_SYN_Packets_SourceDestination", "Rolling_SYN_Packets_Destination",
    "Unique_Ports_In_SourceDestinationIP",
    "Rolling_ICMP_Requests_SourceDestination", "Rolling_ICMP_Requests_Destination",
    "Rolling_http_port_SourceDestination", "Rolling_http_port_Destination",
    "Rolling_Duration_Destination", "Rolling_Duration_SourceDestination",
    "Rolling_DNS_request_SourceDestination", "Rolling_DNS_request_Destination",
    "Rolling_DNS_request_SourceDestination2", "Rolling_DNS_request_Destination2",
    "Rolling_vulnerable_port",
    "Rolling_packets_destination", "Rolling_bipackets_destination",
]

ONEHOT_7 = ["Exp_0", "Exp_-1", "proto_1", "proto_2", "proto_6", "proto_17", "proto_58"]

FLOW_FEATURE_NAMES_82 = NFSTREAM_NUMERIC_46 + ["packet_size_variation"] + ROLLING_28 + ONEHOT_7
CLASS8_HEADER_97 = (NFSTREAM_NUMERIC_46 + UDPS_14 + ["packet_size_variation"]
                    + ROLLING_28 + ONEHOT_7 + ["Label"])
assert len(FLOW_FEATURE_NAMES_82) == 82
assert len(CLASS8_HEADER_97) == 97

# Paper Table 1 names <- the authors' destination-grouped columns (15 distinct
# names; the paper lists Rolling_FIN_Sum twice).  Used by schema='table1'.
TABLE1_MAP = {
    "Rolling_UDP_Requests_Destination": "Rolling_UDP_Sum",
    "Rolling_TCP_Requests_Destination": "Rolling_TCP_Sum",
    "Rolling_ACK_Packets_Destination": "Rolling_ACK_Sum",
    "Rolling_FIN_Packets_Destination": "Rolling_FIN_Sum",
    "Rolling_rst_Packets_Destination": "Rolling_RST_Sum",
    "Rolling_psh_Packets_Destination": "Rolling_psh_Sum",
    "Rolling_SYN_Packets_Destination": "Rolling_SYN_Sum",
    "Rolling_ICMP_Requests_Destination": "Rolling_ICMP_Sum",
    "Rolling_http_port_Destination": "Rolling_http_port",
    "Rolling_Duration_Destination": "Rolling_Average_Duration",
    "Rolling_DNS_request_Destination": "Rolling_DNS_Sum",
    "Rolling_vulnerable_port": "Rolling_vulnerable_port",
    "Rolling_packets_destination": "Rolling_packets_Sum",
    "Rolling_bipackets_destination": "Rolling_bipackets_Sum",
    "Unique_Ports_In_SourceDestinationIP": "Unique_Ports_In_SourceDestination",
}
TABLE1_16 = list(TABLE1_MAP.values()) + ["packet_size_variation"]

# ---------------------------------------------------------------------------
# Columns dropped before the flow node (Data_preprocessing notebook cells 12/14,
# verbatim): identifiers, absolute timestamps, and bidirectional_* totals that
# duplicate src2dst_* + dst2src_*.
# ---------------------------------------------------------------------------
IDENTIFIER_DROP_29 = [
    "src_ip", "src_port", "dst_ip", "dst_port", "ip_version",
    "bidirectional_bytes", "bidirectional_first_seen_ms", "bidirectional_last_seen_ms",
    "bidirectional_duration_ms", "bidirectional_packets",
    "src2dst_first_seen_ms", "src2dst_last_seen_ms",
    "dst2src_first_seen_ms", "dst2src_last_seen_ms",
    "id", "src_mac", "src_oui", "dst_mac", "dst_oui", "vlan_id", "tunnel_id",
    "bidirectional_syn_packets", "bidirectional_cwr_packets", "bidirectional_ece_packets",
    "bidirectional_urg_packets", "bidirectional_ack_packets", "bidirectional_psh_packets",
    "bidirectional_rst_packets", "bidirectional_fin_packets",
]
assert len(IDENTIFIER_DROP_29) == 29

# NFStream L7 columns that appear when n_dissections > 0 (nfstream 6.5.3 NFlow
# slots).  They are strings, so they must never reach the flow-node tensor.
L7_COLUMNS_9 = [
    "application_name", "application_category_name", "application_is_guessed",
    "application_confidence", "requested_server_name", "client_fingerprint",
    "server_fingerprint", "user_agent", "content_type",
]
# Present only with splt_analysis > 0 (never used by GNN4ID); dropped defensively.
SPLT_COLUMNS_3 = ["splt_direction", "splt_ps", "splt_piat_ms"]

# Helper columns the pipeline adds and always removes before writing class-8 CSVs.
HELPER_COLUMNS = ("__subtype", "__src_file")

TIME_COL = "bidirectional_first_seen_ms"

# ---------------------------------------------------------------------------
# Port lists used by the explainable feature extractor (authors' defaults).
# ---------------------------------------------------------------------------
DEFAULT_HTTP_PORTS = (443, 8080, 80)
DEFAULT_DNS_PORTS = (53,)
DEFAULT_VULNERABLE_PORTS = (20, 21, 22, 23, 25, 53, 80, 110, 143, 443, 445, 3389, 8080)

# ---------------------------------------------------------------------------
# CIC-IoT2023 metadata
# ---------------------------------------------------------------------------
# Attacker MAC addresses, paper Table 3 (lower-case, as NFStream writes them).
CIC_IOT2023_ATTACKER_MACS = frozenset({
    "dc:a6:32:dc:27:d5", "e4:5f:01:55:90:c4", "dc:a6:32:c9:e4:ab",
    "ac:17:02:05:34:27", "dc:a6:32:c9:e5:a4", "dc:a6:32:c9:e4:d5",
    "dc:a6:32:c9:e5:ef", "dc:a6:32:c9:e4:90", "b0:09:da:3e:82:6c",
})

# Class codes as used everywhere in the code base (the paper spells DoS/DDoS).
DEFAULT_LABELS = {"Benign": 0, "WebBased": 1, "Spoofing": 2, "Recon": 3,
                  "Mirai": 4, "Dos": 5, "DDos": 6, "BruteForce": 7}
LABEL_ALIASES = {"DoS": "Dos", "DDoS": "DDos", "Webbased": "WebBased",
                 "Bruteforce": "BruteForce", "BruteForce": "BruteForce"}

# CIC-IoT2023 PCAP folder / file stems (case exactly as shipped) -> (class code,
# short sub-attack name used by the authors' name_mapping).  Benign pcaps are
# ``BenignTraffic*.pcap`` inside the ``Benign_Final`` folder; both spellings map
# to Benign.  Resolution is case-insensitive longest-prefix (see
# Functions.resolve_class_subtype), so the authors' misspelt canonical names
# ('DDos-SlowLoris', 'Webbased-BrwserHijack') resolve as well.
CIC_SUBTYPES = {
    "Benign_Final": ("Benign", "Benign"),
    "BenignTraffic": ("Benign", "Benign"),
    "Benign": ("Benign", "Benign"),
    # WebBased (6)
    "Backdoor_Malware": ("WebBased", "BckdoorMalware"),
    "BrowserHijacking": ("WebBased", "BrwserHijack"),
    "CommandInjection": ("WebBased", "CmmdInject"),
    "SqlInjection": ("WebBased", "SqlInject"),
    "Uploading_Attack": ("WebBased", "UploadAttack"),
    "XSS": ("WebBased", "XSS"),
    # Spoofing (2)
    "DNS_Spoofing": ("Spoofing", "DNS"),
    "MITM-ArpSpoofing": ("Spoofing", "ARP"),
    # Recon (5)
    "Recon-HostDiscovery": ("Recon", "HostDisc"),
    "Recon-OSScan": ("Recon", "OSScan"),
    "Recon-PingSweep": ("Recon", "PingSweep"),
    "Recon-PortScan": ("Recon", "PortScan"),
    "VulnerabilityScan": ("Recon", "VulScan"),
    # Mirai (3)
    "Mirai-greeth_flood": ("Mirai", "Greeth"),
    "Mirai-greip_flood": ("Mirai", "GREIP"),
    "Mirai-udpplain": ("Mirai", "UDPPlain"),
    # DoS (4)
    "DoS-HTTP_Flood": ("Dos", "HTTPFlood"),
    "DoS-SYN_Flood": ("Dos", "SYNFlood"),
    "DoS-TCP_Flood": ("Dos", "TCPFlood"),
    "DoS-UDP_Flood": ("Dos", "UDPFlood"),
    # DDoS (12)
    "DDoS-ACK_Fragmentation": ("DDos", "AckFrg"),
    "DDoS-HTTP_Flood": ("DDos", "HTTPFlood"),
    "DDoS-ICMP_Flood": ("DDos", "ICMPFlood"),
    "DDoS-ICMP_Fragmentation": ("DDos", "ICMPFrg"),
    "DDoS-PSHACK_Flood": ("DDos", "PSHACK"),
    "DDoS-RSTFINFlood": ("DDos", "RSTFIN"),
    "DDoS-SlowLoris": ("DDos", "SlowLoris"),
    "DDoS-SYN_Flood": ("DDos", "SYNFlood"),
    "DDoS-SynonymousIP_Flood": ("DDos", "SynonymousIPFlood"),
    "DDoS-TCP_Flood": ("DDos", "TCPFlood"),
    "DDoS-UDP_Flood": ("DDos", "UDPFlood"),
    "DDoS-UDP_Fragmentation": ("DDos", "UDPFrg"),
    # BruteForce (1)
    "DictionaryBruteForce": ("BruteForce", "Dictionary"),
}
# 33 attacks + benign (3 spellings) = 36 keys
assert len(CIC_SUBTYPES) == 36
