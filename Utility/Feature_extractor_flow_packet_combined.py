"""Flow and Feature Generator (XG-NID paper Sec. 3.1.1): PCAP -> flow CSV with
per-packet lists, built on NFStream.

Defaults are the paper's: at most 20 packets per flow (the flow is force-expired
on the 20th packet, so a longer conversation becomes several flows), idle
timeout 120 s, no L7 dissection (n_dissections=0 -> 77 NFStream columns + 14
``udps.*`` lists).  ``--n-dissections N`` adds NFStream's 9 L7 columns
(application_name, requested_server_name/SNI, JA3 fingerprints, user_agent, ...);
they are kept in the intermediate CSVs and dropped before the flow-node tensor
unless the pipeline is run with --keep-l7.

Usage (unchanged positional form, used by the notebooks and run_preprocessing.py):

    python Utility/Feature_extractor_flow_packet_combined.py <pcap> <dest_dir> [--n-dissections 0]
"""
import argparse
import os

import nfstream
from nfstream import NFStreamer, NFPlugin
import pandas as pd


class My_Custom(NFPlugin):

    def on_init(self, packet, flow):
        if self.limit == 1:
            flow.expiration_id = -1

        flow.udps.payload_data = []  # payload of each packet (hex)
        if packet.payload_size > 0:
            flow.udps.payload_data.append(packet.ip_packet[-packet.payload_size:].hex())
        else:
            flow.udps.payload_data.append(str('00'))

        flow.udps.delta_time = []  # time since the previous packet of the flow (ms)
        flow.udps.delta_time.append(packet.delta_time)

        flow.udps.packet_direction = []
        flow.udps.packet_direction.append(packet.direction)

        flow.udps.ip_size = []
        flow.udps.ip_size.append(packet.ip_size)

        flow.udps.transport_size = []
        flow.udps.transport_size.append(packet.transport_size)

        flow.udps.payload_size = []
        flow.udps.payload_size.append(packet.payload_size)

        # TCP flags of each packet
        flow.udps.syn = [packet.syn]
        flow.udps.cwr = [packet.cwr]
        flow.udps.ece = [packet.ece]
        flow.udps.urg = [packet.urg]
        flow.udps.ack = [packet.ack]
        flow.udps.psh = [packet.psh]
        flow.udps.rst = [packet.rst]
        flow.udps.fin = [packet.fin]

    def on_update(self, packet, flow):
        if packet.payload_size > 0:
            flow.udps.payload_data.append(packet.ip_packet[-packet.payload_size:].hex())
        else:
            flow.udps.payload_data.append(str('00'))

        flow.udps.delta_time.append(packet.delta_time)
        flow.udps.packet_direction.append(packet.direction)
        flow.udps.ip_size.append(packet.ip_size)
        flow.udps.transport_size.append(packet.transport_size)
        flow.udps.payload_size.append(packet.payload_size)

        flow.udps.syn.append(packet.syn)
        flow.udps.cwr.append(packet.cwr)
        flow.udps.ece.append(packet.ece)
        flow.udps.urg.append(packet.urg)
        flow.udps.ack.append(packet.ack)
        flow.udps.psh.append(packet.psh)
        flow.udps.rst.append(packet.rst)
        flow.udps.fin.append(packet.fin)

        if self.limit == flow.bidirectional_packets:
            flow.expiration_id = -1  # -1 forces expiration: next packet starts a new flow


def extract_pcap(pcap_path, dest_dir, n_dissections=0, limit=20, idle_timeout=120,
                 active_timeout=1800, bpf=None, out_name=None):
    """Run NFStream on one pcap and write ``<dest_dir>/<out_name>.csv``.

    Returns the CSV path.  ``out_name`` defaults to the pcap basename without
    its extension (the notebooks rename pcaps to ``<Class>-<Short>_<n>`` first).
    """
    os.makedirs(dest_dir, exist_ok=True)
    name = out_name or os.path.basename(pcap_path).split('.')[0]
    out_csv = os.path.join(dest_dir, name + '.csv')
    streamer = NFStreamer(source=pcap_path, accounting_mode=1,
                          idle_timeout=idle_timeout, active_timeout=active_timeout,
                          statistical_analysis=True, n_dissections=n_dissections,
                          bpf_filter=bpf or None, udps=My_Custom(limit=limit))
    print("*** Done Reading ***")
    streamer.to_csv(path=out_csv, columns_to_anonymize=[], flows_per_file=0)
    return out_csv


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process pcap files.')
    parser.add_argument('pcap_files', help='pcap file to process.')
    parser.add_argument('Destination_path',
                        help='directory where the extracted flow+packet CSV is stored')
    parser.add_argument('--n-dissections', type=int, default=0,
                        help='NFStream L7 dissection depth (0 = off, paper default)')
    parser.add_argument('--limit', type=int, default=20, help='packets per flow (paper: 20)')
    parser.add_argument('--idle-timeout', type=int, default=120, help='seconds (paper: 120)')
    parser.add_argument('--active-timeout', type=int, default=1800, help='seconds (NFStream default)')
    parser.add_argument('--bpf', default=None, help='optional BPF filter')
    parser.add_argument('--out-name', default=None, help='CSV stem (default: pcap basename)')
    args = parser.parse_args()
    extract_pcap(args.pcap_files, args.Destination_path, n_dissections=args.n_dissections,
                 limit=args.limit, idle_timeout=args.idle_timeout,
                 active_timeout=args.active_timeout, bpf=args.bpf, out_name=args.out_name)
