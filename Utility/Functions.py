import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
import os
import re
import glob
from torch_geometric.data import HeteroData
from torch_geometric.data import Dataset
import torch_geometric.transforms as T
import sys
import random
import json
import warnings
from collections import namedtuple

from Utility.Schema import (
    CIC_SUBTYPES, CIC_IOT2023_ATTACKER_MACS, DEFAULT_LABELS, LABEL_ALIASES,
    IDENTIFIER_DROP_29, L7_COLUMNS_9, SPLT_COLUMNS_3, HELPER_COLUMNS, TIME_COL,
    CLASS8_HEADER_97,
)


class NIDSDataset(Dataset):

    """A dataset classfor  generates graph data objects from a provided
    CSV File/Files. Build uopn torch_geometric.data.Dataset base class for 
    creating graph datasets.

    Args:
        root (str): Root directory where the dataset should be saved.
        
        label_dict (Dict): Dictionary for assigning labels to each attack class.

        filename(List[str]): List of CSV file paths to be used for the development of 
            graph objects.

        skip_processing (bool): If set to `True`, skips the generation of graph 
            objects and utilizes the ones present in the root directory. 
            (default: `False`)

        test (bool): If set to `True`, generates data objects for testing by 
            creating data objects with a test suffix. 
            (default: `False`)

        index_num (int): Index to start with saving the grapgh object. Useful 
            when creating continuing grapgh object and you have to run the same
            class twice.
            (default: 0)

        include_packetflag (bool): If set to True, will also include Flags status
            as packet node features.
            (default: False)

        include_packetpayload (bool): If set to False, will not include payload 
            information as a packet node features.
            (default: True) 

        single_file (bool): If set to True, the provided CSV files is a single 
            file with Label column within CSV.
            (default: False)      

        transform (callable, optional): A function/transform that takes in a
            :class:`~torch_geometric.data.Data` or
            :class:`~torch_geometric.data.HeteroData` object and returns a
            transformed version.
            The data object will be transformed before every access.
            (default: :obj:`None`)

        pre_transform (callable, optional): A function/transform that takes in
            a :class:`~torch_geometric.data.Data` or
            :class:`~torch_geometric.data.HeteroData` object and returns a
            transformed version.
            The data object will be transformed before being saved to disk.
            (default: :obj:`None`)
    """
    
    def __init__(self, root, label_dict, filename, index_num = 0, skip_processing=False, include_packetflag=False, include_packetpayload=True, test=False, single_file=False, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        self.test = test
        self.filename = filename
        self.include_packetflag = include_packetflag
        self.include_packetpayload = include_packetpayload
        self.index = index_num
        self.label_dict = label_dict
        self.skip_processing = skip_processing
        self.length = 0
        self.single_file = single_file
        super(NIDSDataset, self).__init__(root, transform, pre_transform)
        
    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        return self.filename

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        if self.skip_processing:
            if self.test:
                list_process = glob.glob(os.path.join(self.root, 'processed/data_test*'))
                self.length = len(list_process)
                return list_process[0]
            else:
                list_process = glob.glob(os.path.join(self.root, 'processed/data*'))
                self.length = len(list_process) - len(glob.glob(os.path.join(self.root, 'processed/data_test*')))
                return list_process[0]
        else:
            return []

    def download(self):
        pass

    def process(self):

        for files in self.raw_paths:
            
            self.data = pd.read_csv(files)
            print('Reading File ---> '+os.path.basename(files), file=sys.stderr)

            if not self.single_file:
                # Removing Some features that might seems unneccesssary. Features can be added to the node features by removing from the drop list.
                self.data.drop(['src_ip','src_port','dst_ip','dst_port','ip_version'], axis=1, inplace=True)
                self.data.drop(['bidirectional_bytes','bidirectional_first_seen_ms','bidirectional_last_seen_ms','bidirectional_duration_ms',
                         'bidirectional_packets','src2dst_first_seen_ms','src2dst_last_seen_ms','dst2src_first_seen_ms','dst2src_last_seen_ms',
                         'id','src_mac','src_oui','dst_mac','dst_oui','vlan_id','tunnel_id','bidirectional_syn_packets','bidirectional_cwr_packets',
                         'bidirectional_ece_packets','bidirectional_urg_packets','bidirectional_ack_packets','bidirectional_psh_packets',
                         'bidirectional_rst_packets','bidirectional_fin_packets'], axis=1, inplace=True)

                # Creating Dummy variables for Expiration_ID and protocol
                self.data['expiration_id']=pd.Categorical(self.data['expiration_id'], categories=[0,-1])
                # Creating Dummy varaible for protocol, make sure to incorporate all the protocols. There are only 5 protocols in the CIC-IoT2023 dataset. Add protocol number if utilizing other dataset.
                self.data['protocol']=pd.Categorical(self.data['protocol'], categories=[1,2,6,17,58])
                self.data=pd.get_dummies(self.data, prefix=['Exp','proto'], columns=['expiration_id', 'protocol'],dtype=int)
                # Getting the Label from the file name and provided dictionary
                label = self._get_labels(files)

            # Drop the temporary `is_vulnerable_port` helper column produced by
            # the explainable feature extractor; it was used only as a rolling
            # input and must not leak into flow node features (paper Sec 3.1.1
            # specifies 76 flow features, no raw is_vulnerable_port boolean).
            # Also drop NFStream L7 string columns (present when the extractor ran
            # with --n-dissections > 0) and pipeline helper columns: the flow node
            # is np.asarray(dtype=float), so no string may survive here.
            for tmp_col in (('is_vulnerable_port', 'is_http_port', 'is_dns_dst_port',
                             'is_dns_src_port', 'is_vuln_port', 'is_udp_request',
                             'is_tcp_request', 'is_icmp_request')
                            + tuple(L7_COLUMNS_9) + tuple(SPLT_COLUMNS_3) + tuple(HELPER_COLUMNS)):
                if tmp_col in self.data.columns:
                    self.data.drop(tmp_col, axis=1, inplace=True)

    
            ## Converting String into iterable list; needed for extracting individual packet features
            self.data['udps.payload_data'] = self.data['udps.payload_data'].map(lambda x: x.strip('][').replace("'","").split(', '))
            self.data['udps.packet_direction'] = self.data['udps.packet_direction'].map(lambda x: x.strip('][').replace("'","").split(', '))
            self.data['udps.ip_size'] = self.data['udps.ip_size'].map(lambda x: x.strip('][').replace("'","").split(', '))
            self.data['udps.transport_size'] = self.data['udps.transport_size'].map(lambda x: x.strip('][').replace("'","").split(', '))
            self.data['udps.payload_size'] = self.data['udps.payload_size'].map(lambda x: x.strip('][').replace("'","").split(', '))
            self.data['udps.delta_time'] = self.data['udps.delta_time'].map(lambda x: x.strip('][').replace("'","").split(', '))
    
            ## If include_packetflag set True, then 
            if self.include_packetflag==True:
                self.data['udps.syn'] = self.data['udps.syn'].map(lambda x: x.strip('][').replace("'","").split(', '))
                self.data['udps.cwr'] = self.data['udps.cwr'].map(lambda x: x.strip('][').replace("'","").split(', '))
                self.data['udps.ece'] = self.data['udps.ece'].map(lambda x: x.strip('][').replace("'","").split(', '))
                self.data['udps.urg'] = self.data['udps.urg'].map(lambda x: x.strip('][').replace("'","").split(', '))
                self.data['udps.ack'] = self.data['udps.ack'].map(lambda x: x.strip('][').replace("'","").split(', '))
                self.data['udps.psh'] = self.data['udps.psh'].map(lambda x: x.strip('][').replace("'","").split(', '))
                self.data['udps.rst'] = self.data['udps.rst'].map(lambda x: x.strip('][').replace("'","").split(', '))
                self.data['udps.fin'] = self.data['udps.fin'].map(lambda x: x.strip('][').replace("'","").split(', '))
            
            
            
            for index, flow in tqdm(self.data.iterrows(), total=self.data.shape[0]):
                # Get Node Feaetures
                flow_node_feats = self._get_flow_node_features(flow) # Get Flow_node features
                packet_node_feats = self._get_packet_node_features(flow) # Get Packet_node features
                
                # Get Edge Index/Adjacency Matrix
                contain_edge_index = self._get_contain_edge_index(len(flow['udps.payload_data'])) # Get Contain_edge Index
                link_edge_index = self._get_link_edge_index(len(flow['udps.payload_data'])) # Get Link_edge Index
                
                # Get Edge Features/Attributes
                contain_edge_feats = self._get_contain_edge_features(flow) # Get contain_edge features
                link_edge_feats = self._get_link_edge_features(flow) # Get link_edge features
    
                ### Get labels info (If label is present in a dataset column. You may need to edit the _get_labels function as per the dataset columns.
                if self.single_file:
                    label = torch.tensor(np.asarray(flow["Label"]), dtype=torch.int64)
    
                # Create data object
                data = HeteroData()
                data['flow'].x = flow_node_feats
                data['packet'].x = packet_node_feats
                data['flow', 'contain', 'packet'].edge_index = contain_edge_index
                data['packet', 'link', 'packet'].edge_index = link_edge_index
                data['flow', 'contain', 'packet'].edge_attr = contain_edge_feats
                data['packet', 'link', 'packet'].edge_attr = link_edge_feats
                data.y = label

                data = T.ToUndirected()(data)
    
                if self.test:
                    torch.save(data, 
                        os.path.join(self.processed_dir, 
                                     f'data_test_{self.index}.pt'))
                else:
                    torch.save(data, 
                        os.path.join(self.processed_dir, 
                                     f'data_{self.index}.pt'))
                self.index+=1
            
            if not self.skip_processing:
                if self.test:
                    list_process = glob.glob(os.path.join(self.root, 'processed/data_test*'))
                    self.length = len(list_process)
                else:
                    list_process = glob.glob(os.path.join(self.root, 'processed/data*'))
                    self.length = len(list_process) - len(glob.glob(os.path.join(self.root, 'processed/data_test*')))
                

    def _get_flow_node_features(self, flow):
        """ Extract the features for Flow_node
        This will return a 2d array of the shape
        [1 , Node Feature size] """
        
        ## Removing the columns that are not part of flow_node features
        if self.single_file:
            ## If Using a Single CSV File that have Label Column in it
            flow_data=flow.drop(['Label','udps.payload_data','udps.delta_time', 'udps.packet_direction', 'udps.ip_size','udps.transport_size', 
                    'udps.payload_size', 'udps.syn', 'udps.cwr','udps.ece', 'udps.urg', 'udps.ack', 'udps.psh', 'udps.rst', 'udps.fin'])
        else:
            ## If Using multiple CSV files that do not have Label Column in them
            flow_data=flow.drop(['udps.payload_data','udps.delta_time', 'udps.packet_direction', 'udps.ip_size','udps.transport_size', 
                    'udps.payload_size', 'udps.syn', 'udps.cwr','udps.ece', 'udps.urg', 'udps.ack', 'udps.psh', 'udps.rst', 'udps.fin'])

    
        ## Transforming into tensors for pytorch
        all_flow_node_feats = np.asarray(flow_data, dtype=float)
        all_flow_node_feats = np.reshape(all_flow_node_feats, (-1, all_flow_node_feats.shape[0]))
        all_flow_node_feats=torch.tensor(all_flow_node_feats, dtype=torch.float32)
        
        return all_flow_node_feats

    def _get_packet_node_features(self, flow):
        """ Extract the features for packet_node
        Features for each packet is its payload data. We can also incorporate flags data for each packet. 
        This will return a 2d array of the shape
        [Num_packets_in_flow, Node Feature size] """
        
        dims = 1500 ## Length for payload Bytes to be incorporated/Number for columns for payload data
        all_packet_node_feats=[]

        for index_value in range(len(flow['udps.payload_data'])):
            packet_feats_combined = []
            
            ## Each Packet Flags as Packet Node Features
            if self.include_packetflag==True:
                packet_feats_combined.append(flow['udps.syn'][index_value])
                packet_feats_combined.append(flow['udps.cwr'][index_value])
                packet_feats_combined.append(flow['udps.ece'][index_value])
                packet_feats_combined.append(flow['udps.urg'][index_value])
                packet_feats_combined.append(flow['udps.ack'][index_value])
                packet_feats_combined.append(flow['udps.psh'][index_value])
                packet_feats_combined.append(flow['udps.rst'][index_value])
                packet_feats_combined.append(flow['udps.fin'][index_value])

            ## Packet Payload as Packet Node Features
            if self.include_packetpayload==True:
                byte_array = bytes.fromhex(flow['udps.payload_data'][index_value])
                byte_lst = list(byte_array)
                if (len(byte_lst) < dims):
                    packet_feat = np.pad(byte_lst, (0, dims-len(byte_lst)), 'constant')
                else:
                    packet_feat = np.array(byte_lst[0:dims].copy())
                packet_feat = np.abs(np.uint8(packet_feat))
                packet_feats_combined.extend(packet_feat.tolist())

            all_packet_node_feats.append(packet_feats_combined)

        all_packet_node_feats = np.asarray(all_packet_node_feats, dtype=int)
        all_packet_node_feats = torch.tensor(all_packet_node_feats, dtype=torch.float32)

        return all_packet_node_feats
            

    def _get_contain_edge_features(self, flow):
        """ Extract the features for contain_edge.
        Features for each contain_edge is packet_direction, ip_size, transport_size, payload_size of that particular packet.        
        This will return a 2d array of the shape
        [Number of contain_edges, contain_edge Feature size]
        """
        contain_edge_all_feats = []
        
        for index_value in range(len(flow['udps.packet_direction'])):
            contain_edge_feat = []
            contain_edge_feat.append(flow['udps.packet_direction'][index_value])
            contain_edge_feat.append(flow['udps.ip_size'][index_value])
            contain_edge_feat.append(flow['udps.transport_size'][index_value])
            contain_edge_feat.append(flow['udps.payload_size'][index_value])
            contain_edge_all_feats.append(contain_edge_feat)
    
        contain_edge_all_feats = np.asarray(contain_edge_all_feats, dtype=int)
        contain_edge_all_feats=torch.tensor(contain_edge_all_feats, dtype=torch.float32)

        return contain_edge_all_feats

    def _get_link_edge_features(self, flow):
        """ Extract the features for link_edge.
        Features for each link_edge is delta_time between the two packets.
        This will return a 2d array of the shape
        [Number of link_edges, link_edge Feature size]
        """
        link_edge_feats = np.asarray(flow['udps.delta_time'][1:], dtype=int)
        link_edge_feats=torch.tensor(link_edge_feats, dtype=torch.float32)
        return link_edge_feats

    def _get_contain_edge_index(self, length):
        """ Extract the adjacency matrix for the edges connection between flow_node and packet_nodes.
        Contain_Edge links the flow_node to each and every packet_node that it contains.
        For Example: If there are 20 packets in a particular flow then there will be 20 Contain_Edge.
                     As each packet is linked to the flow.
        This will return a matrix / 2d array of the shape
        [2, num_contain_edges]
        """
        Flow = np.zeros(length,dtype=int)
        Packet = np.arange(0, length)
        contain_edge = np.vstack ((Flow, Packet))
        contain_edge = torch.tensor(contain_edge, dtype=torch.int64)
        return contain_edge

    def _get_link_edge_index(self, length):
        """ Extract the adjacency matrix for the edges connection between packet_nodes.
        Link_Edge links the packet_node to other packet_node available.
        For Example: Each packet is linked with its following packet. If there are 20 packets then 
                     there will be 19 link_edges.
        This will return a matrix / 2d array of the shape
        [2, num_link_edges]
        """
        packet_ini = np.arange(0, length-1)
        packet_next = np.arange(1, length)
        link_edge =  np.vstack ((packet_ini, packet_next))
        link_edge = torch.tensor(link_edge, dtype=torch.int64)
        return link_edge

    def _get_labels(self, file_name):
        """Label from the file name: '<Class>-<Sub>_<n>.csv' -> label_dict[Class].
        Uses the CIC-IoT2023 resolver so raw CIC names (e.g. BenignTraffic3,
        DDoS-SlowLoris1) and the authors' canonical names both work."""
        try:
            class_code = resolve_class_subtype(file_name).class_code
        except KeyError:
            class_code = os.path.basename(file_name).split('-')[0]
        label_dict = _with_label_aliases(self.label_dict)
        label = label_dict[class_code]
        return torch.tensor(np.asarray([label]), dtype=torch.int64)

    def len(self):

        return self.length
        
        

    def get(self, idx):
        """ _ Equivalent to __getitem__ in pytorch"""
        if self.test:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_test_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_{idx}.pt'))   
        return data
        
        

# ===========================================================================
# CIC-IoT2023 file-name resolution
# ===========================================================================
ClassInfo = namedtuple("ClassInfo", "class_code subtype_key short canonical")

# lower-cased lookup: CIC stem and the authors' canonical '<Class>-<Short>' -> CIC key
_LOOKUP = {}
for _key, (_cls, _short) in CIC_SUBTYPES.items():
    _LOOKUP[_key.lower()] = _key
    _LOOKUP[("%s-%s" % (_cls, _short)).lower()] = _key
_EXTS = {'.pcap', '.pcapng', '.cap', '.gz', '.csv', '.parquet', '.pt'}


def _with_label_aliases(label_dict):
    """Accept paper spellings (DoS/DDoS) next to the code spellings (Dos/DDos)."""
    d = dict(DEFAULT_LABELS if label_dict is None else label_dict)
    for alias, code in LABEL_ALIASES.items():
        if code in d and alias not in d:
            d[alias] = d[code]
    return d


def _strip_stem(name):
    stem = os.path.basename(str(name))
    root, ext = os.path.splitext(stem)
    while ext.lower() in _EXTS:
        stem = root
        root, ext = os.path.splitext(stem)
    s = re.sub(r'(_train|_test)$', '', stem, flags=re.I)
    s = re.sub(r'[_\-\s]*\d+$', '', s)
    return stem, s.lower()


def resolve_class_subtype(name):
    """Map a pcap/csv name to ``ClassInfo(class_code, subtype_key, short, canonical)``.

    Case-insensitive longest-prefix match against the 34 CIC-IoT2023 folder
    stems (``DDoS-SlowLoris1.pcap``, ``BenignTraffic3.pcap``, ``XSS.pcap`` ...)
    and the authors' canonical names (``DDos-SlowLoris_0.csv``,
    ``Webbased-BrwserHijack_0_test.csv`` ...).  Raises ``KeyError`` when the
    name matches nothing, so unknown files are never silently dropped.
    """
    stem, s = _strip_stem(name)
    best = None
    for k in _LOOKUP:
        if s.startswith(k) and (best is None or len(k) > len(best)):
            best = k
    if best is None:
        raise KeyError("cannot resolve CIC-IoT2023 class/sub-attack for %r" % (name,))
    key = _LOOKUP[best]
    cls, short = CIC_SUBTYPES[key]
    return ClassInfo(cls, key, short, "%s-%s" % (cls, short))


def canonical_stem(name):
    """'DDoS-ICMP_Flood10.pcap' -> 'DDos-ICMPFlood_10'; 'BenignTraffic.pcap' -> 'Benign-Benign_0'."""
    info = resolve_class_subtype(name)
    stem, _ = _strip_stem(name)
    stem = re.sub(r'(_train|_test)$', '', stem, flags=re.I)
    m = re.search(r'(\d+)$', stem)
    return "%s_%s" % (info.canonical, m.group(1) if m else "0")


def assert_all_resolvable(paths):
    bad = []
    for p in paths:
        try:
            resolve_class_subtype(p)
        except KeyError:
            bad.append(os.path.basename(str(p)))
    if bad:
        raise ValueError("files that map to no CIC-IoT2023 class (fix the name or extend "
                         "Utility.Schema.CIC_SUBTYPES): %s" % bad)


def group_files_by_class(paths):
    """{class_code: [(path, short_subtype), ...]} in sorted path order."""
    out = {}
    for p in sorted(paths):
        info = resolve_class_subtype(p)
        out.setdefault(info.class_code, []).append((p, info.short))
    return out


def rename_files(directory, name_mapping=None):
    """Rename every pcap under ``directory`` (recursively) to the authors'
    canonical ``<Class>-<Short>_<n>.pcap`` form.

    ``name_mapping`` (the notebooks' dict) is accepted for backward
    compatibility but the CIC-IoT2023 resolver is authoritative: it is
    case-insensitive and knows that benign pcaps are ``BenignTraffic*``.
    """
    if not os.path.exists(directory):
        print(f"The directory '{directory}' does not exist.")
        return
    files = glob.glob(os.path.join(directory, "**", "*pcap*"), recursive=True)
    files = [f for f in files if os.path.isfile(f)]
    assert_all_resolvable(files)
    for filename in files:
        new_name = os.path.join(os.path.dirname(filename), canonical_stem(filename) + ".pcap")
        if os.path.abspath(new_name) == os.path.abspath(filename):
            continue
        if os.path.exists(new_name):
            raise FileExistsError("refusing to overwrite %s" % new_name)
        os.rename(filename, new_name)
        print(f"Renamed: {os.path.basename(filename)} -> {os.path.basename(new_name)}")


def extract_number(file_name: str) -> str:
    """First sequence of digits in a file name (legacy helper)."""
    return re.search(r'\d+', file_name).group(0)


# ===========================================================================
# Row-level helpers: MAC filter, feature view, dedup, proportional sampling
# ===========================================================================
def mac_filter(df, is_benign, attacker_macs=CIC_IOT2023_ATTACKER_MACS):
    """Paper Sec. 3.2.1: attack files keep only flows that touch an attacker
    MAC; the benign file drops every flow that touches one."""
    if 'src_mac' not in df.columns or 'dst_mac' not in df.columns:
        warnings.warn("mac_filter: no src_mac/dst_mac columns, nothing filtered")
        return df
    macs = {m.lower() for m in attacker_macs}
    src_a = df['src_mac'].astype(str).str.lower().isin(macs)
    dst_a = df['dst_mac'].astype(str).str.lower().isin(macs)
    return df[~src_a & ~dst_a] if is_benign else df[src_a | dst_a]


def feature_view_columns(df, drop_columns=IDENTIFIER_DROP_29, extra_drop=None):
    """Columns that end up in the graph (flow features + udps.* packet lists):
    everything except identifiers, L7 strings, helpers and Label."""
    skip = set(drop_columns) | set(L7_COLUMNS_9) | set(SPLT_COLUMNS_3) | set(HELPER_COLUMNS) | {'Label'}
    if extra_drop:
        skip |= set(extra_drop)
    return [c for c in df.columns if c not in skip]


def row_keys(df, feat_cols=None):
    """One uint64 key per row over ``feat_cols`` (numeric columns compared as
    float64, everything else as text) so that a test row byte-identical to a
    train row gets the same key whichever file it came from."""
    if feat_cols is None:
        feat_cols = feature_view_columns(df)
    if len(df) == 0:
        return np.zeros(0, dtype=np.uint64)
    view = pd.DataFrame(index=df.index)
    for c in feat_cols:
        col = df[c]
        if pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col):
            view[c] = col.astype('float64')
        else:
            view[c] = col.astype(str)
    return pd.util.hash_pandas_object(view, index=False).to_numpy()


def dedup_across_splits(train, test, feat_cols=None, dedup_test_within=False):
    """Drop duplicate train rows; drop test rows whose feature key exists in
    train (optionally also test-internal duplicates).  Returns
    ``(train_unique, test_clean, stats)``."""
    if feat_cols is None:
        feat_cols = [c for c in feature_view_columns(train) if c in test.columns]
    tk = row_keys(train, feat_cols)
    keep_tr = ~pd.Series(tk).duplicated().to_numpy()
    train_u = train.loc[keep_tr].reset_index(drop=True)
    train_set = set(tk[keep_tr].tolist())
    ek = row_keys(test, feat_cols)
    leaked = np.fromiter((k in train_set for k in ek.tolist()), dtype=bool, count=len(ek))
    keep_te = ~leaked
    if dedup_test_within:
        keep_te &= ~pd.Series(ek).duplicated().to_numpy()
    test_c = test.loc[keep_te].reset_index(drop=True)
    stats = {"train_before": int(len(train)), "train_unique": int(len(train_u)),
             "test_before": int(len(test)), "test_leaked_removed": int(leaked.sum()),
             "test_within_dups_removed": int((~keep_te).sum() - leaked.sum()),
             "test_after": int(len(test_c))}
    return train_u, test_c, stats


def _allocate(sizes, total):
    """Largest-remainder allocation of ``total`` over groups proportional to
    ``sizes`` (dict), never exceeding a group's own size, exact sum."""
    sizes = {k: int(v) for k, v in sizes.items() if v > 0}
    n = sum(sizes.values())
    if n == 0 or total <= 0:
        return {k: 0 for k in sizes}
    if total >= n:
        return dict(sizes)
    raw = {k: total * v / n for k, v in sizes.items()}
    alloc = {k: int(raw[k]) for k in sizes}
    rem = total - sum(alloc.values())
    for k in sorted(raw, key=lambda k: raw[k] - alloc[k], reverse=True):
        if rem <= 0:
            break
        if alloc[k] < sizes[k]:
            alloc[k] += 1
            rem -= 1
    # at least one row per non-empty group when the budget allows it
    for k in sizes:
        if alloc[k] == 0 and sizes[k] > 0 and total >= len(sizes):
            donor = max(alloc, key=alloc.get)
            if alloc[donor] > 1:
                alloc[donor] -= 1
                alloc[k] = 1
    return alloc


def proportional_cap(df, cap, subtype_col="__subtype", rng=None):
    """Undersample ``df`` to at most ``cap`` rows, proportionally per subtype
    (no replacement).  ``cap`` None/0 disables."""
    if not cap or len(df) <= cap:
        return df
    rng = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    if subtype_col not in df.columns:
        idx = rng.choice(len(df), size=cap, replace=False)
        return df.iloc[np.sort(idx)].reset_index(drop=True)
    sizes = df[subtype_col].value_counts().to_dict()
    alloc = _allocate(sizes, cap)
    parts = []
    for sub, grp in df.groupby(subtype_col, sort=False):
        k = alloc.get(sub, 0)
        if k <= 0:
            continue
        idx = rng.choice(len(grp), size=min(k, len(grp)), replace=False)
        parts.append(grp.iloc[np.sort(idx)])
    return pd.concat(parts, ignore_index=True)


def oversample_train(df, target, subtype_col="__subtype", rng=None):
    """Duplicate rows (with replacement, proportional per subtype) until ``df``
    has ``target`` rows.  TRAIN ONLY.  Rows of ``df`` are all kept."""
    n = len(df)
    if not target or n == 0 or n >= target:
        return df
    rng = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    deficit = target - n
    if subtype_col in df.columns:
        sizes = df[subtype_col].value_counts().to_dict()
        tot = sum(sizes.values())
        raw = {k: deficit * v / tot for k, v in sizes.items()}
        alloc = {k: int(v) for k, v in raw.items()}
        rem = deficit - sum(alloc.values())
        for k in sorted(raw, key=lambda k: raw[k] - alloc[k], reverse=True)[:rem]:
            alloc[k] += 1
        extra = []
        for sub, grp in df.groupby(subtype_col, sort=False):
            k = alloc.get(sub, 0)
            if k > 0:
                extra.append(grp.iloc[rng.choice(len(grp), size=k, replace=True)])
        out = pd.concat([df] + extra, ignore_index=True)
    else:
        out = pd.concat([df, df.iloc[rng.choice(n, size=deficit, replace=True)]], ignore_index=True)
    perm = rng.permutation(len(out))
    return out.iloc[perm].reset_index(drop=True)


def duplicate_rows(df, target_rows):
    """Superseded by :func:`oversample_train` (train-only, after dedup); kept
    so old notebooks keep importing.  Duplicates rows to reach ``target_rows``."""
    return oversample_train(df, target_rows, subtype_col="__no_such_column__", rng=42)


def random_pick_rows(df, original, over):
    """Legacy helper (kept for import compatibility)."""
    for i in range(over):
        row_to_duplicate = random.randint(0, original.shape[0]-1)
        df = pd.concat([df, original.iloc[row_to_duplicate:row_to_duplicate+1]], ignore_index=True)
    return df


# ===========================================================================
# split_csv: MAC filter + temporal 80/20 split + per-file caps  (NO rolling here)
# ===========================================================================
def split_csv(file_path, out_dir=None, test_frac=0.2, test_cap=4000, train_cap=20000,
              temporal=True, test_pick="random", dedup="feature", mac_filter_on=True,
              attacker_macs=CIC_IOT2023_ATTACKER_MACS, seed=42,
              test_sample=None, Number_in_individaul_class=None, **legacy):
    """Split one enriched per-pcap CSV into train / test parts.

    Order inside the function: attacker-MAC filter (paper Sec. 3.2.1) ->
    stable sort by ``bidirectional_first_seen_ms`` -> the latest ``test_frac``
    of the flows is the test pool -> ``test = min(test_cap, pool)`` rows, picked
    at random inside the pool (``test_pick='random'``) or as the very last
    rows (``'tail'``) -> the earlier flows are the train pool -> optional
    feature-level dedup -> ``train_cap`` random rows.

    The rolling features must ALREADY be in the file (computed on the whole
    time-ordered pcap by ``additional_features``); this function never
    computes or re-computes them, and never writes in place: outputs go to
    ``<out_dir>/train/<stem>_train.csv`` and ``<out_dir>/test/<stem>_test.csv``
    (``out_dir`` defaults to ``<dir of file>/split``).

    Returns a dict of counts for the manifest.  Legacy keyword names
    ``test_sample`` / ``Number_in_individaul_class`` map onto ``test_cap`` /
    ``train_cap``; ``temporal=False`` (a random row split) is kept only for
    experiments that want to *measure* the leak of the original protocol.
    """
    if test_sample is not None:
        test_cap = test_sample
    if Number_in_individaul_class is not None:
        train_cap = Number_in_individaul_class
    if legacy:
        warnings.warn("split_csv: ignoring unknown kwargs %s" % sorted(legacy))

    df = pd.read_csv(file_path, low_memory=False)
    stem = os.path.basename(file_path).rsplit('.', 1)[0]
    info = resolve_class_subtype(file_path)
    n_raw = len(df)

    if mac_filter_on:
        df = mac_filter(df, is_benign=(info.class_code == 'Benign'), attacker_macs=attacker_macs)
    n_after_mac = len(df)

    rng = np.random.default_rng(seed)
    if temporal:
        if TIME_COL not in df.columns:
            raise KeyError("%s lacks %s; a temporal split is impossible" % (file_path, TIME_COL))
        df = df.sort_values(TIME_COL, kind="mergesort").reset_index(drop=True)
        n_pool = int(round(len(df) * test_frac))
        test_pool = df.iloc[len(df) - n_pool:]
        train = df.iloc[:len(df) - n_pool]
        n_test = min(test_cap, n_pool) if test_cap else n_pool
        if test_pick == "tail" or n_test >= n_pool:
            test = test_pool.iloc[len(test_pool) - n_test:]
        else:
            idx = np.sort(rng.choice(n_pool, size=n_test, replace=False))
            test = test_pool.iloc[idx]
    else:
        n_pool = int(round(len(df) * test_frac))
        idx = rng.permutation(len(df))
        n_test = min(test_cap, n_pool) if test_cap else n_pool
        test = df.iloc[np.sort(idx[:n_test])]
        train = df.iloc[np.sort(idx[n_pool:])]

    n_train_pool = len(train)
    if dedup == "feature" and len(train):
        keys = row_keys(train)
        train = train.loc[~pd.Series(keys, index=train.index).duplicated().to_numpy()]
    n_train_unique = len(train)
    if train_cap and len(train) > train_cap:
        idx = np.sort(rng.choice(len(train), size=train_cap, replace=False))
        train = train.iloc[idx]

    out_dir = out_dir or os.path.join(os.path.dirname(os.path.abspath(file_path)), "split")
    train_path = os.path.join(out_dir, "train", stem + "_train.csv")
    test_path = os.path.join(out_dir, "test", stem + "_test.csv")
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_path), exist_ok=True)
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    return {"stem": stem, "class": info.class_code, "subtype": info.short,
            "n_raw": int(n_raw), "n_after_mac": int(n_after_mac),
            "n_test_pool": int(n_pool), "n_test": int(len(test)),
            "n_train_pool": int(n_train_pool), "n_train_unique": int(n_train_unique),
            "n_train": int(len(train)), "train_path": train_path, "test_path": test_path}


# ===========================================================================
# Combining_classes: per-class train/test with dedup, proportional caps and
# train-only oversampling (paper Sec. 3.2.1, Table 4)
# ===========================================================================
def _discover_split_files(directory):
    d = os.path.normpath(directory)
    train_files = sorted(glob.glob(os.path.join(d, "train", "*_train.csv")))
    test_files = sorted(glob.glob(os.path.join(d, "test", "*_test.csv")))
    if not train_files:  # legacy layout: <dir>/<stem>.csv (+ <dir>/test/<stem>_test.csv)
        train_files = sorted(f for f in glob.glob(os.path.join(d, "*.csv"))
                             if not os.path.basename(f).startswith("df_class_8"))
        test_files = sorted(glob.glob(os.path.join(d, "test", "*.csv")))
    return train_files, test_files


def _read_tagged(files_with_subtype):
    parts = []
    for path, short in files_with_subtype:
        df = pd.read_csv(path, low_memory=False)
        if len(df):
            df["__subtype"] = short
            df["__src_file"] = os.path.basename(path)
            parts.append(df)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def Combining_classes(directory, classes_list=None, Number_in_individaul_class=20000,
                      Number_of_test_samples=4000, label_dict=None, oversample=True,
                      subtype_alloc="proportional", dedup=True, dedup_test_within=False,
                      seed=42, out_dir=None, write_class_weights=True):
    """Build one train and one test CSV per class from the per-pcap split files.

    Three passes (all classes at once, so the test-vs-train check is global):

    1. train: concat the class's sub-attack files, tag ``__subtype``, drop
       duplicate feature rows, undersample to ``Number_in_individaul_class``
       proportionally per sub-attack, remember the row keys;
    2. test : concat, drop every row whose key appears in ANY class's train,
       (optionally drop test-internal duplicates), cap to
       ``Number_of_test_samples`` proportionally -- never duplicated;
    3. train: if ``oversample`` and the class is below the target, duplicate
       rows proportionally per sub-attack up to the target (paper Table 4).
       ``class_weights.json`` is always written from the counts BEFORE this
       step so the class-weight variant stays available (train.py --class-weights).

    Outputs ``<out_dir>/train/<Class>_train.csv`` and ``<out_dir>/test/<Class>_test.csv``
    with a ``Label`` column (``out_dir`` defaults to ``<parent of directory>/combined``).
    Returns the per-class / per-sub-attack counts.
    """
    labels = _with_label_aliases(label_dict)
    train_files, test_files = _discover_split_files(directory)
    if not train_files:
        raise FileNotFoundError("no split CSVs under %s" % directory)
    assert_all_resolvable(train_files + test_files)
    by_class_train = group_files_by_class(train_files)
    by_class_test = group_files_by_class(test_files)

    wanted = list(by_class_train) if not classes_list else \
        [LABEL_ALIASES.get(c, c) for c in classes_list]
    missing = [c for c in wanted if c not in by_class_train]
    if missing:
        warnings.warn("Combining_classes: no train files for classes %s (skipped)" % missing)
    wanted = [c for c in wanted if c in by_class_train]
    unknown = [c for c in wanted if c not in labels]
    if unknown:
        raise KeyError("classes without a label: %s" % unknown)

    out_dir = out_dir or os.path.join(os.path.dirname(os.path.normpath(directory)), "combined")
    os.makedirs(os.path.join(out_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "test"), exist_ok=True)
    rng = np.random.default_rng(seed)
    report = {}
    train_frames, train_keys = {}, set()

    # ---- pass 1: train ---------------------------------------------------------
    for cls in tqdm(wanted, desc="[Combining_classes] train"):
        tr = _read_tagged(by_class_train[cls])
        n_pool = len(tr)
        if dedup and n_pool:
            keys = row_keys(tr)
            tr = tr.loc[~pd.Series(keys).duplicated().to_numpy()].reset_index(drop=True)
        n_unique = len(tr)
        if subtype_alloc == "proportional":
            tr = proportional_cap(tr, Number_in_individaul_class, rng=rng)
        elif Number_in_individaul_class and len(tr) > Number_in_individaul_class:
            tr = tr.sample(n=Number_in_individaul_class, random_state=seed).reset_index(drop=True)
        train_keys.update(row_keys(tr).tolist())
        train_frames[cls] = tr
        report[cls] = {"train": {"pool": int(n_pool), "unique": int(n_unique), "capped": int(len(tr))},
                       "subtypes": {}}
        for sub, cnt in tr["__subtype"].value_counts().items():
            report[cls]["subtypes"].setdefault(sub, {})["train"] = int(cnt)

    # ---- pass 2: test ----------------------------------------------------------
    for cls in tqdm(wanted, desc="[Combining_classes] test "):
        te = _read_tagged(by_class_test.get(cls, []))
        n_pool = len(te)
        leaked = 0
        within = 0
        if n_pool:
            ek = row_keys(te)
            keep = ~np.fromiter((k in train_keys for k in ek.tolist()), dtype=bool, count=len(ek))
            leaked = int((~keep).sum())
            if dedup_test_within:
                dup = pd.Series(ek).duplicated().to_numpy()
                within = int((keep & dup).sum())
                keep &= ~dup
            te = te.loc[keep].reset_index(drop=True)
        if subtype_alloc == "proportional":
            te = proportional_cap(te, Number_of_test_samples, rng=rng)
        elif Number_of_test_samples and len(te) > Number_of_test_samples:
            te = te.sample(n=Number_of_test_samples, random_state=seed).reset_index(drop=True)
        report[cls]["test"] = {"pool": int(n_pool), "leaked_removed": leaked,
                               "within_dups_removed": within, "capped": int(len(te))}
        if len(te):
            for sub, cnt in te["__subtype"].value_counts().items():
                report[cls]["subtypes"].setdefault(sub, {})["test"] = int(cnt)
            te = te.copy()
            te["Label"] = labels[cls]
            te = te.drop(columns=[c for c in HELPER_COLUMNS if c in te.columns])
            te.to_csv(os.path.join(out_dir, "test", cls + "_test.csv"), index=False)
        else:
            warnings.warn("Combining_classes: class %s has no test rows" % cls)

    # ---- class weights from the pre-oversample unique counts -------------------
    if write_class_weights:
        counts = {labels[cls]: report[cls]["train"]["capped"] for cls in wanted}
        total = sum(counts.values())
        weights = {int(c): (float(total / (len(counts) * n)) if n else 0.0) for c, n in counts.items()}
        with open(os.path.join(out_dir, "class_weights.json"), "w") as fh:
            json.dump(dict(sorted(weights.items())), fh, indent=2)

    # ---- pass 3: oversample train ----------------------------------------------
    for cls in wanted:
        tr = train_frames[cls]
        if oversample:
            tr = oversample_train(tr, Number_in_individaul_class, rng=rng)
        report[cls]["train"]["oversampled"] = int(len(tr))
        tr = tr.copy()
        tr["Label"] = labels[cls]
        tr = tr.drop(columns=[c for c in HELPER_COLUMNS if c in tr.columns])
        tr.to_csv(os.path.join(out_dir, "train", cls + "_train.csv"), index=False)
        print(f"[Combining_classes] {cls}: train {report[cls]['train']} | test {report[cls]['test']}")
    return report


# ===========================================================================
# build_class8_csvs: the notebook's concat + 29-column drop, with a header check
# ===========================================================================
def build_class8_csvs(combined_dir, out_dir, drop_columns=IDENTIFIER_DROP_29, keep_l7=False,
                      expected_header=CLASS8_HEADER_97, strict_header=True,
                      train_name="df_class_8_train.csv", test_name="df_class_8_test.csv"):
    """``combined/{train,test}/*.csv`` -> ``df_class_8_{train,test}.csv``.

    Drops the 29 identifier / redundant columns (Data_preprocessing notebook
    cells 12/14), the NFStream L7 strings (unless ``keep_l7``) and helper
    columns, moves ``Label`` last, and -- when ``strict_header`` -- asserts the
    header equals the authors' 97-column Drive layout so graphs and checkpoints
    stay compatible.  Returns per-split row counts and label counts.
    """
    os.makedirs(out_dir, exist_ok=True)
    result = {}
    for split, name in (("train", train_name), ("test", test_name)):
        files = sorted(glob.glob(os.path.join(combined_dir, split, "*.csv")))
        if not files:
            warnings.warn("build_class8_csvs: no %s files in %s" % (split, combined_dir))
            continue
        parts = [pd.read_csv(f, low_memory=False) for f in files]
        df = pd.concat(parts, ignore_index=True)
        drops = [c for c in drop_columns if c in df.columns]
        drops += [c for c in list(HELPER_COLUMNS) + SPLT_COLUMNS_3 if c in df.columns]
        if not keep_l7:
            drops += [c for c in L7_COLUMNS_9 if c in df.columns]
        df = df.drop(columns=drops)
        if "Label" not in df.columns:
            raise KeyError("Label column missing in %s files" % split)
        df = df[[c for c in df.columns if c != "Label"] + ["Label"]]
        bad = [c for c in df.columns if df[c].dtype == object and not c.startswith("udps.")
               and c not in L7_COLUMNS_9]
        if bad:
            raise ValueError("non-numeric columns would break NIDSDataset: %s" % bad)
        header = [c.strip() for c in df.columns]
        header_ok = header == list(expected_header)
        if strict_header and not header_ok:
            extra = [c for c in header if c not in expected_header]
            missing = [c for c in expected_header if c not in header]
            raise AssertionError("class-8 %s header differs from the authors' 97-column layout: "
                                 "extra=%s missing=%s (order also matters)" % (split, extra, missing))
        out = os.path.join(out_dir, name)
        df.to_csv(out, index=False)
        result[split] = {"rows": int(len(df)), "n_cols": int(df.shape[1]), "header_ok": bool(header_ok),
                         "label_counts": {int(k): int(v) for k, v in df["Label"].value_counts().sort_index().items()},
                         "path": out}
        print(f"[build_class8_csvs] {split}: {len(df)} rows x {df.shape[1]} cols -> {out}")
    return result
