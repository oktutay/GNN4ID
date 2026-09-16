"""Unit tests for the CIC-IoT2023 file-name resolver."""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from Utility.Functions import (  # noqa: E402
    resolve_class_subtype, canonical_stem, assert_all_resolvable, group_files_by_class)
from Utility.Schema import CIC_SUBTYPES  # noqa: E402


class TestResolver(unittest.TestCase):

    def test_hard_names(self):
        cases = {
            "DDoS-SlowLoris1.pcap": ("DDos", "SlowLoris"),
            "DDos-SlowLoris": ("DDos", "SlowLoris"),           # authors' misspelt canonical
            "BrowserHijacking.pcap": ("WebBased", "BrwserHijack"),
            "Webbased-BrwserHijack_0.csv": ("WebBased", "BrwserHijack"),
            "BenignTraffic3.pcap": ("Benign", "Benign"),
            "BenignTraffic.pcap": ("Benign", "Benign"),
            "Benign_Final": ("Benign", "Benign"),
            "Benign-Benign_2_train.csv": ("Benign", "Benign"),
            "DDoS-ICMP_Fragmentation2": ("DDos", "ICMPFrg"),
            "DDoS-ICMP_Flood10": ("DDos", "ICMPFlood"),
            "DDoS-UDP_Flood.pcap": ("DDos", "UDPFlood"),
            "DDoS-UDP_Fragmentation.pcap": ("DDos", "UDPFrg"),
            "DoS-UDP_Flood": ("Dos", "UDPFlood"),
            "DoS-SYN_Flood3.pcap": ("Dos", "SYNFlood"),
            "DDoS-SYN_Flood3.pcap": ("DDos", "SYNFlood"),
            "Recon-PortScan1.pcap": ("Recon", "PortScan"),
            "Recon-PingSweep.pcap": ("Recon", "PingSweep"),
            "VulnerabilityScan.pcap": ("Recon", "VulScan"),
            "WebBased-XSS_0_test.csv": ("WebBased", "XSS"),
            "XSS.pcap": ("WebBased", "XSS"),
            "DictionaryBruteForce.pcap": ("BruteForce", "Dictionary"),
            "Mirai-udpplain.pcap": ("Mirai", "UDPPlain"),
            "MITM-ArpSpoofing1.pcap": ("Spoofing", "ARP"),
            "/some/dir/DNS_Spoofing.pcap.gz": ("Spoofing", "DNS"),
        }
        for name, (cls, short) in cases.items():
            info = resolve_class_subtype(name)
            self.assertEqual((info.class_code, info.short), (cls, short), name)

    def test_every_cic_key_with_suffixes(self):
        for key, (cls, short) in CIC_SUBTYPES.items():
            for suf in ("", "1", "10", "_0", "_0_test", ".pcap", "3.pcap", ".csv", "_7_train.csv"):
                info = resolve_class_subtype(key + suf)
                self.assertEqual((info.class_code, info.short), (cls, short), key + suf)
            canon = "%s-%s" % (cls, short)
            for suf in ("", "_0", "_12_test.csv", ".pcap"):
                info = resolve_class_subtype(canon.lower() + suf)
                self.assertEqual(info.class_code, cls, canon + suf)

    def test_canonical_stem(self):
        self.assertEqual(canonical_stem("DDoS-ICMP_Flood10.pcap"), "DDos-ICMPFlood_10")
        self.assertEqual(canonical_stem("BenignTraffic.pcap"), "Benign-Benign_0")
        self.assertEqual(canonical_stem("BenignTraffic3.pcap"), "Benign-Benign_3")
        self.assertEqual(canonical_stem("WebBased-XSS_0_test.csv"), "WebBased-XSS_0")
        # idempotent
        self.assertEqual(canonical_stem(canonical_stem("DDoS-SlowLoris2.pcap") + ".pcap"), "DDos-SlowLoris_2")

    def test_unknown_raises(self):
        with self.assertRaises(KeyError):
            resolve_class_subtype("Foo.pcap")
        with self.assertRaises(ValueError):
            assert_all_resolvable(["XSS.pcap", "Foo.pcap", "Bar1.pcap"])
        assert_all_resolvable(["XSS.pcap", "BenignTraffic1.pcap"])

    def test_group_files_by_class(self):
        g = group_files_by_class(["a/XSS_train.csv", "a/SqlInjection_train.csv", "a/BenignTraffic1_train.csv"])
        self.assertEqual(sorted(g), ["Benign", "WebBased"])
        self.assertEqual(sorted(s for _, s in g["WebBased"]), ["SqlInject", "XSS"])


if __name__ == "__main__":
    unittest.main()
