#!/usr/bin/env python3
"""Package the GNN4ID / XG-NID preprocessing code (no data, no checkpoints) as

    dist/gnn4id_preprocess_<gitsha>_<YYYYMMDD>.zip

so it can be unpacked and run on another machine:

    unzip gnn4id_preprocess_*.zip -d GNN4ID && cd GNN4ID
    conda create -n xmfgnn python=3.10 && conda activate xmfgnn && pip install -r requirement.txt
    python run_preprocessing.py --pcap-dir <CIC_IoT2023/PCAP> --out-dir <out>

Notebook outputs are stripped before packaging.  A MANIFEST.txt (sha256 per
file + git HEAD) is written into the zip.
"""
import argparse
import hashlib
import json
import os
import subprocess
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))

INCLUDE_FILES = [
    "run_preprocessing.py", "build_graphs.py", "clean_existing_data.py", "train.py",
    "train_tuned.py", "explain.py", "make_preprocess_zip.py",
    "README.md", "requirement.txt", "LICENSE", ".gitignore",
    "docs/BAOCAO_PREPROCESS_V2.md", "docs/FEATURES_VA_Y_NGHIA.md",
    "GNN4ID.ipynb", "Data_preprocessing_CIC-IoT2023.ipynb", "GNN4ID_Model.ipynb",
    "1_GNN4ID_pcap.ipynb", "2_Data_preprocessing_pcap.ipynb", "3_GNN4ID_Model_pcap.ipynb",
]
INCLUDE_DIRS = ["Utility", "Debug", "tests", ".vscode"]
EXCLUDE_DIR_NAMES = {"__pycache__", ".git", "data", "checkpoints", "logs", "docs", "dist"}
EXCLUDE_SUFFIXES = (".pyc", ".pt", ".pth", ".pdf", ".log", ".trace", ".csv", ".pcap", ".pcapng")


def _git_sha():
    try:
        return subprocess.check_output(["git", "-C", HERE, "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


def _strip_notebook(raw):
    nb = json.loads(raw.decode("utf-8"))
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    return json.dumps(nb, indent=1, ensure_ascii=False).encode("utf-8")


def collect():
    files = []
    for f in INCLUDE_FILES:
        p = os.path.join(HERE, f)
        if os.path.isfile(p):
            files.append(f)
    for d in INCLUDE_DIRS:
        for root, dirs, names in os.walk(os.path.join(HERE, d)):
            dirs[:] = [x for x in dirs if x not in EXCLUDE_DIR_NAMES]
            for n in names:
                if n.endswith(EXCLUDE_SUFFIXES):
                    continue
                files.append(os.path.relpath(os.path.join(root, n), HERE))
    return sorted(set(files))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "dist"))
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    sha = _git_sha()
    name = args.name or "gnn4id_preprocess_%s_%s.zip" % (sha, time.strftime("%Y%m%d"))
    out = os.path.join(args.out_dir, name)
    files = collect()
    manifest = ["git HEAD: %s" % sha, "packaged: %s" % time.strftime("%Y-%m-%d %H:%M:%S"), ""]
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in files:
            with open(os.path.join(HERE, rel), "rb") as fh:
                data = fh.read()
            if rel.endswith(".ipynb"):
                data = _strip_notebook(data)
            zf.writestr(rel, data)
            manifest.append("%s  %s" % (hashlib.sha256(data).hexdigest(), rel))
        zf.writestr("MANIFEST.txt", "\n".join(manifest) + "\n")
    print("wrote %s (%d files, %.1f MB)" % (out, len(files), os.path.getsize(out) / 1e6))
    for rel in files:
        print("   ", rel)


if __name__ == "__main__":
    main()
