"""
fetch_uci_dataset.py - Downloads and converts the real UCI Seismic Bumps Dataset.

Sources:
- Canonical UCI: https://archive.ics.uci.edu/ml/machine-learning-databases/00266/seismic-bumps.arff
- GitHub Mirror: https://raw.githubusercontent.com/RoelBouman/outlierdetection/master/raw_data/seismic-bumps.arff
Target: data/uci_seismic_bumps.csv
"""

import os
import io
import urllib.request
import pandas as pd
from scipy.io import arff

DATA_DIR = "data"
PRIMARY_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00266/seismic-bumps.arff"
MIRROR_URL = "https://raw.githubusercontent.com/RoelBouman/outlierdetection/master/raw_data/seismic-bumps.arff"
OUT_CSV = os.path.join(DATA_DIR, "uci_seismic_bumps.csv")


def fetch_arff_bytes():
    urls = [PRIMARY_URL, MIRROR_URL]
    for url in urls:
        try:
            print(f"[*] Trying to fetch from: {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                content = resp.read()
                if len(content) > 1000:
                    print(f"[*] Download successful ({len(content)} bytes)")
                    return content
        except Exception as e:
            print(f"[!] Fetch failed for {url}: {e}")
    raise RuntimeError("Could not download seismic-bumps.arff from any source.")


def process_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)
    raw_bytes = fetch_arff_bytes()
    text_content = io.StringIO(raw_bytes.decode("utf-8", errors="ignore"))
    
    data, meta = arff.loadarff(text_content)
    df = pd.DataFrame(data)

    # Decode byte strings in categorical columns
    for col in df.select_dtypes(include=[object]):
        df[col] = df[col].apply(lambda x: x.decode("utf-8") if isinstance(x, bytes) else str(x))

    # Target class: '1' -> 1 (hazardous bump), '0' -> 0 (normal shift)
    df["hazard_label"] = df["class"].astype(str).str.strip().apply(lambda x: 1 if x in ("1", "b'1'") else 0)
    df = df.drop(columns=["class"])

    df.to_csv(OUT_CSV, index=False)
    print(f"[*] Real UCI Seismic Bumps dataset saved to: {OUT_CSV}")
    print(f"    Total samples: {len(df)}")
    print(f"    Hazardous bumps: {df['hazard_label'].sum()} ({df['hazard_label'].mean():.2%})")
    print(f"    Features: {list(df.columns[:8])}...")
    return df


if __name__ == "__main__":
    process_dataset()
