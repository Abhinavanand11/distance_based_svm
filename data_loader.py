"""Load the UCI Banknote Authentication and Rice (Cammeo/Osmancik) datasets.

Files are looked up in --data-dir first; if absent they are downloaded from UCI.
If your machine is offline, download the files manually into the data directory:
  banknote : data_banknote_authentication.txt  (UCI dataset 267)
  rice     : Rice_Cammeo_Osmancik.arff         (UCI dataset 545)
"""
from __future__ import annotations

import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SPECS = {
    "banknote": dict(
        expected_shape=(1372, 4),
        patterns=["data_banknote_authentication.txt", "*banknote*.txt", "*banknote*.csv"],
        urls=["https://archive.ics.uci.edu/static/public/267/banknote+authentication.zip",
              "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"],
        columns=["variance", "skewness", "curtosis", "entropy", "class"],
    ),
    "rice": dict(
        expected_shape=(3810, 7),
        patterns=["*.arff"],
        urls=["https://archive.ics.uci.edu/static/public/545/rice+cammeo+and+osmancik.zip"],
        columns=None,   # read from the ARFF header
    ),
}


@dataclass
class Dataset:
    name: str
    X: pd.DataFrame          # features only (target removed)
    y: np.ndarray            # 0/1 integer labels
    class_names: list
    n_dropped: int           # rows removed because of missing values


def _find_local(data_dir: Path, patterns):
    for pat in patterns:
        hits = sorted(data_dir.glob(pat))
        if hits:
            return hits[0]
    return None


def _fetch(spec, data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    found = _find_local(data_dir, spec["patterns"])
    if found:
        return found
    errors = []
    for url in spec["urls"]:
        try:
            print(f"Downloading {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                dest = data_dir / url.rsplit("/", 1)[-1]
                dest.write_bytes(resp.read())
            if dest.suffix == ".zip":
                with zipfile.ZipFile(dest) as z:
                    z.extractall(data_dir)
            found = _find_local(data_dir, spec["patterns"])
            if found:
                return found
        except Exception as exc:  # network errors, 404, bad zip ...
            errors.append(f"{url}: {exc}")
    raise RuntimeError("Could not obtain the dataset. Place the file manually in "
                       f"'{data_dir}' (see data_loader.py docstring).\n" + "\n".join(errors))


def _read_arff(path: Path) -> pd.DataFrame:
    names, rows, in_data = [], [], False
    for line in path.read_text(errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("%"):
            continue
        low = s.lower()
        if low.startswith("@attribute"):
            names.append(s.split(None, 2)[1].strip("'\""))
        elif low.startswith("@data"):
            in_data = True
        elif in_data:
            rows.append(s)
    df = pd.read_csv(io.StringIO("\n".join(rows)), header=None, names=names)
    last = df.columns[-1]
    df[last] = df[last].astype(str).str.strip().str.strip("'\"")
    return df


def load_dataset(name: str, data_dir: str = "data") -> Dataset:
    if name not in SPECS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {list(SPECS)}")
    spec = SPECS[name]
    path = _fetch(spec, Path(data_dir))
    if name == "banknote":
        df = pd.read_csv(path, header=None, names=spec["columns"])
    else:
        df = _read_arff(path)

    print(f"[{name}] file: {path.name}")
    print(f"[{name}] raw shape (rows, columns incl. target): {df.shape}")
    print(f"[{name}] columns: {list(df.columns)}")

    target_col = df.columns[-1]                       # target is the last column in both files
    feats = df.drop(columns=[target_col]).apply(pd.to_numeric, errors="coerce")
    if (df.shape[0], feats.shape[1]) != spec["expected_shape"]:
        raise ValueError(f"[{name}] expected {spec['expected_shape']} (instances, features) "
                         f"but got {(df.shape[0], feats.shape[1])}. Wrong file?")

    bad = feats.isna().any(axis=1) | df[target_col].isna()
    n_dropped = int(bad.sum())
    if n_dropped:
        print(f"[{name}] dropping {n_dropped} rows with missing values")
    feats, labels = feats[~bad].reset_index(drop=True), df.loc[~bad, target_col].astype(str).reset_index(drop=True)

    class_names = sorted(labels.unique())
    if len(class_names) != 2:
        raise ValueError(f"[{name}] expected 2 classes, found {class_names}")
    y = labels.map({c: i for i, c in enumerate(class_names)}).to_numpy(dtype=int)
    assert target_col not in feats.columns              # target never in the feature matrix
    print(f"[{name}] features: {list(feats.columns)}")
    print(f"[{name}] classes (0,1): {class_names}  counts: {np.bincount(y).tolist()}")
    return Dataset(name, feats, y, class_names, n_dropped)
