import os
import sys

import pickle
import pandas as pd
import numpy as np
import torch
from networkx.drawing import kamada_kawai_layout
from torch.utils.data import Dataset
from datetime import datetime, timedelta
from io import StringIO
from sklearn.preprocessing import StandardScaler


try:
    import requests
except Exception:
    requests = None

SEA_COLS = ["WH_SIG", "WP", "WO", "WS1", "WD1"]
MOTION_COLS = ["roll", "pitch", "yaw"]


def load_all_mru_data(folder_path: str) -> pd.DataFrame:
    mru_dfs = []
    for fname in sorted(os.listdir(folder_path)):
        if fname.endswith(".dat"):
            fpath = os.path.join(folder_path, fname)
            df = request_mru(fpath)
            mru_dfs.append(df)
    if not mru_dfs:
        raise RuntimeError(f"No .dat files found in {folder_path}")
    return pd.concat(mru_dfs, ignore_index=True)


def request_mru(file_path: str) -> pd.DataFrame:
    with open(file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith("# Current local time"):
            timestamp_str = line.split(":", 1)[1].strip()
            start_time = datetime.strptime(timestamp_str, "%a %b %d %H:%M:%S %Y")
            break

    header_line_index = next(i for i, line in enumerate(lines) if line.startswith("variables="))
    columns = lines[header_line_index].replace("variables=", "").strip().split()

    data_lines = [
        line.strip()
        for line in lines[header_line_index + 1:]
        if not line.startswith("#") and line.strip() != ""
    ]

    data_text = "\n".join(data_lines)
    df = pd.read_csv(StringIO(data_text), sep=r'\s+', names=columns)
    df["timestamp"] = start_time + pd.to_timedelta(df["Time"], unit="s")
    df["tm_dt"] = df["timestamp"].dt.floor("30min")

    return df


def aggregate_mru_30min(mru_df: pd.DataFrame) -> pd.DataFrame:
    agg = mru_df.groupby("tm_dt")[["roll", "pitch", "yaw"]].mean().reset_index()
    return agg


def request_typ01(tm1: str, tm2: str, stn: int, token: str) -> pd.DataFrame:
    if requests is None:
        raise RuntimeError("requests is unavailable in this environment.")

    url = 'https://apihub.kma.go.kr/api/typ01/url/kma_buoy2.php?tm1=202307231200&tm2=202307241200&stn=0&help=1&authKey=qXRqQWWGQ7a0akFlhtO20Q'
    params = {
        "tm1": tm1,
        "tm2": tm2,
        "stn": stn,
        "help": 0,
        "authKey": token
    }

    columns = [
        "TM","STN","WD1","WS1","WS1_GST","WD2","WS2","WS2_GST",
        "PA","HM","TA","TW","WH_MAX","WH_SIG","WH_AVE","WP","WO"
    ]

    response = requests.get(url, params=params)
    csv_text = response.text

    lines = [
        line.strip() for line in csv_text.strip().splitlines()
        if not line.startswith("#")
    ]

    data_text = "\n".join(lines)

    df = pd.read_csv(StringIO(data_text.strip()), header=None)
    df = df.iloc[:, :17]
    df.columns = columns

    df['TM'] = pd.to_datetime(df['TM'], format='%Y%m%d%H%M')
    return df

def normalize_kma_30min(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tm_dt"] = df["TM"].dt.floor("30min")
    keep = ["tm_dt"] + SEA_COLS

    for c in SEA_COLS:
        if c not in df.columns:
            df[c] = np.nan
    df = df.sort_values(["tm_dt","TM"]).groupby("tm_dt", as_index=False).last()
    return df[keep]


def build_samples_from_aligned(kma_30: pd.DataFrame, mru_30: pd.DataFrame):
    kma_map = {row.tm_dt: row for row in kma_30.itertuples(index=False)}
    mru_map = {row.tm_dt: row for row in mru_30.itertuples(index=False)}

    inputs, targets = [], []

    all_times = sorted(set(kma_map.keys()) | set(mru_map.keys()))
    delta3h = timedelta(hours=3)
    for t in all_times:
        t_minus = t - delta3h
        t_plus  = t + delta3h

        if t_minus in kma_map and t in kma_map and t_plus in kma_map and t in mru_map and t_plus in mru_map:
            x = np.stack([
                np.array([getattr(kma_map[t_minus], c) for c in SEA_COLS], dtype=float),
                np.array([getattr(kma_map[t], c) for c in SEA_COLS], dtype=float),
                np.array([getattr(kma_map[t_plus], c) for c in SEA_COLS], dtype=float),
            ], axis=0)

            y = np.stack([
                np.array([getattr(mru_map[t], c) for c in MOTION_COLS], dtype=float),
                np.array([getattr(mru_map[t_plus], c) for c in MOTION_COLS], dtype=float),
            ], axis=0)

            if np.isfinite(x).all() and np.isfinite(y).all():
                inputs.append(x)
                targets.append(y)
    return inputs, targets


class SeaMotionDataset(Dataset):
    def __init__(self, inputs, targets, fit_scaler=True, sea_scaler=None, motion_scaler=None):
        inputs = np.asarray(inputs, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32)

        N, T_in, F1 = inputs.shape   # (N, 3, F1)
        _, T_out, F2 = targets.shape # (N, 2, F2)

        if fit_scaler or sea_scaler is None:
            self.sea_scaler = StandardScaler()
            inputs_2d = inputs.reshape(-1, F1)
            self.sea_scaler.fit(inputs_2d)
        else:
            self.sea_scaler = sea_scaler

        inputs_norm = self.sea_scaler.transform(inputs.reshape(-1, F1)).reshape(N, T_in, F1)

        if fit_scaler or motion_scaler is None:
            self.motion_scaler = StandardScaler()
            targets_2d = targets.reshape(-1, F2)
            self.motion_scaler.fit(targets_2d)
        else:
            self.motion_scaler = motion_scaler

        targets_norm = self.motion_scaler.transform(targets.reshape(-1, F2)).reshape(N, T_out, F2)

        # --- Torch Tensor 변환 ---
        self.inputs = torch.tensor(inputs_norm, dtype=torch.float32)
        self.targets = torch.tensor(targets_norm, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def build_dataset_kma_mru(MRU_DIR: str, stn: int, out_path: str, token: str, is_pkl: bool) -> str:
    mru_df = load_all_mru_data(MRU_DIR)
    mru_30 = aggregate_mru_30min(mru_df)
    print(mru_30)

    start_time = mru_df["timestamp"].min().floor("h") - pd.Timedelta(hours=1)
    end_time   = mru_df["timestamp"].max().ceil("h")  + pd.Timedelta(hours=1)
    tm1 = start_time.strftime("%Y%m%d%H%M")
    tm2 = end_time.strftime("%Y%m%d%H%M")
    print('start: {}, end: {}'.format(tm1, tm2))

    kma_raw = request_typ01(tm1, tm2, stn, token)
    kma_30  = normalize_kma_30min(kma_raw)
    print(kma_30)

    x, y = build_samples_from_aligned(kma_30, mru_30)
    print(len(x))

    if is_pkl:
        ds = SeaMotionDataset(x, y, fit_scaler=True)
        with open("sea_scaler.pkl", "wb") as f:
            pickle.dump(ds.sea_scaler, f)
        with open("motion_scaler.pkl", "wb") as f:
            pickle.dump(ds.motion_scaler, f)
    else:
        with open("sea_scaler.pkl", "rb") as f:
            sea_scaler = pickle.load(f)
        with open("motion_scaler.pkl", "rb") as f:
            motion_scaler = pickle.load(f)
        ds = SeaMotionDataset(x, y, fit_scaler=True, sea_scaler=sea_scaler, motion_scaler=motion_scaler)

    torch.save(ds, out_path)

    return out_path


def main():
    DATA_PATH = os.environ.get("DATA_PATH", "../data/jowp_train.pt")
    MRU_DIR = os.environ.get("MRU_DIR", "../data/mru_train")
    KMA_STN = int(os.environ.get("KMA_STN", "22102"))
    TOKEN = os.environ.get("TOKEN", "qXRqQWWGQ7a0akFlhtO20Q")

    print("Building dataset from KMA + MRU...")
    out = build_dataset_kma_mru(MRU_DIR, KMA_STN, DATA_PATH, TOKEN, False)
    print(f"Saved dataset to {out}")
    return out


if __name__ == "__main__":
    main()
    sys.exit(0)