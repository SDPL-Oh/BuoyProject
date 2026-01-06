import os
import requests
import pandas as pd
import numpy as np
import torch

from torch.utils.data import Dataset
from datetime import datetime
from io import StringIO



def request_typ06(tma_fc, tma_ef, lzone):
    """
     tma_fc: 발표시각 일 2회(00, 12UTC) 발표 / (포맷) yyyymmddhh
     tma_ef: 발효시각 +75시간까지 3시간 간격 예측자료 / (포맷) yyyymmddhh

    :return:
    """
    url = 'https://apihub.kma.go.kr/api/typ06/url/marine_small_zone.php?tma_fc=2025012500&tma_ef=2025012609&Lzone=1,3,4,5&Szone=0&disp=0&help=1&authKey=qXRqQWWGQ7a0akFlhtO20Q'

    params = {
        "tma_fc": tma_fc,
        "tma_ef": tma_ef,
        "Lzone": lzone,
        "Szone": 0,
        "disp": 0,
        "help": 0,
        "dataType": "JSON"
    }

    columns = [
        'TMA_FC', 'TMA_EF', 'LZONE', 'SZONE',
        'LAT_LB', 'LON_LB', 'LAT_LT', 'LON_LT',
        'LAT_RT', 'LON_RT', 'LAT_RB', 'LON_RB',
        'WH_SIG', 'WVPRD_MAX', 'WVDR', 'WS', 'WD'
    ]

    response = requests.get(url, params=params)
    csv_text = response.text

    lines = [
        line.strip() for line in csv_text.strip().splitlines()
        if not line.startswith("#")
    ]

    data_text = "\n".join(lines)

    df = pd.read_csv(StringIO(data_text.strip()), sep=r'\s+', names=columns)

    df['TMA_FC'] = pd.to_datetime(df['TMA_FC'], format='%Y%m%d%H')
    df['TMA_EF'] = pd.to_datetime(df['TMA_EF'], format='%Y%m%d%H')

    return df


def request_typ01(tm1, tm2, stn):
    url = 'https://apihub.kma.go.kr/api/typ01/url/kma_buoy2.php?tm1=202307231200&tm2=202307241200&stn=0&help=1&authKey=qXRqQWWGQ7a0akFlhtO20Q'

    params = {
        "tm1": tm1,
        "tm2": tm2,
        "stn": stn,
        "help": 0,
        "authKey": "qXRqQWWGQ7a0akFlhtO20Q"
    }

    columns = [
        "TM", "STN",
        "WD1", "WS1", "WS1_GST",
        "WD2", "WS2", "WS2_GST",
        "PA", "HM", "TA", "TW",
        "WH_MAX", "WH_SIG", "WH_AVE",
        "WP", "WO"
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


def request_mru(file_path):
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
    df["tm"] = df["timestamp"].dt.floor("30min").dt.strftime("%Y%m%d%H%M")

    return df


def load_all_mru_data(folder_path):
    mru_dfs = []
    for file_name in sorted(os.listdir(folder_path)):
        if file_name.endswith(".dat"):  # 또는 다른 확장자
            file_path = os.path.join(folder_path, file_name)
            mru_df = request_mru(file_path)
            mru_dfs.append(mru_df)
    return pd.concat(mru_dfs, ignore_index=True)


def prepare_dataset(typ_df, mru_df):
    agg_df = mru_df.groupby("tm")[["roll", "pitch", "yaw"]].mean().reset_index()

    typ_df["tm"] = typ_df["TM"].dt.strftime("%Y%m%d%H%M")
    sea_cols = ["WH_SIG", "WP", "WO", "WS1", "WD1"]
    typ_selected = typ_df[["tm"] + sea_cols]

    merged = pd.merge(typ_selected, agg_df, on="tm", how="inner")

    return merged


def make_samples(merged_df):
    samples, targets = [], []

    for i in range(1, len(merged_df) - 1 - 1):
        input_seq = merged_df.iloc[i - 1:i + 2][1:6].to_numpy().astype(float)
        output_seq = merged_df.iloc[i:i+2][["roll", "pitch", "yaw"]].to_numpy().astype(float)

        samples.append(input_seq)
        targets.append(output_seq)

    return samples, targets


def build_and_save_dataset(mru_folder, stn, output_path):
    mru_df = load_all_mru_data(mru_folder)

    start_time = mru_df["timestamp"].min().floor("h") - pd.Timedelta(hours=1)
    end_time = mru_df["timestamp"].max().ceil("h") + pd.Timedelta(hours=1)

    tm1 = start_time.strftime("%Y%m%d%H%M")
    tm2 = end_time.strftime("%Y%m%d%H%M")

    print(f"MRU time range: {tm1} ~ {tm2}")
    print("Requesting KMA API...")
    typ_df = request_typ01(tm1, tm2, stn)
    merged_df = prepare_dataset(typ_df, mru_df)

    x, y = make_samples(merged_df)
    print(x)
    dataset = SeaMotionDataset(x, y)
    torch.save(dataset, output_path)
    print("Finish")


class SeaMotionDataset(Dataset):
    def __init__(self, input_seqs, output_seqs):
        input_array = np.array(input_seqs)   # shape: (N, 3, 5)
        output_array = np.array(output_seqs) # shape: (N, 2, 3)

        self.inputs = torch.tensor(input_array, dtype=torch.float32)
        self.outputs = torch.tensor(output_array, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]


if __name__ == '__main__':
    # df_6 = request_typ06(2025080500, 2025080800, "193, 194. 203, 204")
    df_1 = request_typ01(202506161000, 202506171100, 22493)
    # df_2 = request_mru("MRU_20250425_070001.dat")
    print(df_1)

    # build_and_save_dataset('../data/mru_1', 22102, '../data/jowp_1.pt')

