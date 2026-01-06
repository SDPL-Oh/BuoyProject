import sys
import matplotlib.pyplot as plt

from dataset import load_all_mru_data, aggregate_mru_30min


def plot_mru_data(df, save_dir):
    plt.figure(figsize=(12,6))
    plt.plot(df["tm_dt"], df["roll"], label="Roll (deg)")
    plt.plot(df["tm_dt"], df["pitch"], label="Pitch (deg)")
    # plt.plot(df["tm_dt"], df["yaw"], label="Yaw (deg)")

    plt.xlabel("Time (s)")
    plt.ylabel("Angle (deg)")
    plt.title("MRU Motion Data")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir, dpi=300)
    plt.close()


def main():
    MRU_DIR = '../data/mru_train'
    SAVE_DIR = '../data/roll.png'

    mru_df = load_all_mru_data(MRU_DIR)
    mru_30 = aggregate_mru_30min(mru_df)
    plot_mru_data(mru_30, SAVE_DIR)

if __name__ == '__main__':
    main()
    sys.exit(0)