import os
import time
import matplotlib.pyplot as plt
import pickle
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.serialization import add_safe_globals

from tqdm import tqdm

from model import TransformerSeq2Seq, TransformerEncoderOnly
from dataset import SeaMotionDataset
from processing import save_checkpoint


TRAIN_PATH = os.environ.get("TRAIN_PATH", "../data/jowp_train.pt")
TEST_PATH = os.environ.get("TEST_PATH", "../data/jowp_test.pt")
SAVE_DIR = os.environ.get("SAVE_DIR", "../models/")

add_safe_globals([SeaMotionDataset])


class Learning:
    def __init__(self):
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")

        self.best_acc = 100000
        self.latest_epoch = 0
        self.epochs = 100000000
        self.lr = 1e-6
        self.num_gpus = num_gpus
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_dataset = torch.load(TRAIN_PATH, weights_only=False)
        test_dataset = torch.load(TEST_PATH, weights_only=False)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=16,
            shuffle=True,
            drop_last=True)

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=4,
            shuffle=False,
            drop_last=True)

        self.model = TransformerEncoderOnly(input_dim=5, output_dim=3, d_model=256).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer,
                                                           base_lr=self.lr,
                                                           max_lr=self.lr * 10,
                                                           step_size_up=50,
                                                           step_size_down=100,
                                                           mode='triangular')

        self.criterion = nn.MSELoss(reduction="none")

        if (self.device.type == 'cuda') and (self.num_gpus > 1):
            self.model = nn.DataParallel(self.model, device_ids=list(range(self.num_gpus)))

        pth_list = [pth for pth in os.listdir(SAVE_DIR) if pth.endswith('.pth')]
        if pth_list:
            latest = max(pth_list, key=lambda f: os.path.getmtime(os.path.join(SAVE_DIR, f)))
            latest_path = os.path.join(SAVE_DIR, latest)
            checkpoint = torch.load(latest_path, weights_only=True)
            self.latest_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"Loaded checkpoint from {latest_path}")
        else:
            print(f"No checkpoint found at {SAVE_DIR}")

    def autoregressive_decode(self, x, y_start, steps):
        preds = []
        y_in = y_start
        for _ in range(steps):
            out = self.model(x, y_in)       # (B, cur_len, F_out)
            next_pred = out[:, -1:, :]      # 마지막 step만 추출
            preds.append(next_pred)
            y_in = torch.cat([y_in, next_pred], dim=1)  # decoder 입력 확장
        return torch.cat(preds, dim=1)      # (B, steps, F_out)

    def train_loop(self):
        for t in range(self.epochs):

                losses = 0.0
            #
            # with tqdm(self.train_loader, unit="batch", ncols=100, ascii=" *") as pbar:
            #     pbar.set_description('Epoch: {}'.format(t))

                for x, y in self.train_loader:
                    x, y = x.to(self.device), y.to(self.device)

                    pred = self.model(x)  # (B, 2, 3)

                    self.optimizer.zero_grad()
                    loss = self.criterion(pred, y)
                    loss.backward()
                    self.optimizer.step()
                    losses += float(loss.item())

                    # lr = self.optimizer.param_groups[0]['lr']
                    # pbar.set_postfix(loss=losses, lr=lr)

                self.scheduler.step()

                if t % 1000 == 0:
                    state = {
                        'epoch': t,
                        'model': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict()
                    }

                    save_checkpoint(state, checkpoint_dir=SAVE_DIR, max_checkpoints=5)
                    val_acc = self.test_loop()

                    if val_acc < self.best_acc:
                        self.best_acc = val_acc
                        torch.save(state, os.path.join(SAVE_DIR, "best.pth"))
                        print(f"New best model saved with accuracy: %.4f" % self.best_acc)

                if t % 500 == 0:
                    self.evaluate_and_plot(t, save_path=SAVE_DIR)

    def test_loop(self):
        self.model.eval()
        losses = 0.0
        loss_list, trend_acc_list = [], []
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)

                pred = self.model(x)  # (B, 2, 3)
                loss = self.criterion(pred, y)
                losses += float(loss.item())

                # loss_raw = self.criterion(pred, y)  # [B, T, F]
                # loss_time_feature = loss_raw.mean(dim=0)  # [T, F]
                # loss_list.append(loss_time_feature.cpu())
                #
                # dy_true = y[:, 1, :] - y[:, 0, :]  # [B, F]
                # dy_pred = pred[:, 1, :] - pred[:, 0, :]  # [B, F]
                #
                # # 방향 일치 여부 (sign 비교)
                # trend_correct = (torch.sign(dy_true) == torch.sign(dy_pred)).float()  # [B, F]
                #
                # # 배치 평균 정확도
                # trend_acc = trend_correct.mean(dim=0)  # feature별 정확도
                # trend_acc_list.append(trend_acc.cpu())


        time.sleep(0.5)
        print(f"\nValidation Loss: %.8f" % losses)

        # loss_time_feature = torch.stack(loss_list).mean(dim=0)
        # trend_acc_all = torch.stack(trend_acc_list).mean(dim=0)  # [F]
        #
        # print(loss_time_feature)
        # print(trend_acc_all)

        return losses

    def evaluate_and_plot(self, t, save_path=None, motion_cols=None, motion_scaler_path="motion_scaler.pkl"):
        if motion_cols is None:
            motion_cols = ['roll', 'pitch', 'yaw']

        self.model.eval()
        preds, trues = [], []

        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)  # (B, 2, 3)

                preds.append(pred.cpu())
                trues.append(y.cpu())

        preds = torch.cat(preds, dim=0).numpy()
        trues = torch.cat(trues, dim=0).numpy()

        preds_seq = preds.reshape(-1, preds.shape[-1])
        trues_seq = trues.reshape(-1, trues.shape[-1])

        with open(motion_scaler_path, "rb") as f:
            motion_scaler = pickle.load(f)

        preds_seq = motion_scaler.inverse_transform(preds_seq)
        trues_seq = motion_scaler.inverse_transform(trues_seq)
        mse = np.mean((preds_seq - trues_seq) ** 2, axis=0)
        rmse = np.sqrt(mse)

        time_axis = range(len(preds_seq))

        plt.figure(figsize=(12, 6))
        for f in range(2, 3):
            plt.plot(time_axis, trues_seq[:, f], label=(motion_cols[f] if motion_cols else f"motion{f}") + " true")
            plt.plot(time_axis, preds_seq[:, f], '--',
                     label=(motion_cols[f] if motion_cols else f"motion{f}") + " pred")

            plt.fill_between(time_axis,
                             trues_seq[:, f] - rmse[f],
                             trues_seq[:, f] + rmse[f],
                             alpha=0.2, label=f"{motion_cols[f]} ±RMSE")

        plt.title(f"[{t}] Prediction vs Ground Truth (Time Series)")
        plt.xlabel("Time step")
        plt.ylabel("Motion values")
        plt.legend()
        plt.grid(True)
        # plt.show()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path + f'/{t}.png', dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

    def test(self):
        self.test_loop()


def train_fast():
    best_acc = 10000
    epochs = 10000
    lr = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = torch.load(TRAIN_PATH, weights_only=False)
    test_dataset = torch.load(TEST_PATH, weights_only=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        drop_last=True)

    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=True,
        drop_last=True)

    model = TransformerSeq2Seq(input_dim=5, output_dim=3, d_model=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=lr,
        max_lr=lr * 10,
        step_size_up=50,
        step_size_down=100,
        mode='triangular'
    )

    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            y_in = torch.cat([y[:, :1, :], y[:, :-1, :]], dim=1)
            optimizer.zero_grad()
            pred = model(x, y_in)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total += float(loss.item())

        scheduler.step()

        if epoch % 100 == 0:
            print(f"[{epoch}] Loss: {total:.4f}")

            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }

            save_checkpoint(state, checkpoint_dir=SAVE_DIR, max_checkpoints=5)
            model.eval()
            val_losses = 0.0

            with torch.no_grad():
                for step, (x, y) in enumerate(test_loader):
                    x = x.to(device)
                    y = y.to(device)

                    y_in = torch.cat([y[:, :1, :], y[:, :-1, :]], dim=1)
                    pred = model(x, y_in)
                    loss = criterion(pred, y)
                    val_losses += float(loss.item())

            time.sleep(0.5)
            print(f"[{epoch}] Val Loss: {val_losses:.4f}")

            if val_losses < best_acc:
                best_acc = val_losses
                torch.save(state, os.path.join(SAVE_DIR, "best.pth"))
                print(f"New best model saved with accuracy: %.4f" % best_acc)
#
def main():
    runner = Learning()
    # runner.train_loop()
    runner.test()
    # runner.evaluate_and_plot(20000, save_path=SAVE_DIR)
    #
    # train_fast()


if __name__ == "__main__":
    main()