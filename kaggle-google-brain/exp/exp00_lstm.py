# default package
import os
import gc
import sys
import json
import time
import math
import random
from datetime import datetime
from collections import Counter, defaultdict
import pathlib
import logging

# third party package
import scipy as sp
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import category_encoders as ce
import wandb
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
from dotenv import load_dotenv

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    CosineAnnealingLR,
    ReduceLROnPlateau,
)

import pytorch_lightning as pl
from transformers import AdamW
from transformers import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

# from apex import amp

# global variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_logger = logging.getLogger(__name__)
if "KAGGLE_URL_BASE" in set(os.environ.keys()):
    from kaggle_secrets import UserSecretsClient

    ENV = "KAGGLE"
    USER_SECRETS = UserSecretsClient()
elif "google.colab" in sys.modules:
    ENV = "COLAB"
else:
    path = (pathlib.Path(os.path.dirname(__file__))).joinpath("../.env")
    load_dotenv(path)
    ENV = "LOCAL"


# environment variable
def load_env(label: str):
    if ENV in ["KAGGLE"]:
        return USER_SECRETS.get_secret(label)
    elif ENV in ["LOCAL", "COLAB"]:
        return os.environ.get(label)


WANDB_API = load_env("WANDB_API")
OUTPUT_DIR = load_env("OUTPUT_DIR")
INPUT_DIR = load_env("INPUT_DIR")


# =================================================
# Config
# =================================================
class CFG:
    competition = "ventilator"
    _wandb_kernel = "tmdiesel"
    log_dir = "./logs"
    apex = False
    seed = 42
    print_freq = 100
    num_workers = 4

    model_name = "rnn"
    scheduler = "CosineAnnealingLR"
    batch_scheduler = False
    T_max = 50
    epochs = 50
    max_grad_norm = 1000
    gradient_accumulation_steps = 1
    hidden_size = 64
    lr = 5e-3
    min_lr = 1e-6
    weight_decay = 1e-6
    batch_size = 64
    n_fold = 5
    trn_fold = [0, 1, 2, 3, 4]
    cate_seq_cols = ["R", "C"]
    cont_seq_cols = ["time_step", "u_in", "u_out"] + ["breath_time", "u_in_time"]
    train = True
    inference = True


# =================================================
# Utils
# =================================================
def init_root_logger(
    outdir: pathlib.Path,
    filename_normal: str = "log.log",
    filename_error: str = "error.log",
):

    outdir.mkdir(exist_ok=True)
    logging.getLogger().addHandler(_add_handler(outdir, logging.INFO, filename_normal))
    logging.getLogger().addHandler(_add_handler(outdir, logging.ERROR, filename_error))


def _add_handler(outdir: pathlib.Path, level, filename):
    fh = logging.FileHandler(outdir / filename)
    fh.setLevel(level)
    fh_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(filename)s - %(name)s - %(funcName)s - %(message)s"
    )
    fh.setFormatter(fh_formatter)
    return fh


def class2dict(f):
    return dict(
        (name, getattr(f, name)) for name in dir(f) if not name.startswith("__")
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return "%s (remain %s)" % (asMinutes(s), asMinutes(rs))


# =================================================
# Load data
# =================================================
def load_data():
    train = pd.read_csv(f"{INPUT_DIR}train.csv")
    test = pd.read_csv(f"{INPUT_DIR}test.csv")
    sub = pd.read_csv(f"{INPUT_DIR}sample_submission.csv")

    for c in ["u_in"]:
        train[c] = np.log1p(train[c])
        test[c] = np.log1p(test[c])

    r_map = {5: 0, 20: 1, 50: 2}
    c_map = {10: 0, 20: 1, 50: 2}
    train["R"] = train["R"].map(r_map)
    test["R"] = test["R"].map(r_map)
    train["C"] = train["C"].map(c_map)
    test["C"] = test["C"].map(c_map)
    return train, test, sub


# =================================================
# Feature
# =================================================
def add_feature(df):
    # breath_time
    df["breath_time"] = df["time_step"] - df["time_step"].shift(1)
    df.loc[df["time_step"] == 0, "breath_time"] = 0
    # u_in_time
    df["u_in_time"] = df["u_in"] - df["u_in"].shift(1)
    df.loc[df["time_step"] == 0, "u_in_time"] = 0
    return df


# =================================================
# Dataset
# =================================================
class TrainDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.groups = df.groupby("breath_id").groups
        self.keys = list(self.groups.keys())

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        indexes = self.groups[self.keys[idx]]
        df = self.df.iloc[indexes]
        cate_seq_x = torch.LongTensor(df[CFG.cate_seq_cols].values)
        cont_seq_x = torch.FloatTensor(df[CFG.cont_seq_cols].values)
        u_out = torch.LongTensor(df["u_out"].values)
        label = torch.FloatTensor(df["pressure"].values)
        return cate_seq_x, cont_seq_x, u_out, label


class TestDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.groups = df.groupby("breath_id").groups
        self.keys = list(self.groups.keys())

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        indexes = self.groups[self.keys[idx]]
        df = self.df.iloc[indexes]
        cate_seq_x = torch.LongTensor(df[CFG.cate_seq_cols].values)
        cont_seq_x = torch.FloatTensor(df[CFG.cont_seq_cols].values)
        return cate_seq_x, cont_seq_x


# =================================================
# Model
# =================================================
class CustomModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.hidden_size = cfg.hidden_size
        self.r_emb = nn.Embedding(3, 2, padding_idx=0)
        self.c_emb = nn.Embedding(3, 2, padding_idx=0)
        self.seq_emb = nn.Sequential(
            nn.Linear(4 + len(cfg.cont_seq_cols), self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.lstm = nn.LSTM(
            self.hidden_size,
            self.hidden_size,
            dropout=0.2,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size * 2),
            nn.LayerNorm(self.hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(self.hidden_size * 2, 1),
        )
        for n, m in self.named_modules():
            if isinstance(m, nn.LSTM):
                print(f"init {m}")
                for param in m.parameters():
                    if len(param.shape) >= 2:
                        nn.init.orthogonal_(param.data)
                    else:
                        nn.init.normal_(param.data)
            elif isinstance(m, nn.GRU):
                print(f"init {m}")
                for param in m.parameters():
                    if len(param.shape) >= 2:
                        init.orthogonal_(param.data)
                    else:
                        init.normal_(param.data)

    def forward(self, cate_seq_x, cont_seq_x):
        bs = cont_seq_x.size(0)
        r_emb = self.r_emb(cate_seq_x[:, :, 0]).view(bs, 80, -1)
        c_emb = self.c_emb(cate_seq_x[:, :, 1]).view(bs, 80, -1)
        seq_x = torch.cat((r_emb, c_emb, cont_seq_x), 2)
        seq_emb = self.seq_emb(seq_x)
        seq_emb, _ = self.lstm(seq_emb)
        output = self.head(seq_emb).view(bs, -1)
        return output  # (bs,80)


# =================================================
# Train
# =================================================
def train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device):
    model.train()
    losses = AverageMeter()
    start = end = time.time()
    for step, (cate_seq_x, cont_seq_x, u_out, y) in enumerate(train_loader):
        loss_mask = u_out == 0
        cate_seq_x, cont_seq_x, y = (
            cate_seq_x.to(device),
            cont_seq_x.to(device),
            y.to(device),
        )
        batch_size = cont_seq_x.size(0)
        pred = model(cate_seq_x, cont_seq_x)
        loss = 2.0 * criterion(pred[loss_mask], y[loss_mask]) + criterion(
            pred[loss_mask == 0], y[loss_mask == 0]
        )
        losses.update(loss.item(), batch_size)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        if CFG.apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), CFG.max_grad_norm
        )
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            if CFG.batch_scheduler:
                scheduler.step()
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(train_loader) - 1):
            print(
                "Epoch: [{0}][{1}/{2}] "
                "Elapsed {remain:s} "
                "Loss: {loss.val:.4f}({loss.avg:.4f}) "
                "Grad: {grad_norm:.4f}  "
                "LR: {lr:.6f}  ".format(
                    epoch + 1,
                    step,
                    len(train_loader),
                    remain=timeSince(start, float(step + 1) / len(train_loader)),
                    loss=losses,
                    grad_norm=grad_norm,
                    lr=scheduler.get_lr()[0],
                )
            )
        wandb.log(
            {
                f"[fold{fold}] loss": losses.val,
                f"[fold{fold}] lr": scheduler.get_lr()[0],
            }
        )
    return losses.avg


def valid_fn(valid_loader, model, criterion, device):
    model.eval()
    preds = []
    losses = AverageMeter()
    start = end = time.time()
    for step, (cate_seq_x, cont_seq_x, u_out, y) in enumerate(valid_loader):
        loss_mask = u_out == 0
        cate_seq_x, cont_seq_x, y = (
            cate_seq_x.to(device),
            cont_seq_x.to(device),
            y.to(device),
        )
        batch_size = cont_seq_x.size(0)
        with torch.no_grad():
            pred = model(cate_seq_x, cont_seq_x)
        loss = 2.0 * criterion(pred[loss_mask], y[loss_mask]) + criterion(
            pred[loss_mask == 0], y[loss_mask == 0]
        )
        losses.update(loss.item(), batch_size)
        preds.append(pred.view(-1).detach().cpu().numpy())
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(valid_loader) - 1):
            print(
                "EVAL: [{0}/{1}] "
                "Elapsed {remain:s} "
                "Loss: {loss.val:.4f}({loss.avg:.4f}) ".format(
                    step,
                    len(valid_loader),
                    remain=timeSince(start, float(step + 1) / len(valid_loader)),
                    loss=losses,
                )
            )
    preds = np.concatenate(preds)
    return losses.avg, preds


def inference_fn(test_loader, model, device):
    model.eval()
    model.to(device)
    preds = []
    tk0 = tqdm(enumerate(test_loader), total=len(test_loader))
    for step, (cate_seq_x, cont_seq_x) in tk0:
        cate_seq_x, cont_seq_x = cate_seq_x.to(device), cont_seq_x.to(device)
        with torch.no_grad():
            pred = model(cate_seq_x, cont_seq_x)
        preds.append(pred.view(-1).detach().cpu().numpy())
    preds = np.concatenate(preds)
    return preds


def get_score(y_trues, y_preds):
    score = mean_absolute_error(y_trues, y_preds)
    return score


def train_loop(folds, fold):

    _logger.info(f"========== fold: {fold} training ==========")

    # ====================================================
    # loader
    # ====================================================
    trn_idx = folds[folds["fold"] != fold].index
    val_idx = folds[folds["fold"] == fold].index

    train_folds = folds.loc[trn_idx].reset_index(drop=True)
    valid_folds = folds.loc[val_idx].reset_index(drop=True)
    y_true = valid_folds["pressure"].values
    non_expiratory_phase_val_idx = valid_folds[
        valid_folds["u_out"] == 0
    ].index  # The expiratory phase is not scored

    train_dataset = TrainDataset(train_folds)
    valid_dataset = TrainDataset(valid_folds)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # ====================================================
    # model & optimizer
    # ====================================================
    model = CustomModel(CFG)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    num_train_steps = int(len(train_folds) / CFG.batch_size * CFG.epochs)

    def get_scheduler(optimizer):
        if CFG.scheduler == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=CFG.num_warmup_steps,
                num_training_steps=num_train_steps,
            )
        elif CFG.scheduler == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=CFG.num_warmup_steps,
                num_training_steps=num_train_steps,
                num_cycles=CFG.num_cycles,
            )
        elif CFG.scheduler == "ReduceLROnPlateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=CFG.factor,
                patience=CFG.patience,
                verbose=True,
                eps=CFG.eps,
            )
        elif CFG.scheduler == "CosineAnnealingLR":
            scheduler = CosineAnnealingLR(
                optimizer, T_max=CFG.T_max, eta_min=CFG.min_lr, last_epoch=-1
            )
        elif CFG.scheduler == "CosineAnnealingWarmRestarts":
            scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=CFG.T_0, T_mult=1, eta_min=CFG.min_lr, last_epoch=-1
            )
        return scheduler

    scheduler = get_scheduler(optimizer)

    # ====================================================
    # apex
    # ====================================================
    if CFG.apex:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)

    # ====================================================
    # loop
    # ====================================================
    criterion = nn.L1Loss()

    best_score = np.inf

    for epoch in range(CFG.epochs):

        start_time = time.time()

        # train
        avg_loss = train_fn(
            fold, train_loader, model, criterion, optimizer, epoch, scheduler, device
        )

        # eval
        avg_val_loss, preds = valid_fn(valid_loader, model, criterion, device)

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        elif isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()
        elif isinstance(scheduler, CosineAnnealingWarmRestarts):
            scheduler.step()

        # scoring
        score = get_score(
            y_true[non_expiratory_phase_val_idx], preds[non_expiratory_phase_val_idx]
        )

        elapsed = time.time() - start_time

        _logger.info(
            f"Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s"
        )
        _logger.info(
            f"Epoch {epoch+1} - MAE Score (without expiratory phase): {score:.4f}"
        )
        wandb.log(
            {
                f"[fold{fold}] epoch": epoch + 1,
                f"[fold{fold}] avg_train_loss": avg_loss,
                f"[fold{fold}] avg_val_loss": avg_val_loss,
                f"[fold{fold}] score": score,
            }
        )

        if score < best_score:
            best_score = score
            _logger.info(f"Epoch {epoch+1} - Save Best Score: {score:.4f} Model")
            torch.save(
                {"model": model.state_dict(), "preds": preds},
                OUTPUT_DIR + f"fold{fold}_best.pth",
            )

    preds = torch.load(
        OUTPUT_DIR + f"fold{fold}_best.pth", map_location=torch.device("cpu")
    )["preds"]
    valid_folds["preds"] = preds

    torch.cuda.empty_cache()
    gc.collect()

    return valid_folds


# =================================================
# Runner
# =================================================
def main():
    # logger
    init_root_logger(pathlib.Path(CFG.log_dir))
    _logger.setLevel(logging.INFO)

    # wandb
    wandb.login(key=WANDB_API)
    run = wandb.init(
        project=CFG.competition,
        name=CFG.model_name,
        config=class2dict(CFG),
        group=CFG.model_name,
        job_type="train",
    )

    # pytorch setting
    pl.seed_everything(CFG.seed)

    # data
    train, test, sub = load_data()
    train = add_feature(train)
    test = add_feature(test)

    # cv split
    Fold = GroupKFold(n_splits=5)
    groups = train["breath_id"].values
    for n, (train_index, val_index) in enumerate(
        Fold.split(train, train["pressure"], groups)
    ):
        train.loc[val_index, "fold"] = int(n)
    train["fold"] = train["fold"].astype(int)
    print(train.groupby("fold").size())

    def get_result(result_df):
        preds = result_df["preds"].values
        labels = result_df["pressure"].values
        non_expiratory_phase_val_idx = result_df[
            result_df["u_out"] == 0
        ].index  # The expiratory phase is not scored
        score = get_score(
            labels[non_expiratory_phase_val_idx], preds[non_expiratory_phase_val_idx]
        )
        _logger.info(f"Score (without expiratory phase): {score:<.4f}")

    if CFG.train:
        # train
        oof_df = pd.DataFrame()
        for fold in range(CFG.n_fold):
            if fold in CFG.trn_fold:
                _oof_df = train_loop(train, fold)
                oof_df = pd.concat([oof_df, _oof_df])
                _logger.info(f"========== fold: {fold} result ==========")
                get_result(_oof_df)
        # CV result
        _logger.info("========== CV ==========")
        get_result(oof_df)
        # save result
        oof_df.to_csv(OUTPUT_DIR + "oof_df.csv", index=False)

    if CFG.inference:
        test_dataset = TestDataset(test)
        test_loader = DataLoader(
            test_dataset,
            batch_size=CFG.batch_size * 2,
            shuffle=False,
            num_workers=CFG.num_workers,
            pin_memory=True,
        )
        for fold in CFG.trn_fold:
            model = CustomModel(CFG)
            path = OUTPUT_DIR + f"fold{fold}_best.pth"
            state = torch.load(path, map_location=torch.device("cpu"))
            model.load_state_dict(state["model"])
            predictions = inference_fn(test_loader, model, device)
            test[f"fold{fold}"] = predictions
            del state, predictions
            gc.collect()
            torch.cuda.empty_cache()
        # submission
        test["pressure"] = test[[f"fold{fold}" for fold in range(CFG.n_fold)]].mean(1)
        test[["id", "pressure"] + [f"fold{fold}" for fold in range(CFG.n_fold)]].to_csv(
            OUTPUT_DIR + "raw_submission.csv", index=False
        )
        test[["id", "pressure"]].to_csv(OUTPUT_DIR + "submission.csv", index=False)

    wandb.finish()


if __name__ == "__main__":
    main()
