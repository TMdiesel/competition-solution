# Python Libraries
import os
import sys
import math
import random
import glob
import pickle
from collections import defaultdict
from pathlib import Path
import logging
import pathlib

# Third party
import numpy as np
import pandas as pd

if "ipykernel" in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import wandb
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
from sklearn.preprocessing import RobustScaler, normalize, QuantileTransformer
from sklearn.metrics import mean_absolute_error  # [roc_auc_score, accuracy_score]

# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    CosineAnnealingLR,
    ReduceLROnPlateau,
)
from torch.optim.optimizer import Optimizer, required
import torch_optimizer as optim
import pytorch_lightning as pl
from pytorch_lightning import Callback, seed_everything
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
)
from pytorch_lightning.loggers import WandbLogger, CSVLogger


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
_logger.info(ENV)

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
    debug = True
    competition = "ventilator"
    exp_name = "public"
    seed = 29
    log_dir = "./logs"

    # data
    target_col = "pressure"
    target_size = 1

    # optimizer
    optimizer_name = "RAdam"  # ['RAdam', 'sgd']
    lr = 5e-3
    weight_decay = 1e-5
    amsgrad = False

    # scheduler
    epochs = 2 if debug else 300
    scheduler = "CosineAnnealingLR"
    T_max = 300
    min_lr = 1e-5

    # criterion
    # u_out = 1 を考慮しないLoss
    criterion_name = "CustomLoss1"

    # training
    train = True
    inference = True
    n_fold = 5
    trn_fold = [0]
    precision = 16  # [16, 32, 64]
    grad_acc = 1
    # DataLoader
    loader = {
        "train": {
            "batch_size": 1024,
            "num_workers": 0,
            "shuffle": True,
            "pin_memory": True,
            "drop_last": True,
        },
        "valid": {
            "batch_size": 1024,
            "num_workers": 0,
            "shuffle": False,
            "pin_memory": True,
            "drop_last": False,
        },
    }
    # pl
    trainer = {
        "gpus": 1,
        "progress_bar_refresh_rate": 1,
        "benchmark": False,
        "deterministic": True,
    }
    # LSTM
    num_layers = 4

    cate_cols = ["R", "C"]
    cont_cols = ["time_step", "u_in", "u_out"] + ["breath_time", "u_in_time"]

    feature_cols = cate_cols + cont_cols
    dense_dim = 512
    hidden_size = 512
    logit_dim = 512


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


# =================================================
# Load data
# =================================================
def load_data():
    df_train = pd.read_csv(f"{INPUT_DIR}train.csv")
    df_test = pd.read_csv(f"{INPUT_DIR}test.csv")
    df_sub = pd.read_csv(f"{INPUT_DIR}sample_submission.csv")

    for c in ["u_in"]:
        df_train[c] = np.log1p(df_train[c])
        df_test[c] = np.log1p(df_test[c])

    r_map = {5: 0, 20: 1, 50: 2}
    c_map = {10: 0, 20: 1, 50: 2}
    df_train["R"] = df_train["R"].map(r_map)
    df_test["R"] = df_test["R"].map(r_map)
    df_train["C"] = df_train["C"].map(c_map)
    df_test["C"] = df_test["C"].map(c_map)
    return df_train, df_test, df_sub


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


def add_fold(df_train):
    df_train["fold"] = -1
    Fold = GroupKFold(n_splits=CFG.n_fold)
    for n, (train_index, val_index) in enumerate(
        Fold.split(df_train, df_train[CFG.target_col], groups=df_train.breath_id.values)
    ):
        df_train.loc[val_index, "fold"] = int(n)
    df_train["fold"] = df_train["fold"].astype(int)
    df_oof = df_train[["id", "breath_id", "u_out", "pressure", "fold"]].copy()

    return df_train, df_oof


def quantile_transform(df):
    for col in tqdm(CFG.cont_cols):
        qt = QuantileTransformer(random_state=0, output_distribution="normal")
        df[[col]] = qt.fit_transform(df[[col]])
        df[[col]] = qt.transform(df[[col]])
    return df


# =================================================
# Dataset
# =================================================
class TrainDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.u_out = self.X[:, :, 2]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.X[idx])
        u_out = torch.LongTensor(self.u_out[idx])
        label = torch.FloatTensor(self.y[idx]).squeeze(1)
        return x, u_out, label


class TestDataset(Dataset):
    def __init__(self, X):
        self.X = X
        self.u_out = self.X[:, :, 2]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx])


class DataModule(pl.LightningDataModule):
    """
    numpy arrayで受け取る
    """

    def __init__(self, tr_X, tr_y, val_X, val_y, test_X, cfg):
        super().__init__()
        self.train_data = tr_X
        self.train_label = tr_y
        self.valid_data = val_X
        self.valid_label = val_y
        self.test_data = test_X
        self.cfg = cfg

    def setup(self, stage=None):
        self.train_dataset = TrainDataset(self.train_data, self.train_label)
        self.valid_dataset = TrainDataset(self.valid_data, self.valid_label)
        self.test_dataset = TestDataset(self.test_data)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.cfg.loader["train"])

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, **self.cfg.loader["valid"])

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self.cfg.loader["valid"])


# =================================================
# Model
# =================================================
class CustomModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.dense_dim = cfg.dense_dim
        self.hidden_size = cfg.hidden_size
        self.num_layers = cfg.num_layers
        self.logit_dim = cfg.logit_dim
        self.mlp = nn.Sequential(
            nn.Linear(len(cfg.feature_cols), self.dense_dim // 2),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(self.dense_dim // 2, self.dense_dim),
            nn.ReLU(),
        )
        self.lstm1 = nn.LSTM(
            self.dense_dim,
            self.dense_dim,
            dropout=0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm2 = nn.LSTM(
            self.dense_dim * 2,
            self.dense_dim // 2,
            dropout=0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm3 = nn.LSTM(
            self.dense_dim // 2 * 2,
            self.dense_dim // 4,
            dropout=0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm4 = nn.LSTM(
            self.dense_dim // 4 * 2,
            self.dense_dim // 8,
            dropout=0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(self.hidden_size // 8 * 2),
            nn.GELU(),
            nn.Linear(self.hidden_size // 8 * 2, 1),
        )
        # LSTMやGRUは直交行列に初期化する
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
                        nn.init.orthogonal_(param.data)
                    else:
                        nn.init.normal_(param.data)

    def forward(self, x):
        bs = x.size(0)
        features = self.mlp(x)
        features, _ = self.lstm1(features)
        features, _ = self.lstm2(features)
        features, _ = self.lstm3(features)
        features, _ = self.lstm4(features)
        output = self.head(features).view(bs, -1)
        return output


def get_model(cfg):
    model = CustomModel(cfg)
    return model


# ====================================================
# criterion
# ====================================================
def compute_metric(df, preds):
    """
    Metric for the problem, as I understood it.
    """

    y = np.array(df["pressure"].values.tolist())
    w = 1 - np.array(df["u_out"].values.tolist())

    assert y.shape == preds.shape and w.shape == y.shape, (
        y.shape,
        preds.shape,
        w.shape,
    )

    mae = w * np.abs(y - preds)
    mae = mae.sum() / w.sum()

    return mae


class VentilatorLoss(nn.Module):
    """
    Directly optimizes the competition metric
    """

    def __call__(self, preds, y, u_out):
        w = 1 - u_out
        mae = w * (y - preds).abs()
        mae = mae.sum(-1) / w.sum(-1)

        return mae


def get_criterion():
    if CFG.criterion_name == "BCEWithLogitsLoss":
        criterion = nn.BCEWithLogitsLoss(reduction="mean")
    if CFG.criterion_name == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss()
    if CFG.criterion_name == "CustomLoss1":
        # [reference]https://www.kaggle.com/theoviel/deep-learning-starter-simple-lstm
        criterion = VentilatorLoss()
    else:
        raise NotImplementedError
    return criterion


# ====================================================
# optimizer
# ====================================================
def get_optimizer(model: nn.Module):
    """
    input:
    model:model
    config:optimizer_nameやlrが入ったものを渡す

    output:optimizer
    """
    optimizer_name = CFG.optimizer_name
    if "Adam" == optimizer_name:
        return Adam(
            model.parameters(),
            lr=CFG.lr,
            weight_decay=CFG.weight_decay,
            amsgrad=CFG.amsgrad,
        )
    elif "RAdam" == optimizer_name:
        return optim.RAdam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    elif "sgd" == optimizer_name:
        return SGD(
            model.parameters(),
            lr=CFG.lr,
            momentum=0.9,
            nesterov=True,
            weight_decay=CFG.weight_decay,
        )
    else:
        raise NotImplementedError


# ====================================================
# scheduler
# ====================================================
def get_scheduler(optimizer):
    if CFG.scheduler == "ReduceLROnPlateau":
        """
        factor : 学習率の減衰率
        patience : 何ステップ向上しなければ減衰するかの値
        eps : nanとかInf回避用の微小数
        """
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=CFG.factor,
            patience=CFG.patience,
            verbose=True,
            eps=CFG.eps,
        )
    elif CFG.scheduler == "CosineAnnealingLR":
        """
        T_max : 1 半周期のステップサイズ
        eta_min : 最小学習率(極小値)
        """
        scheduler = CosineAnnealingLR(
            optimizer, T_max=CFG.T_max, eta_min=CFG.min_lr, last_epoch=-1
        )
    elif CFG.scheduler == "CosineAnnealingWarmRestarts":
        """
        T_0 : 初期の繰りかえし回数
        T_mult : サイクルのスケール倍率
        """
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=CFG.T_0, T_mult=1, eta_min=CFG.min_lr, last_epoch=-1
        )
    else:
        raise NotImplementedError
    return scheduler


# =================================================
# Train
# =================================================
class Trainer(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = get_model(cfg)
        self.criterion = get_criterion()

    def forward(self, x):
        output = self.model(x)
        return output

    def training_step(self, batch, batch_idx):
        x, u_out, y = batch
        output = self.forward(x)
        labels = y
        loss = self.criterion(output, labels, u_out).mean()

        self.log("train_loss", loss, on_step=True, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": output, "labels": labels}

    def training_epoch_end(self, outputs):
        self.log("lr", self.optimizer.param_groups[0]["lr"], prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        x, u_out, y = batch
        output = self.forward(x)
        labels = y
        loss = self.criterion(output, labels, u_out).mean()
        self.log("val_loss", loss, on_step=True, prog_bar=True, logger=True)
        return {"predictions": output, "labels": labels, "loss": loss.item()}

    def validation_epoch_end(self, outputs):
        preds = []
        labels = []
        loss = 0
        for output in outputs:
            preds += output["predictions"]
            labels += output["labels"]
            loss += output["loss"]

        labels = torch.stack(labels)
        preds = torch.stack(preds)
        loss = loss / len(outputs)

        self.log("val_loss_epoch", loss, prog_bar=True, logger=True)

    def predict_step(self, batch, batch_idx):
        x = batch
        output = self.forward(x)
        return output

    def test_step(self, batch, batch_idx):
        x = batch
        output = self.forward(x)
        return output

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self)
        self.scheduler = {
            "scheduler": get_scheduler(self.optimizer),
            "interval": "step",  # or 'epoch'
            "frequency": 1,
        }
        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}


# =================================================
# Runner
# =================================================
def main():
    # logger
    init_root_logger(pathlib.Path(CFG.log_dir))
    _logger.setLevel(logging.INFO)

    # pytroch setting
    seed_everything(CFG.seed)
    if ENV != "LOCAL":
        CFG.loader["train"]["num_workers"] = 4
        CFG.loader["valid"]["num_workers"] = 4

    # data
    df_train, df_test, df_sub = load_data()
    df_train = add_feature(df_train)
    df_test = add_feature(df_test)
    df_train, df_oof = add_fold(df_train)
    df_train = quantile_transform(df_train)
    df_test = quantile_transform(df_test)

    X = np.float32(df_train[CFG.feature_cols]).reshape(-1, 80, len(CFG.feature_cols))
    test_X = np.float32(df_test[CFG.feature_cols]).reshape(
        -1, 80, len(CFG.feature_cols)
    )
    y = np.float32(df_train["pressure"]).reshape(-1, 80, 1)
    Fold = np.int16(df_train["fold"]).reshape(-1, 80, 1)
    Fold = Fold.mean(axis=1).flatten()
    print(X.shape, y.shape, test_X.shape, Fold.shape)

    # train
    for fold in range(CFG.n_fold):
        if fold not in CFG.trn_fold:
            continue
        print(f"{'='*38} Fold: {fold} {'='*38}")
        lr_monitor = LearningRateMonitor(logging_interval="step")
        loss_checkpoint = ModelCheckpoint(
            dirpath=OUTPUT_DIR,
            filename=f"best_loss_fold{fold}",
            monitor="val_loss",
            save_last=True,
            save_top_k=1,
            save_weights_only=True,
            mode="min",
        )

        # wandb
        wandb.login(key=WANDB_API)
        wandb_logger = WandbLogger(
            project=f"{CFG.competition}",
            group=f"{CFG.exp_name}",
            name=f"Fold{fold}",
            save_dir=OUTPUT_DIR,
            config=class2dict(CFG),
        )
        data_module = DataModule(
            X[Fold != fold],
            y[Fold != fold],
            X[Fold == fold],
            y[Fold == fold],
            test_X,
            CFG,
        )
        data_module.setup()

        CFG.T_max = int(
            math.ceil(len(data_module.train_dataloader()) / CFG.grad_acc) * CFG.epochs
        )
        print(f"set schedular T_max {CFG.T_max}")
        # early_stopping_callback = EarlyStopping(monitor='val_loss_epoch',mode="min", patience=5)

        trainer = pl.Trainer(
            logger=wandb_logger,
            callbacks=[loss_checkpoint],  # lr_monitor,early_stopping_callback
            default_root_dir=OUTPUT_DIR,
            accumulate_grad_batches=CFG.grad_acc,
            max_epochs=CFG.epochs,
            precision=CFG.precision,
            **CFG.trainer,
        )
        # 学習
        model = Trainer(CFG)
        trainer.fit(model, data_module)
        torch.save(
            model.model.state_dict(),
            OUTPUT_DIR + "/" + f"{CFG.exp_name}_fold{fold}.pth",
        )

        del model, data_module

        if CFG.inference:
            data_module = DataModule(X[0:1], y[0:1], X[0:1], y[0:1], test_X, CFG)
            data_module.setup()
            # Load best loss model
            best_model = Trainer.load_from_checkpoint(
                cfg=CFG, checkpoint_path=loss_checkpoint.best_model_path
            )
            predictions = trainer.predict(best_model, data_module.test_dataloader())
            preds = []
            for p in predictions:
                preds += p
            preds = torch.stack(preds).flatten()

            submission = pd.DataFrame()
            submission["pressure"] = preds.to("cpu").detach().numpy()
            submission.to_csv(
                OUTPUT_DIR + "/" + f"submission_fold{fold}.csv", index=False
            )

            # oof
            data_module = DataModule(
                X[0:1], y[0:1], X[0:1], y[0:1], X[Fold == fold], CFG
            )
            data_module.setup()
            predictions = trainer.predict(best_model, data_module.test_dataloader())
            preds = []
            for p in predictions:
                preds += p
            preds = torch.stack(preds).flatten()
            df_oof.loc[df_oof["fold"] == fold, ["pred"]] = (
                preds.to("cpu").detach().numpy()
            )
            df_oof.to_csv(OUTPUT_DIR + "/" + "oof.csv", index=False)

        wandb.finish()

        # skip in 21,22


if __name__ == "__main__":
    main()
