# default package
import logging
import pathlib
import sys
import os
import typing as t
import random

# third party package
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from PIL import Image


if "ipykernel" in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

from dotenv import load_dotenv

# my package

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
OUTPUT_DIR = pathlib.Path(load_env("OUTPUT_DIR"))
INPUT_DIR = pathlib.Path(load_env("INPUT_DIR"))


# =================================================
# Config
# =================================================
class CFG:
    debug = False
    log_dir = "./logs"

    # pytorch
    seed = 46

    # data
    use_train_data = ["tp"]

    # split
    n_split = 5
    random_state = 42
    shuffle = True
    folds = [1]

    # dataset
    height = 224
    width = 224

    # datamodule
    batch_size: int = 8
    num_workers: int = 2

    # trainer
    min_epochs: int = 1
    max_epochs: int = 100
    fast_dev_run: bool = False
    gpus = [0]

    # model
    learning_rate: float = 1e-2
    model_params: dict = {
        "base_model_name": "efficientnet-b2",
        "pretrained": True,
        "num_classes": 24,
    }


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


# =================================================
# Load data
# =================================================
def load_metadata(mode: str) -> t.Tuple[pd.DataFrame, pathlib.Path]:
    assert mode in ["train", "val", "test"], "mode must be train, val or test"
    if mode == "train":
        df_meta = pd.read_csv(INPUT_DIR / "train.csv")
    elif mode == "test":
        df_meta = pd.read_csv(INPUT_DIR / "test.csv")

    dir_photo = INPUT_DIR / "photos"
    return df_meta, dir_photo


# =================================================
# Torch dataset
# =================================================
class Dataset(data.Dataset):
    def __init__(
        self, df_meta: pd.DataFrame, dir_photo: pathlib.Path, mode: str, conf: CFG
    ):
        assert mode in ["train", "val", "test"], "mode must be train, val or test"
        self.df_meta = df_meta
        self.dir_photo = dir_photo
        self.mode = mode
        self.conf = conf

        size = (self.conf.height, self.conf.width)
        additional_items = (
            [transforms.Resize(size)]
            if mode != "train"
            else [
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.3,
                    contrast=0.5,
                    saturation=[0.8, 1.3],
                    hue=[-0.05, 0.05],
                ),
                transforms.RandomResizedCrop(size),
            ]
        )
        self.transform = transforms.Compose(
            [
                *additional_items,
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return len(self.df_meta)

    def __getitem__(self, index):
        row = self.df_meta.iloc[index]

        photo_path = self.dir_photo / (row["object_id"] + ".jpg")
        img = Image.open(photo_path)
        img = self.transform(img)

        label = row["target"]
        return img, label


# =================================================
# Torch datamodule
# =================================================
class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        df_meta_train: pd.DataFrame,
        df_meta_val: pd.DataFrame,
        df_meta_sub: pd.DataFrame,
        dir_photo: pathlib.Path,
        conf: CFG,
    ):
        self.df_meta_train = df_meta_train
        self.df_meta_val = df_meta_val
        self.df_meta_sub = df_meta_sub
        self.dir_photo = dir_photo
        self.conf = conf

    def setup(self, stage: t.Optional[str] = None):
        assert stage in ["fit", "test", None], "stage must be fit or test"

        if stage == "fit" or stage is None:
            self.dataset_train = Dataset(
                df_meta=self.df_meta_train,
                dir_photo=self.dir_photo,
                mode="train",
                conf=self.conf,
            )
            self.dataset_val = Dataset(
                df_meta=self.df_meta_train,
                dir_photo=self.dir_photo,
                mode="val",
                conf=self.conf,
            )

        if stage == "test" or stage is None:
            self.dataset_val = Dataset(
                df_meta=self.df_meta_train,
                dir_photo=self.dir_photo,
                mode="test",
                conf=self.conf,
            )

    def train_dataloader(self):
        return data.DataLoader(
            self.dataset_train,
            shuffle=True,
            batch_size=self.conf.batch_size,
            num_workers=self.conf.num_workers,
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.dataset_val,
            shuffle=False,
            batch_size=self.conf.batch_size,
            num_workers=self.conf.num_workers,
        )

    def test_dataloader(self):
        return data.DataLoader(
            self.dataset_test,
            shuffle=False,
            batch_size=self.conf.batch_size,
            num_workers=self.conf.num_workers,
        )


# =================================================
# Network
# =================================================


# =================================================
# Trainer
# =================================================
class Trainer(pl.LightningModule):
    def __init__(self, model: nn.Module, config: CFG):
        super().__init__()
        self.config = config
        self.model = model
        self.output_key = config.output_key
        self.criterion = LSEPStableLoss(output_key=self.output_key)
        self.f1 = F1(num_classes=24)

    def training_step(self, batch, batch_idx):
        x, y = batch
        p = random.random()
        do_mixup = True if p < self.config["mixup"]["prob"] else False

        if self.config["mixup"]["flag"] and do_mixup:
            x, y, y_shuffle, lam = mixup_data(x, y, alpha=self.config["mixup"]["alpha"])

        output = self.model(x)
        pred = output[self.output_key]
        if "framewise" in self.output_key:
            pred, _ = pred.max(dim=1)

        if self.config["mixup"]["flag"] and do_mixup:
            loss = mixup_criterion(
                self.criterion, output, y, y_shuffle, lam, phase="train"
            )
        else:
            loss = self.criterion(output, y, phase="train")

        lwlrap = LWLRAP(pred, y)
        f1_score = self.f1(pred, y)

        self.log(
            "loss/train",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "LWLRAP/train",
            lwlrap,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "F1/train",
            f1_score,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Notes
        -----
        - batchのxはlist型
        - xが複数の場合
        """
        x_list, y = batch
        x = x_list.view(
            -1, x_list.shape[2], x_list.shape[3], x_list.shape[4]
        )  # batch>1でも可

        output = self.model(x)
        loss = self.criterion(output, y, phase="valid")
        pred = output[self.output_key]
        if "framewise" in self.output_key:
            pred, _ = pred.max(dim=1)
        pred = split2one(pred, y)
        lwlrap = LWLRAP(pred, y)
        f1_score = self.f1(pred, y)
        self.log(
            "loss/val", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True
        )
        self.log(
            "LWLRAP/val",
            lwlrap,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "F1/val",
            f1_score,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)


# =================================================
# Metrics
# =================================================


# =================================================
# Main
# =================================================
def main() -> None:
    # setting
    init_root_logger(pathlib.Path(CFG.log_dir))
    _logger.setLevel(logging.INFO)

    _logger.info(ENV)
    pl.seed_everything(CFG.seed)

    # load data
    df_meta_train, dir_photo = load_metadata(mode="train")
    df_meta_test, dir_photo = load_metadata(mode="test")

    # cv
    skf = StratifiedKFold(CFG.n_split)
    for fold, (idx_train, idx_val) in enumerate(
        skf.split(df_meta, df_meta["species_id"])
    ):
        if fold not in CFG.folds:
            continue

        # data
        df_train = df_meta.loc[idx_train].reset_index(drop=True)
        df_val = df_meta.loc[idx_val].reset_index(drop=True)
        dm = SpectrogramDataModule(
            df_train=df_train,
            df_val=df_val,
            df_sub=df_sub,
            train_audio_dir=train_audio_dir,
            test_audio_dir=test_audio_dir,
        )

        # train
        callbacks = []

        trainer = pl.Trainer(
            logger=None,
            callbacks=callbacks,
            min_epochs=CFG.min_epochs,
            max_epochs=CFG.max_epochs,
            gpus=CFG.gpus,
            fast_dev_run=CFG.fast_dev_run,
            deterministic=False,
            precision=16,
        )

        network = EfficientNetSED(CFG)
        model = Trainer(network, CFG)
        trainer.fit(model, dm)

        # inference


if __name__ == "__main__":
    main()
