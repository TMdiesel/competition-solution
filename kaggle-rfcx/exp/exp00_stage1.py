# default package
import logging
import pathlib
import sys
import os
import typing as t

# third party package
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import cv2
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from sklearn.model_selection import StratifiedKFold

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
    width = 400
    period = 10
    shift_time = 10
    strong_label_prob = 1.0
    params_melspec = {
        "n_fft": 2048,
        "n_mels": 128,
        "fmin": 80,
        "fmax": 15000,
        "power": 2.0,
    }

    # datamodule
    batch_size: int = 8
    num_workers: int = 2


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
def load_train_metadata(config: CFG) -> t.Tuple[pd.DataFrame, pathlib.Path]:
    train_audio_path = INPUT_DIR / "train"

    train_tp = pd.read_csv(INPUT_DIR / "train_tp.csv").reset_index(drop=True)
    train_fp = pd.read_csv(INPUT_DIR / "train_fp.csv").reset_index(drop=True)
    train_tp["data_type"] = "tp"
    train_fp["data_type"] = "fp"
    train = pd.concat([train_tp, train_fp])

    df_meta = train[train["data_type"].isin(config.use_train_data)].reset_index(
        drop=True
    )

    return df_meta, train_audio_path


def load_test_metadata(config: dict) -> t.Tuple[pd.DataFrame, pathlib.Path]:
    df_sub = pd.read_csv(INPUT_DIR / "sample_submission.csv").reset_index(drop=True)
    test_audio_path = INPUT_DIR / "test"

    return df_sub, test_audio_path


# =================================================
# Data process
# =================================================

# =================================================
# Torch dataset
# =================================================
class SpectrogramDataset(data.Dataset):
    TOTAL_TIME = 60

    def __init__(
        self,
        df: pd.DataFrame,
        audio_dir: pathlib.Path,
        phase: str,
        period: int,
        height: int,
        width: int,
        shift_time: float,
        params_melspec: dict,
    ):
        assert phase in ["train", "val", "test"], "phase must be train, val, or test"

        self.df = df
        self.audio_dir = audio_dir
        self.phase = phase
        self.period = period
        self.height = height
        self.width = width
        self.train_pseudo = None
        self.shift_time = shift_time
        self.params_melspec = params_melspec

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """get item.

        Parameters
        ----------
        idx : int

        Returns
        -------
        image: np.ndarray
        label: int

        Notes
        ------
        - audioを60sに調整
        """
        sample = self.df.iloc[idx]
        audio_path = self.audio_dir / f"{sample['recording_id']}.flac"
        y, sr = sf.read(audio_path)

        effective_length = sr * self.period
        y = adjust_audio_length(y, sr, self.TOTAL_TIME)

        if self.phase == "train":
            y, labels = strong_clip_audio(
                self.df, y, sr, idx, effective_length, self.train_pseudo
            )
            image = wave2image_normal(
                y, sr, self.width, self.height, self.params_melspec
            )
            return image, labels
        else:
            # PERIOD単位に6分割
            split_y = split_audio(y, self.TOTAL_TIME, self.period, self.shift_time, sr)

            images = []
            # 分割した音声を一つずつ画像化してリストで返す
            for y in split_y:
                image = wave2image_normal(
                    y, sr, self.width, self.height, self.params_melspec
                )
                images.append(image)

            if self.phase == "val":
                recording_id = sample["recording_id"]
                query_string = f"recording_id == '{recording_id}'"
                all_tp_events = self.df.query(query_string)
                labels = np.zeros(len(self.df["species_id"].unique()), dtype=np.float32)
                for species_id in all_tp_events["species_id"].unique():
                    labels[int(species_id)] = 1.0
                return np.asarray(images), labels

            elif self.phase == "test":
                labels = -1
                return np.asarray(images), labels


def split_audio(y, total_time, period, shift_time, sr):
    # PERIOD単位に分割
    num_data = int(total_time / shift_time)
    shift_length = sr * shift_time
    effective_length = sr * period
    split_y = []
    for i in range(num_data):
        start = shift_length * i
        finish = start + effective_length
        split_y.append(y[start:finish])

    return split_y


def adjust_audio_length(y: np.ndarray, sr: int, total_time: int = 60) -> np.ndarray:
    """データの長さを全てtotal_time分にする."""
    try:
        assert len(y) == total_time * sr
    except Exception as e:
        _logger.info("Assert Error")
        len_y = len(y)
        total_length = total_time * sr
        if len_y < total_length:
            new_y = np.zeros(total_length, dtype=y.dtype)
            start = np.random.randint(total_length - len_y)
            new_y[start : start + len_y] = y
            y = new_y.astype(np.float32)
        elif len_y > total_length:
            start = np.random.randint(len_y - total_length)
            y = y[start : start + total_length].astype(np.float32)
        else:
            y = y.astype(np.float32)
    return y


def wave2image_normal(
    y: np.ndarray, sr: int, width: int, height: int, melspectrogram_parameters: dict
) -> np.ndarray:
    """通常のmelspectrogram変換."""
    melspec = librosa.feature.melspectrogram(y, sr=sr, **melspectrogram_parameters)
    melspec = librosa.power_to_db(melspec).astype(np.float32)

    image = mono_to_color(melspec)
    image = cv2.resize(image, (width, height))
    image = np.moveaxis(image, 2, 0)
    image = (image / 255.0).astype(np.float32)
    return image


def mono_to_color(
    X: np.ndarray, mean=None, std=None, norm_max=None, norm_min=None, eps=1e-6
) -> np.ndarray:
    X = np.stack([X, X, X], axis=-1)

    # Standardize
    mean = mean or X.mean()
    X = X - mean
    std = std or X.std()
    Xstd = X / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    norm_max = norm_max or _max
    norm_min = norm_min or _min
    if (_max - _min) > eps:
        # Normalize to [0, 255]
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        # Just zero
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return V


def strong_clip_audio(
    df: pd.DataFrame, y: np.ndarray, sr: int, idx: int, effective_length: int, pseudo_df
):

    t_min = df["t_min"].values[idx] * sr
    t_max = df["t_max"].values[idx] * sr
    t_center = np.round((t_min + t_max) / 2)

    beginning = t_center - effective_length / 2
    if beginning < 0:
        beginning = 0
    beginning = np.random.randint(beginning, t_center)

    ending = beginning + effective_length
    if ending > len(y):
        ending = len(y)
    beginning = ending - effective_length

    y = y[beginning:ending].astype(np.float32)

    # flame→time変換
    beginning_time = beginning / sr
    ending_time = ending / sr

    # dfには同じrecording_idだけどclipしたt内に別のラベルがあるものもある
    # そこでそれには正しいidを付けたい
    recording_id = df.loc[idx, "recording_id"]
    try:  # tp data
        main_species_id = df.loc[idx, "species_id"]
    except:  # fp data
        main_species_id = None

    query_string = f"recording_id == '{recording_id}' & "
    query_string += f"t_min < {ending_time} & t_max > {beginning_time}"

    # 同じrecording_idのものを
    all_tp_events = df.query(query_string)

    labels = np.zeros(len(df["species_id"].unique()), dtype=np.float32)
    for species_id in all_tp_events["species_id"].unique():
        if species_id == main_species_id:
            labels[int(species_id)] = 1.0  # main label
        else:
            labels[int(species_id)] = 1.0  # secondary label

    return y, labels


# =================================================
# Torch datamodule
# =================================================
class SpectrogramDataModule(pl.LightningDataModule):
    def __init__(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        df_sub: pd.DataFrame,
        train_audio_dir: pathlib.Path,
        test_audio_dir: pathlib.Path,
    ):
        self.df_train = df_train
        self.df_val = df_val
        self.df_sub = df_sub
        self.train_audio_dir = train_audio_dir
        self.test_audio_dir = test_audio_dir

    def setup(self, stage: t.Optional[str] = None):
        assert stage in ["fit", "test", None], "stage must be fit or test"

        if stage == "fit" or stage is None:
            self.dataset_train = SpectrogramDataset(
                self.df_train,
                self.train_audio_dir,
                "train",
                CFG.period,
                CFG.height,
                CFG.width,
                CFG.shift_time,
                CFG.params_melspec,
            )
            self.dataset_val = SpectrogramDataset(
                self.df_val,
                self.train_audio_dir,
                "val",
                CFG.period,
                CFG.height,
                CFG.width,
                CFG.shift_time,
                CFG.params_melspec,
            )
        if stage == "test" or stage is None:
            self.dataset_test = SpectrogramDataset(
                self.df_sub,
                self.test_audio_dir,
                "test",
                CFG.period,
                CFG.height,
                CFG.width,
                CFG.shift_time,
                CFG.params_melspec,
            )

    def train_dataloader(self):
        return data.DataLoader(
            self.dataset_train,
            shuffle=True,
            batch_size=CFG.batch_size,
            num_workers=CFG.num_workers,
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.dataset_val,
            shuffle=False,
            batch_size=CFG.batch_size,
            num_workers=CFG.num_workers,
        )

    def test_dataloader(self):
        return data.DataLoader(
            self.dataset_test,
            shuffle=False,
            batch_size=CFG.batch_size,
            num_workers=CFG.num_workers,
        )


# =================================================
# Network
# =================================================
# =================================================
# Trainer
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
    df_meta, train_audio_dir = load_train_metadata(CFG)
    df_sub, test_audio_dir = load_test_metadata(CFG)

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

        # train

        # inference


if __name__ == "__main__":
    main()
