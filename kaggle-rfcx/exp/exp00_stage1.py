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
import soundfile as sf
import librosa
import cv2
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.model_selection import StratifiedKFold
from efficientnet_pytorch import EfficientNet
from torchmetrics import F1

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

    # trainer
    min_epochs: int = 1
    max_epochs: int = 100
    fast_dev_run: bool = False
    gpus = [0]

    # model
    output_key: str = "logit"
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
class EfficientNetSED(nn.Module):
    def __init__(self, config):
        super().__init__()

        model_params = config.model_params
        if model_params["pretrained"]:
            self.base_model = EfficientNet.from_pretrained(
                model_params["base_model_name"]
            )
        else:
            self.base_model = EfficientNet.from_name(model_params["base_model_name"])

        in_features = self.base_model._fc.in_features

        self.fc1 = nn.Linear(in_features, in_features, bias=True)
        self.att_block = AttBlockV2(
            in_features, model_params["num_classes"], activation="sigmoid"
        )

        self.init_weight()
        self.interpolate_ratio = 30  # Downsampled ratio

    def init_weight(self):
        init_layer(self.fc1)

    def forward(self, input):
        frames_num = input.size(3)

        # (batch_size, channels, freq, frames) ex->(120, 1408, 7, 12)
        x = self.base_model.extract_features(input)

        # (batch_size, channels, frames) ex->(120, 1408, 12)
        x = torch.mean(x, dim=2)

        # channel smoothing
        # channel次元上でpoolingを行う
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2

        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(
            1, 2
        )  # torch.Size([120, 1408, 12]) -> torch.Size([120, 12, 1408])
        x = F.relu_(self.fc1(x))
        x = x.transpose(
            1, 2
        )  # torch.Size([120, 12, 1408]) -> torch.Size([120, 1408, 12])
        x = F.dropout(x, p=0.5, training=self.training)
        (clipwise_output, norm_att, segmentwise_output) = self.att_block(x)
        logit = torch.sum(
            norm_att * self.att_block.cla(x), dim=2
        )  # claにsigmoidをかけない状態でclipwiseを計算
        segmentwise_logit = self.att_block.cla(x).transpose(
            1, 2
        )  # torch.Size([120, 12, 24])
        segmentwise_output = segmentwise_output.transpose(
            1, 2
        )  # torch.Size([120, 12, 24])

        # Get framewise output
        framewise_output = interpolate(
            segmentwise_output, self.interpolate_ratio
        )  # n_time次元上でをupsampling
        framewise_output = pad_framewise_output(
            framewise_output, frames_num
        )  # n_timesの最後の値で穴埋めしてframes_numに合わせる

        framewise_logit = interpolate(segmentwise_logit, self.interpolate_ratio)
        framewise_logit = pad_framewise_output(framewise_logit, frames_num)

        output_dict = {
            "clipwise_output": clipwise_output,
            "framewise_output": framewise_output,
            "segmentwise_output": segmentwise_output,
            "logit": logit,
            "framewise_logit": framewise_logit,
            "segmentwise_logit": segmentwise_logit,
        }

        return output_dict


def interpolate(x: torch.Tensor, ratio: int):
    """Interpolate data in time domain. This is used to compensate the
    resolution reduction in downsampling of a CNN.
    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate
    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


def pad_framewise_output(framewise_output: torch.Tensor, frames_num: int):
    """Pad framewise_output to the same length as input frames.
       The pad value is the same as the value of the last frame.
    Args:
      framewise_output: (batch_size, frames_num, classes_num)
      frames_num: int, number of frames to pad
    Outputs:
      output: (batch_size, frames_num, classes_num)
    Notes:
     n_timeの最後の値で穴埋めしてframe数になるようにする
    """
    pad = framewise_output[:, -1:, :].repeat(
        1, frames_num - framewise_output.shape[1], 1
    )
    """tensor for padding"""

    output = torch.cat((framewise_output, pad), dim=1)
    """(batch_size, frames_num, classes_num)"""

    return output


def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)


class AttBlockV2(nn.Module):
    def __init__(self, in_features: int, out_features: int, activation="linear"):
        super().__init__()

        self.activation = activation
        self.att = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.cla = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.init_weights()

    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)

    def forward(self, x):
        """
        Args:
        x: (n_samples, n_in, n_time)  ex)torch.Size([120, 1408, 12])
        Outputs:
        x:(batch_size, classes_num) ex)torch.Size([120, 24])
        norm_att: batch_size, classes_num, n_time) ex)torch.Size([120, 24, 12])
        cla: batch_size, classes_num, n_time) ex)torch.Size([120, 24, 12])
        """
        norm_att = torch.softmax(
            torch.tanh(self.att(x)), dim=-1
        )  # torch.Size([batch, n_class, 1]) クラス数に圧縮/valueを-1~1/n_timeの次元の総和=１に変換
        cla = self.nonlinear_transform(self.cla(x))  # self.cla()=self.att()/sigmoid変換
        x = torch.sum(
            norm_att * cla, dim=2
        )  # 要素同士の積 torch.Size([120, 24]): (batch, n_class)
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == "linear":
            return x
        elif self.activation == "sigmoid":
            return torch.sigmoid(x)


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


def LWLRAP(preds, labels):
    preds = preds.to("cpu")
    labels = labels.to("cpu")

    labels[labels > 0.0] = 1.0  # label smoothingする場合もスコア計算のため1にしてしまう
    # Ranks of the predictions
    ranked_classes = torch.argsort(preds, dim=-1, descending=True)
    # i, j corresponds to rank of prediction in row i
    class_ranks = torch.zeros_like(ranked_classes)
    for i in range(ranked_classes.size(0)):
        for j in range(ranked_classes.size(1)):
            class_ranks[i, ranked_classes[i][j]] = j + 1
    # Mask out to only use the ranks of relevant GT labels
    ground_truth_ranks = class_ranks * labels + (1e6) * (1 - labels)
    # All the GT ranks are in front now
    sorted_ground_truth_ranks, _ = torch.sort(
        ground_truth_ranks, dim=-1, descending=False
    )
    # Number of GT labels per instance
    # num_labels = labels.sum(-1)
    pos_matrix = torch.tensor(
        np.array([i + 1 for i in range(labels.size(-1))])
    ).unsqueeze(0)
    score_matrix = pos_matrix / sorted_ground_truth_ranks
    score_mask_matrix, _ = torch.sort(labels, dim=-1, descending=True)
    scores = score_matrix * score_mask_matrix
    score = scores.sum() / labels.sum()
    return score.item()


def mixup_criterion(criterion, pred, y_a, y_b, lam, phase="train"):
    return lam * criterion(pred, y_a, phase) + (1 - lam) * criterion(pred, y_b, phase)


def split2one(input, target):
    """
    validは60sの音声をn分割されバッチとしてモデルに入力される
    そこで出力後に分割されたデータを１つのデータに変換する必要がある
    クラスごとのmaxを出力とする
    """
    input_ = input.view(target.shape[0], -1, target.shape[1])  # y.shape[1]==num_classes
    input_ = torch.max(input_, dim=1)[0]  # 1次元目(分割sしたやつ)で各クラスの最大を取得
    return input_


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


# =================================================
# Metrics
# =================================================
class LSEPStableLoss(nn.Module):
    def __init__(self, output_key="logit", average=True):
        super(LSEPStableLoss, self).__init__()
        self.average = average
        self.output_key = output_key

    def forward(self, inputs, target, phase="train"):
        input = inputs[self.output_key]
        if "framewise" in self.output_key:
            input, _ = input.max(dim=1)
        target = target.float()
        # validの場合view, maxで分割したデータを１つのデータとして集約する必要がある
        if phase == "valid":
            input = split2one(input, target)

        n = input.size(0)
        differences = input.unsqueeze(1) - input.unsqueeze(2)
        where_lower = (target.unsqueeze(1) < target.unsqueeze(2)).float()

        differences = differences.view(n, -1)
        where_lower = where_lower.view(n, -1)

        max_difference, index = torch.max(differences, dim=1, keepdim=True)
        differences = differences - max_difference
        exps = differences.exp() * where_lower

        lsep = max_difference + torch.log(torch.exp(-max_difference) + exps.sum(-1))

        if self.average:
            return lsep.mean()
        else:
            return lsep


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
