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
    def __init__(self, df_meta: pd.DataFrame, dir_photo: pathlib.Path, mode: str):
        assert mode in ["train", "val", "test"], "mode must be train, val or test"
        self.df_meta = df_meta
        self.dir_photo = dir_photo
        self.mode = mode

        size = (CFG.height, CFG.width)
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
