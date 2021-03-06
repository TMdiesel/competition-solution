# default package
import logging
import pathlib
import sys
import os
import typing as t
import smtplib
from email.mime.text import MIMEText
from email.utils import formatdate

# third party package
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
import pytorch_lightning as pl
import wandb
import timm
from sklearn.model_selection import StratifiedKFold
from PIL import Image
from dotenv import load_dotenv
from torchvision import transforms
from torchvision.models import resnet18
from vivid.utils import timer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

if "ipykernel" in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


# global variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_logger = logging.getLogger(__name__)
if "KAGGLE_URL_BASE" in set(os.environ.keys()):
    from kaggle_secrets import UserSecretsClient

    ENV = "KAGGLE"
    USER_SECRETS = UserSecretsClient()
elif "google.colab" in sys.modules:
    path = (pathlib.Path(os.path.dirname(__file__))).joinpath("../.env")
    load_dotenv(path)
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
FROM_ADDRESS = load_env("FROM_ADDRESS")
TO_ADDRESS = load_env("TO_ADDRESS")
PASSWORD = load_env("PASSWORD")


# =================================================
# Config
# =================================================
class CFG:
    debug = True
    log_dir = "./logs"

    # experiment
    use_wandb = True
    project = "atma11"
    exp_name = "exp00.02"
    use_mail = True

    # pytorch
    seed = 46

    # data
    use_train_data = ["tp"]

    # split
    n_split = 5
    random_state = 42
    shuffle = True
    folds = [0] if debug else [0, 1, 2, 3, 4]

    # dataset
    height = 224
    width = 224

    # datamodule
    batch_size: int = 8
    num_workers: int = 8

    # trainer
    min_epochs: int = 2 if debug else 50
    max_epochs: int = 2 if debug else 150
    fast_dev_run: bool = False
    gpus = [0]

    # model
    learning_rate: float = 1e-3


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


class Sender:
    def __init__(
        self,
        from_address: str,
        to_address: str,
        password: str,
    ):
        self.from_address = from_address
        self.to_address = to_address
        self.password = password

    def create_message(self, subject: str, body: str):
        self.msg = MIMEText(body)
        self.msg["Subject"] = subject
        self.msg["From"] = self.from_address
        self.msg["To"] = self.to_address
        self.msg["Date"] = formatdate()

    def send(self):
        smtpobj = smtplib.SMTP("smtp.gmail.com", 587)
        smtpobj.ehlo()
        smtpobj.starttls()
        smtpobj.ehlo()
        smtpobj.login(self.from_address, self.password)
        smtpobj.sendmail(self.from_address, self.to_address, self.msg.as_string())
        smtpobj.close()


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
        if self.mode == "test":
            label = -1
        else:
            label = row["target"]
        return img, np.float32(label)


# =================================================
# Torch datamodule
# =================================================
class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        df_meta_train: pd.DataFrame,
        df_meta_val: pd.DataFrame,
        df_meta_test: pd.DataFrame,
        dir_photo: pathlib.Path,
        conf: CFG,
    ):
        self.df_meta_train = df_meta_train
        self.df_meta_val = df_meta_val
        self.df_meta_test = df_meta_test
        self.dir_photo = dir_photo
        self.conf = conf

    def setup(self, stage: t.Optional[str] = None):

        self.dataset_train = Dataset(
            df_meta=self.df_meta_train,
            dir_photo=self.dir_photo,
            mode="train",
            conf=self.conf,
        )
        self.dataset_val = Dataset(
            df_meta=self.df_meta_val,
            dir_photo=self.dir_photo,
            mode="val",
            conf=self.conf,
        )
        self.dataset_test = Dataset(
            df_meta=self.df_meta_test,
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
def create_network():
    network = timm.create_model("resnet18d", pretrained=False)
    network.fc = nn.Linear(in_features=512, out_features=1, bias=True)
    return network


# =================================================
# Trainer
# =================================================
class Trainer(pl.LightningModule):
    def __init__(self, network: nn.Module, conf: CFG):
        super().__init__()
        self.save_hyperparameters()
        self.conf = conf
        self.network = network
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        output = output.view(y.shape)
        loss = self.criterion(output, y)

        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        output = output.view(y.shape)
        loss = self.criterion(output, y)
        loss = loss.view(-1)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch
        return self.forward(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.conf.learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        return [optimizer], [scheduler]


# =================================================
# Main
# =================================================
def main() -> None:
    # setting
    init_root_logger(pathlib.Path(CFG.log_dir))
    log_level = logging.DEBUG if CFG.debug else logging.INFO
    _logger.setLevel(log_level)
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    _logger.info(ENV)
    pl.seed_everything(CFG.seed)

    # experiment
    pl_logger = None
    if CFG.use_wandb:
        wandb.login(key=WANDB_API)

    # load data
    df_meta_train, dir_photo = load_metadata(mode="train")
    df_meta_test, dir_photo = load_metadata(mode="test")

    # cv
    df_oof = df_meta_train.copy()
    df_sub = df_meta_test.copy()
    skf = StratifiedKFold(CFG.n_split, shuffle=True, random_state=CFG.seed)
    for fold, (idx_train, idx_val) in enumerate(
        skf.split(df_meta_train, df_meta_train["target"])
    ):
        if fold not in CFG.folds:
            continue

        # experiment
        if CFG.use_wandb:
            pl_logger = WandbLogger(
                project=f"{CFG.project}",
                group=f"{CFG.exp_name}",
                name=f"Fold{fold}",
                save_dir=str(OUTPUT_DIR),
                config=class2dict(CFG),
                log_model=True,
            )

        # data
        if CFG.debug:
            idx_train = idx_train[:20]
        df_train = df_meta_train.loc[idx_train].reset_index(drop=True)
        df_val = df_meta_train.loc[idx_val].reset_index(drop=True)
        dm = DataModule(
            df_meta_train=df_train,
            df_meta_val=df_val,
            df_meta_test=df_meta_test,
            dir_photo=dir_photo,
            conf=CFG,
        )

        # callbacks
        callbacks: t.List[t.Any] = []
        checkpoint = ModelCheckpoint(
            dirpath=OUTPUT_DIR,
            monitor="val_loss",
            save_last=False,
            save_top_k=1,
            save_weights_only=True,
            mode="min",
            every_n_epochs=1,
        )
        callbacks.append(checkpoint)
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=5,
                min_delta=0,
                mode="min",
            )
        )

        # train
        trainer = pl.Trainer(
            logger=pl_logger,
            callbacks=callbacks,
            min_epochs=CFG.min_epochs,
            max_epochs=CFG.max_epochs,
            gpus=CFG.gpus,
            fast_dev_run=CFG.fast_dev_run,
            deterministic=False,
            precision=32,
        )

        network = create_network()
        model = Trainer(network, CFG)
        trainer.fit(model, dm)

        # inference
        best_model = model.load_from_checkpoint(
            checkpoint_path=checkpoint.best_model_path, network=network, conf=CFG
        )
        # val
        pred_val = trainer.predict(best_model, dataloaders=dm.val_dataloader())
        pred_val = torch.vstack(pred_val)
        df_oof.loc[idx_val, "pred"] = pred_val.detach().numpy().reshape(-1)

        # test
        pred_test = trainer.predict(best_model, dataloaders=dm.test_dataloader())
        pred_test = torch.vstack(pred_test)
        df_sub.loc[:, "target"] = pred_test.detach().numpy().reshape(-1)

        # save
        df_sub.to_csv(OUTPUT_DIR / f"sub_{fold}.csv", index=False)
        if CFG.use_wandb:
            run = pl_logger.experiment
            artifact = wandb.Artifact(
                f"dataset-sub-{fold}-{pl_logger.experiment.id}", type="dataset"
            )
            artifact.add_file(str(OUTPUT_DIR / f"sub_{fold}.csv"))
            run.log_artifact(artifact)
            if fold != CFG.folds[-1]:
                wandb.finish()
        if CFG.use_mail:
            sender = Sender(
                from_address=FROM_ADDRESS, to_address=TO_ADDRESS, password=PASSWORD
            )
            subject = f"project: {CFG.project} exp_name: {CFG.exp_name} fold: {fold}"
            body = f"""Best validation score in fold {fold} is {checkpoint.best_model_score:.3f}.
            \nRun information is as follows:
            - project: {CFG.project}
            - exp_name: {CFG.exp_name}
            - fold: {fold}
            - run_id: {pl_logger.experiment.id}
            \nSee https://wandb.ai/home for more details.
            """
            sender.create_message(subject=subject, body=body)
            sender.send()

    # save
    df_oof.to_csv(OUTPUT_DIR / "oof.csv", index=False)
    if CFG.use_wandb:
        run = pl_logger.experiment
        artifact = wandb.Artifact(
            f"dataset-oof-{pl_logger.experiment.id}", type="dataset"
        )
        artifact.add_file(str(OUTPUT_DIR / f"oof.csv"))
        run.log_artifact(artifact)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        _logger.exception(e)
        if CFG.use_wandb:
            wandb.finish()
        if ENV in ["kaggle", "colab"]:
            print(e)
