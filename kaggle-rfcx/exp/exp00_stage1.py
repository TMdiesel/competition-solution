# default package
import logging
import pathlib
import sys
import os
import typing as t

# third party package
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl

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
OUTPUT_DIR = load_env("OUTPUT_DIR")
INPUT_DIR = load_env("INPUT_DIR")


# =================================================
# Config
# =================================================
class CFG:
    debug = False
    log_dir = "./kaggle-rfcx/logs"

    seed = 46

    # data
    use_train_data = "tp"


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


def main() -> None:
    # setting
    init_root_logger(pathlib.Path(CFG.log_dir))
    _logger.setLevel(logging.INFO)
    _logger.info(ENV)

    pl.seed_everything(CFG.seed)

    # load data
    df_meta, train_audio_path = load_train_metadata(CFG)
    df_sub, test_audio_path = load_test_metadata(CFG)


if __name__ == "__main__":
    main()
