import warnings
import logging

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore")

logging.getLogger("mlflow.models.model").setLevel(logging.ERROR)
logging.basicConfig()

from tqdm.auto import tqdm
import mlflow
import torch
import fire
import gc

from typing import List

try:
    from .utils import load_dataset, load_model, get_training_arguments
except ImportError:
    from utils import load_dataset, load_model, get_training_arguments

try:
    from .train import manual_train
except ImportError:
    from train import manual_train


train_arg = {
    "epochs": 14,
    "batch_size": 92,
    "accumulation_steps": 1,
    "warmup_ratio": 0.1,
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    # "blstm": True,
    # "lstm_hidden_dim": 128,
    # "lstm_num_layers": 2,
}

BASE_MODEL = "pysentimiento/robertuito-base-uncased"
DATASET_PATH = (
    "e:\\Media\\Python\\ID-v3-Scrapper\\model\\data\\parsed\\tweets_parsed_pruned.csv"
)


def check_already_ran(train_arg: dict, model_name: str) -> bool:
    runs = mlflow.search_runs()

    for _, row in runs.iterrows():

        if row["tags.mlflow.runName"] == model_name:

            is_blstm = row.get("params.blstm") == str(train_arg.get("blstm", False))

            if is_blstm:
                same_hidden_dim = row["params.lstm_hidden_dim"] == str(
                    train_arg.get("lstm_hidden_dim", 0)
                )

                same_num_layers = row["params.lstm_num_layers"] == str(
                    train_arg.get("lstm_num_layers", 0)
                )

                if same_hidden_dim and same_num_layers:
                    return True

            is_conv1d = row.get("params.conv1d") == str(train_arg.get("conv1d", False))

            if is_conv1d:
                same_filters = row["params.conv1D_filters"] == str(
                    train_arg.get("conv1D_filters", 0)
                )

                same_kernel_size = row["params.conv1D_kernel_size"] == str(
                    train_arg.get("conv1D_kernel_size", 0)
                )

                if same_filters and same_kernel_size:
                    return True

    return False


def fine_tuning_blstm(base_model: str, dataset_path: str, force: bool = False):
    # blstm
    # lstm_hidden_dim = [64, 128, 256]  # 64, 256 SIN BUENOS RESULTADOS
    # lstm_num_layers = [4, 8, 10]  # 8 SIN BUENOS RESULTADOS

    configs = [{"lstm_hidden_dim": 128, "lstm_num_layers": 4}]

    progress_bar = tqdm(total=len(configs), desc="Fine tuning")

    for config in configs:
        gc.collect()
        torch.cuda.empty_cache()

        hidden_dim = config.get("lstm_hidden_dim", 0)
        num_layers = config.get("lstm_num_layers", 0)

        train_arg["blstm"] = True
        train_arg["conv1d"] = False
        train_arg["lstm_hidden_dim"] = hidden_dim
        train_arg["lstm_num_layers"] = num_layers

        if not force and check_already_ran(train_arg, "RobertuitoBiLSTM"):
            progress_bar.update(1)
            continue

        progress_bar.set_description(
            f"Training: {hidden_dim} hidden dim, {num_layers} layers"
        )

        training_args = get_training_arguments(base_model, "sentiment", train_arg)

        model, tokenizer = load_model(base_model, train_arg)

        dataset = load_dataset(dataset_path)

        dataset = dataset.map(
            lambda x: tokenizer(
                x["text"],
                padding=(True if not train_arg.get("blstm", False) else "max_length"),
                truncation=True,
            ),
            batched=True,
        )

        manual_train(
            model,
            training_args,
            train_arg,
            dataset,
        )

        progress_bar.update(1)


def fine_tuning_conv1D(base_model: str, dataset_path: str, force: bool = False):
    # conv1D
    # conv1D_filters = [64, 128, 256, 512]  # 512 new
    # conv1D_kernel_size = [3, 5, 7, 9]  # 9 new

    configs = [
        {"conv1D_filters": 256, "conv1D_kernel_size": 7},
        {"conv1D_filters": 512, "conv1D_kernel_size": 3},
        {"conv1D_filters": 512, "conv1D_kernel_size": 5},
        {"conv1D_filters": 512, "conv1D_kernel_size": 7},
        {"conv1D_filters": 512, "conv1D_kernel_size": 9},
    ]

    progress_bar = tqdm(total=len(configs), desc="Fine tuning")

    for config in configs:

        gc.collect()
        torch.cuda.empty_cache()

        filters = config["conv1D_filters"]
        kernel_size = config["conv1D_kernel_size"]

        # conv1D
        train_arg["conv1d"] = True
        train_arg["blstm"] = False
        train_arg["conv1D_filters"] = filters
        train_arg["conv1D_kernel_size"] = kernel_size

        # blstm
        train_arg["lstm_hidden_dim"] = 128
        train_arg["lstm_num_layers"] = 4

        if not force and check_already_ran(train_arg, "RobertuitoConv1DBiLSTM"):
            progress_bar.update(1)
            continue

        progress_bar.set_description(
            f"Training: {filters} filters, {kernel_size} kernel size"
        )

        training_args = get_training_arguments(base_model, "sentiment", train_arg)

        model, tokenizer = load_model(base_model, train_arg)

        dataset = load_dataset(dataset_path)

        dataset = dataset.map(
            lambda x: tokenizer(
                x["text"],
                padding="max_length",
                truncation=True,
            ),
            batched=True,
        )

        manual_train(
            model,
            training_args,
            train_arg,
            dataset,
        )

        progress_bar.update(1)


def fine_tuning(
    type_model: str,
    force: bool = False,
    base_model: str = BASE_MODEL,
    dataset_path: str = DATASET_PATH,
):
    assert type_model in ["blstm", "conv1D"], "Invalid type"

    if type_model == "blstm":
        fine_tuning_blstm(base_model, dataset_path, force)

    elif type_model == "conv1D":
        fine_tuning_conv1D(base_model, dataset_path, force)


if __name__ == "__main__":
    fire.Fire(fine_tuning)
