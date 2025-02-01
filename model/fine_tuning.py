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

try:
    from .utils import load_dataset, load_model, load_tokenizer
except ImportError:
    from utils import load_dataset, load_model, load_tokenizer

try:
    from .train import manual_train
except ImportError:
    from train import manual_train


train_arg = {
    "batch_size": 128,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "learning_rate": 0.0001,  #  0.00005
    "accumulation_steps": 1,
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


def fine_tuning_blstm(
    base_model: str, dataset_path: str, epochs: int, force: bool = False
):
    # blstm
    # lstm_hidden_dim = [64, 128, 256]  # 64, 256 SIN BUENOS RESULTADOS
    # lstm_num_layers = [4, 8, 10]  # 8 SIN BUENOS RESULTADOS

    configs = [
        {"lstm_hidden_dim": 128, "lstm_num_layers": 4},
        {"lstm_hidden_dim": 128, "lstm_num_layers": 8},
        {"lstm_hidden_dim": 128, "lstm_num_layers": 10},
        {"lstm_hidden_dim": 256, "lstm_num_layers": 4},
        {"lstm_hidden_dim": 256, "lstm_num_layers": 8},
        {"lstm_hidden_dim": 256, "lstm_num_layers": 10},
        {"lstm_hidden_dim": 512, "lstm_num_layers": 4},
        {"lstm_hidden_dim": 512, "lstm_num_layers": 8},
        {"lstm_hidden_dim": 512, "lstm_num_layers": 10},
    ]

    dataset = load_dataset(dataset_path)

    tokenizer = load_tokenizer(base_model)

    dataset = dataset.map(
        lambda x: tokenizer(
            x["text"],
            padding="max_length",
            truncation=True,
        ),
        batched=True,
    )

    dataset["train"] = dataset["train"].batch(train_arg.get("batch_size"))
    dataset["test"] = dataset["test"].batch(train_arg.get("batch_size"))

    progress_bar = tqdm(total=len(configs), desc="Fine tuning")

    for config in configs:
        gc.collect()
        torch.cuda.empty_cache()

        train_arg["epochs"] = epochs

        hidden_dim = config.get("lstm_hidden_dim", 0)
        num_layers = config.get("lstm_num_layers", 0)

        train_arg["type_model"] = "blstm"
        train_arg["lstm_hidden_dim"] = hidden_dim
        train_arg["lstm_num_layers"] = num_layers

        if not force and check_already_ran(train_arg, "RobertuitoLSTM"):
            progress_bar.update(1)
            continue

        progress_bar.set_description(
            f"Training: {hidden_dim} hidden dim, {num_layers} layers"
        )

        model, _ = load_model(base_model, train_arg)

        model.transformer.resize_token_embeddings(len(tokenizer))

        manual_train(model, train_arg, dataset)

        progress_bar.update(1)


def fine_tuning_conv1D(
    base_model: str, dataset_path: str, epochs: int, force: bool = False
):
    configs = [
        # 64
        # {"conv1D_filters": 64, "conv1D_kernel_size": 3},
        # {"conv1D_filters": 64, "conv1D_kernel_size": 5},
        # {"conv1D_filters": 64, "conv1D_kernel_size": 7},
        # {"conv1D_filters": 64, "conv1D_kernel_size": 9},
        # 128
        # {"conv1D_filters": 128, "conv1D_kernel_size": 3},
        {"conv1D_filters": 128, "conv1D_kernel_size": 5},
        {"conv1D_filters": 128, "conv1D_kernel_size": 12},
        # {"conv1D_filters": 128, "conv1D_kernel_size": 7},
        # {"conv1D_filters": 128, "conv1D_kernel_size": 9},
        # 256
        # {"conv1D_filters": 256, "conv1D_kernel_size": 3},
        # {"conv1D_filters": 256, "conv1D_kernel_size": 5},
        {"conv1D_filters": 256, "conv1D_kernel_size": 7},
        # {"conv1D_filters": 256, "conv1D_kernel_size": 9},
        # 512
        # {"conv1D_filters": 512, "conv1D_kernel_size": 3},
        {"conv1D_filters": 512, "conv1D_kernel_size": 7},
        {"conv1D_filters": 512, "conv1D_kernel_size": 9},
        # {"conv1D_filters": 512, "conv1D_kernel_size": 9},
    ]

    dataset = load_dataset(dataset_path)

    tokenizer = load_tokenizer(base_model)

    dataset = dataset.map(
        lambda x: tokenizer(
            x["text"],
            padding="max_length",
            truncation=True,
        ),
        batched=True,
    )

    dataset["train"] = dataset["train"].batch(train_arg.get("batch_size"))
    dataset["test"] = dataset["test"].batch(train_arg.get("batch_size"))

    progress_bar = tqdm(total=len(configs), unit="model")

    for config in configs:
        gc.collect()
        torch.cuda.empty_cache()

        filters = config["conv1D_filters"]
        kernel_size = config["conv1D_kernel_size"]

        train_arg["epochs"] = epochs

        # conv1D
        train_arg["type_model"] = "conv1D"
        train_arg["conv1D_filters"] = filters
        train_arg["conv1D_kernel_size"] = kernel_size

        if not force and check_already_ran(train_arg, "RobertuitoConv1D"):
            progress_bar.update(1)
            continue

        progress_bar.set_description(
            f"Training model {filters} filters - {kernel_size} kernel size"
        )

        model, _ = load_model(base_model, train_arg)

        model.transformer.resize_token_embeddings(len(tokenizer))

        manual_train(model, train_arg, dataset)

        progress_bar.update(1)


def fine_tuning(
    type_model: str,
    force: bool = False,
    epochs: int = 6,
    base_model: str = BASE_MODEL,
    dataset_path: str = DATASET_PATH,
):
    assert type_model in ["blstm", "conv1D"], "Invalid type"

    if type_model == "blstm":
        fine_tuning_blstm(base_model, dataset_path, epochs, force)

    elif type_model == "conv1D":
        fine_tuning_conv1D(base_model, dataset_path, epochs, force)


if __name__ == "__main__":
    fire.Fire(fine_tuning)
