import warnings
import logging

warnings.simplefilter(action="ignore", category=FutureWarning)

logging.basicConfig()

import torch
import fire
import gc


try:
    from .utils import load_dataset, load_model, get_training_arguments
except ImportError:
    from utils import load_dataset, load_model, get_training_arguments

try:
    from .train import manual_train
except ImportError:
    from train import manual_train

from typing import List

import gc
import fire
import torch


train_arg = {
    "epochs": 4,
    "batch_size": 92,
    "accumulation_steps": 1,
    "warmup_ratio": 0.1,
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "blstm": True,
    "lstm_hidden_dim": 128,
    "lstm_num_layers": 2,
}


def fine_tuning(
    lstm_hidden_dim: List[int] = [64, 128, 256, 512],
    lstm_num_layers: List[int] = [2, 4, 8, 12],
    base_model: str = "pysentimiento/robertuito-base-uncased",
    dataset_path: str = "e:\\Media\\Python\\ID-v3-Scrapper\\model\\tweets_parsed.csv",
):
    dataset = load_dataset(dataset_path)

    dataset = dataset.map(
        lambda x: tokenizer(
            x["text"],
            padding=True if not train_arg.get("blstm", False) else "max_length",
            truncation=True,
        ),
        batched=True,
    )

    for hidden_dim in lstm_hidden_dim:
        for num_layers in lstm_num_layers:
            gc.collect()
            torch.cuda.empty_cache()

            training_args = get_training_arguments(base_model, "sentiment", train_arg)
            training_args["lstm_hidden_dim"] = hidden_dim
            training_args["lstm_num_layers"] = num_layers

            model, tokenizer = load_model(
                base_model,
                blstm=train_arg.get("blstm", False),
                lstm_hidden_dim=train_arg.get("lstm_hidden_dim", 128),
                lstm_num_layers=train_arg.get("lstm_num_layers", 2),
            )

            manual_train(
                model,
                training_args,
                dataset,
            )


if __name__ == "__main__":
    fire.Fire(fine_tuning)
