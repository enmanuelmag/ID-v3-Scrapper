import warnings
import logging

logging.getLogger("mlflow.models.model").setLevel(logging.ERROR)
logging.basicConfig()

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore")

import mlflow
import torch
import fire
import time
import gc
import numpy as np

from tqdm.auto import tqdm
from notifier import Notifier
from mlflow.models import infer_signature
from transformers import AdamW, get_scheduler
from transformers import (
    DataCollatorWithPadding,
    Trainer,
)


try:
    from .utils import (
        load_dataset,
        load_model,
        get_training_arguments,
        compute_metrics,
        get_metrics,
    )
except ImportError:
    from utils import (
        load_dataset,
        load_model,
        get_training_arguments,
        compute_metrics,
        get_metrics,
    )


train_arg = {
    "epochs": 4,
    "batch_size": 92,
    "accumulation_steps": 1,
    "warmup_ratio": 0.1,
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    # "blstm": False,
    # "lstm_hidden_dim": 128,
    # "lstm_num_layers": 2,
}


class CustomTrainer(Trainer):
    def log(self, logs: dict, start_time: float = None):
        step = self.state.global_step
        current_epoch = int(self.state.epoch)

        # Agregar prefijo a los logs
        start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))

        metrics = " | ".join([f"{k}: {v:.4f}" for k, v in logs.items() if k != "epoch"])

        log = f"[{start_time_str}] [Epoch {current_epoch} | Step {step}] {metrics}"

        print()
        print(log)
        print()

        super().log(logs, start_time)


def manual_train(model, train_args, train_arg, dataset):

    run_name = (
        "RobertuitoBiLSTM"
        if train_arg.get("blstm", False)
        else "RobertuitoConv1DBiLSTM"
    )

    params_model = "\n".join(
        [f"{str(k).capitalize()}: {v}" for k, v in train_arg.items()]
    )

    notifier = Notifier(
        title=f"Training \n{params_model}",
        webhook_url="https://discord.com/api/webhooks/1330403976205832284/Vv32bfpZW_aGikEzQ2sEYJlcbtEyAE10n2CHIjigZkKkiLeCHVg0Et2IWKz_SWo_m0a3",
    )

    notifier(
        msg=f"Starting train",
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    num_epochs = train_args.num_train_epochs
    learning_rate = train_args.learning_rate

    train_dataloader = dataset["train"].batch(train_args.per_device_train_batch_size)
    test_dataloader = dataset["test"].batch(train_args.per_device_train_batch_size)

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    num_training_steps = num_epochs * len(train_dataloader)

    current_step = 0

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    best_macro_f1 = float("-inf")

    try:
        with mlflow.start_run(run_name=run_name):
            run_id = mlflow.active_run().info.run_id

            mlflow.log_params(train_arg)

            progress_bar = tqdm(range(num_training_steps))

            for epoch_idx in range(num_epochs):

                model.train()
                for i, batch in enumerate(train_dataloader):
                    batch["input_ids"] = torch.tensor(batch["input_ids"]).to(device)
                    batch["attention_mask"] = torch.tensor(batch["attention_mask"]).to(
                        device
                    )
                    batch["token_type_ids"] = torch.tensor(batch["token_type_ids"]).to(
                        device
                    )
                    batch["label"] = torch.tensor(batch["label"]).to(device)

                    logits_train, loss = model(**batch)

                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    current_step += 1

                    ret = get_metrics(
                        logits_train.detach().cpu().numpy(),
                        batch["label"].detach().cpu().numpy(),
                    )

                    for k, v in ret.items():
                        mlflow.log_metric(f"train_{k}", v, step=current_step)

                    mlflow.log_metric("train_loss", loss.item(), step=current_step)

                model.eval()

                ret_mean = {}

                for batch in test_dataloader:
                    batch["input_ids"] = torch.tensor(batch["input_ids"]).to(device)
                    batch["attention_mask"] = torch.tensor(batch["attention_mask"]).to(
                        device
                    )
                    batch["token_type_ids"] = torch.tensor(batch["token_type_ids"]).to(
                        device
                    )
                    batch["label"] = torch.tensor(batch["label"]).to(device)

                    with torch.no_grad():
                        logits, _ = model(**batch)

                        ret = get_metrics(
                            logits.detach().cpu().numpy(),
                            batch["label"].detach().cpu().numpy(),
                        )

                        for k, v in ret.items():
                            if k not in ret_mean:
                                ret_mean[k] = 0
                            ret_mean[k] += v

                for k in ret_mean:
                    ret_mean[k] /= len(test_dataloader)

                current_macro_f1 = ret_mean["macro_f1"]

                if current_macro_f1 > best_macro_f1:
                    best_macro_f1 = current_macro_f1

                    mlflow.pytorch.log_model(model, "model")

                text = f"Epoch {epoch_idx + 1}\n"

                for k, v in ret_mean.items():
                    mlflow.log_metric(f"eval_{k}", v, step=current_step)
                    text += f"{k}: {v:.6f}\n"

                text += f"Best macro F1: {best_macro_f1:.10f}"

                notifier(msg=f"{text}")

    except Exception as e:
        print(f"Error on training: {e}")
        notifier(
            msg=f"Error: {e}",
        )
        raise e


def train_model(
    base_model: str = "pysentimiento/robertuito-base-uncased",
    dataset_path: str = "e:\\Media\\Python\\ID-v3-Scrapper\\model\\data\\parsed\\tweets_parsed_pruned.csv",
    limit: int = None,
):
    gc.collect()
    torch.cuda.empty_cache()

    training_args = get_training_arguments(base_model, "sentiment", train_arg)

    dataset = load_dataset(dataset_path, limit=limit)

    model, tokenizer = load_model(base_model, train_arg)

    dataset = dataset.map(
        lambda x: tokenizer(
            x["text"],
            padding=True if not train_arg.get("blstm", False) else "max_length",
            truncation=True,
        ),
        batched=True,
    )

    if not train_arg.get("blstm", False):
        with mlflow.start_run():
            model.train()

            data_collator = DataCollatorWithPadding(tokenizer, padding="longest")

            trainer_args = {
                "model": model,
                "args": training_args,
                "compute_metrics": compute_metrics,
                "train_dataset": dataset["train"],
                "eval_dataset": dataset["test"],
                "data_collator": data_collator,
                "tokenizer": tokenizer,
                "callbacks": [],
            }
            active_run = mlflow.active_run()
            artifact_uri = active_run.info.artifact_uri
            trainer = CustomTrainer(**trainer_args)

            model_path = f"{artifact_uri}/model".replace("file:///", "")
            trainer.train()
            trainer.predict(dataset["test"])
            trainer.save_model(model_path)
    else:
        manual_train(
            model,
            training_args,
            train_arg,
            dataset,
        )


if __name__ == "__main__":
    fire.Fire(train_model)
