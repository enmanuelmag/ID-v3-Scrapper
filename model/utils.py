import os
import torch
import logging
import warnings
import pandas as pd
import torch.nn as nn
import torch.optim as optim

from transformers.trainer_utils import set_seed
from sklearn.model_selection import train_test_split
from pysentimiento.preprocessing import preprocess_tweet
from datasets import Dataset, Value, ClassLabel, Features, DatasetDict
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    AutoTokenizer,
)

logging.basicConfig()
warnings.filterwarnings("ignore")
logger = logging.getLogger("Utils")
logger.setLevel(logging.INFO)

random_state = 42
set_seed(random_state)

id2label = {0: "NEG", 1: "NEU", 2: "POS"}
label2id = {v: k for k, v in id2label.items()}

preprocessing_args = {
    "user_token": "@usuario",
    "url_token": "url",
    "hashtag_token": "hashtag",
}

BASE_MODEL = "pysentimiento/robertuito-base-uncased"


"""
MODELS UTILS
"""


class TransformerPlusBLSTM(nn.Module):
    def __init__(
        self,
        transformer_model,
        hidden_dim=128,
        output_dim=3,
        num_layers=2,
        dropout=0.5,
        loss_fn=None,
    ):
        assert loss_fn is not None, "Please provide a loss function"

        super(TransformerPlusBLSTM, self).__init__()

        self.transformer = transformer_model

        self.lstm = nn.LSTM(
            self.transformer.config.hidden_size,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

        self.fc = nn.Linear(
            hidden_dim * 2, output_dim
        )  # *2 por la bidireccionalidad de LSTM

        self.loss_fn = loss_fn

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        label=None,
        **kwargs,
    ):
        transformer_output = self.transformer.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )[
            0
        ]  # Salida del transformer (sin el head de clasificación)

        lstm_out, (hn, cn) = self.lstm(transformer_output)

        final_hidden_state = torch.cat(
            (hn[-2, :, :], hn[-1, :, :]), dim=1
        )  # Concatenar las dos direcciones

        logits = self.fc(final_hidden_state)

        loss = None
        if label is not None:
            loss = self.loss_fn(logits, label)

        return logits, loss


def config_device(model):
    """
    Config device
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Using device {device.upper()}")

    model.to(device)

    return model


def load_model(
    base_model,
    train_arg,
    max_length=128,
    auto_class=AutoModelForSequenceClassification,
    problem_type=None,
    skip_device=False,
):
    """
    Loads model and tokenizer
    """
    assert base_model == BASE_MODEL, "Please provide a valid model name"

    logger.info(f"Loading model {base_model} {id2label}")

    model = auto_class.from_pretrained(
        base_model,
        return_dict=True,
        num_labels=len(id2label),
        problem_type=problem_type,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.model_max_length = max_length

    label2id = {label: i for i, label in id2label.items()}

    special_tokens = list(preprocessing_args.values())

    tokenizer.add_tokens(special_tokens)

    model.resize_token_embeddings(len(tokenizer))
    model.config.id2label = id2label
    model.config.label2id = label2id

    if train_arg.get("blstm", False):
        model = TransformerPlusBLSTM(
            model,
            hidden_dim=train_arg.get("lstm_hidden_dim", 128),
            num_layers=train_arg.get("lstm_num_layers", 2),
            loss_fn=nn.CrossEntropyLoss(),
        )

    if not skip_device:
        model = config_device(model)

    return model, tokenizer


def load_dataset(
    dataset_path: str,
    split_test=0.15,
    preprocess=True,
    return_df=False,
    force=False,
    limit=None,
):

    if os.path.exists(dataset_path.replace(".csv", "_dataset")) and not force:
        logger.info(f"Loading dataset from {dataset_path.replace('.csv', '_dataset')}")
        dataset_dict = DatasetDict.load_from_disk(
            dataset_path.replace(".csv", "_dataset")
        )

        if limit is not None:
            dataset_dict["train"] = dataset_dict["train"].select(range(limit))
            dataset_dict["test"] = dataset_dict["test"].select(range(limit))
            dataset_dict["validation"] = dataset_dict["validation"].select(range(limit))

        return dataset_dict

    df = pd.read_csv(dataset_path)

    df["label"] = df["sentiment"].apply(
        lambda sentiment: label2id[
            sentiment[:3].upper()
        ]  # label2id[sentiment[:3].upper()]
    )

    features = ["tweet_text", "lang", "label"]

    df = df[features]
    columns = ["text", "lang", "label"]
    df.columns = columns

    df["text"] = df["text"].astype(str)
    df["lang"] = df["lang"].astype(str)

    train_df, temp_df = train_test_split(
        df, test_size=split_test, random_state=random_state
    )
    test_df, val_df = train_test_split(
        temp_df, test_size=0.5, random_state=random_state
    )
    logger.info(f"Dataset: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    if preprocess:

        def preprocess_with_args(x):
            return preprocess_tweet(x, **preprocessing_args)

        logger.info("Preprocessing train dataset")
        train_df["text"] = train_df["text"].apply(preprocess_with_args)
        logger.info("Preprocessing test dataset")
        test_df["text"] = test_df["text"].apply(preprocess_with_args)
        logger.info("Preprocessing val dataset")
        val_df["text"] = val_df["text"].apply(preprocess_with_args)

    if return_df:
        return train_df, test_df, val_df

    features = Features(
        {
            "text": Value("string"),
            "lang": Value("string"),
            "label": ClassLabel(num_classes=3, names=["neg", "neu", "pos"]),
        }
    )

    train_dataset = Dataset.from_pandas(
        train_df[columns], features=features, preserve_index=False
    )

    test_dataset = Dataset.from_pandas(
        test_df[columns], features=features, preserve_index=False
    )

    val_dataset = Dataset.from_pandas(
        val_df[columns], features=features, preserve_index=False
    )

    logger.info(f"Dataset instances created")

    dataset_dict = DatasetDict(
        train=train_dataset, test=test_dataset, validation=val_dataset
    )

    if limit is not None:
        dataset_dict["train"] = dataset_dict["train"].select(range(limit))
        dataset_dict["test"] = dataset_dict["test"].select(range(limit))
        dataset_dict["validation"] = dataset_dict["validation"].select(range(limit))

    # save dataset
    logger.info(f"Saving dataset to {dataset_path.replace('.csv', '_dataset')}")
    dataset_dict.save_to_disk(dataset_path.replace(".csv", "_dataset"))

    return dataset_dict


"""
UTILS
"""


def get_training_arguments(
    model_name,
    task_name,
    params={},
):

    args = TrainingArguments(
        output_dir=f"./results/{task_name}-es-{model_name}",
        num_train_epochs=params.get("epochs", 3),
        per_device_train_batch_size=params.get("batch_size", 32),
        per_device_eval_batch_size=params.get("batch_size", 32),
        gradient_accumulation_steps=params.get("accumulation_steps", 1),
        warmup_ratio=params.get("warmup_ratio", 0.1),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=params.get("learning_rate", 5e-5),
        do_eval=False,
        weight_decay=params.get("weight_decay", 0.01),
        fp16=True,
        logging_dir="./logs",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        group_by_length=True,
        overwrite_output_dir=True,
        report_to="mlflow",
        save_total_limit=3,
    )

    return args


def get_metrics(preds, labels, id2label=id2label):
    ret = {}

    f1s = []
    precs = []
    recalls = []

    is_multi_label = len(labels.shape) > 1 and labels.shape[-1] > 1

    if not is_multi_label:
        preds = preds.argmax(-1)

    for i, cat in id2label.items():

        if is_multi_label:
            cat_labels, cat_preds = labels[:, i], preds[:, i]

            cat_preds = cat_preds > 0
        else:
            cat_labels, cat_preds = labels == i, preds == i

        precision, recall, f1, _ = precision_recall_fscore_support(
            cat_labels,
            cat_preds,
            average="binary",
            zero_division=0,
        )

        f1s.append(f1)
        precs.append(precision)
        recalls.append(recall)

        ret[cat.lower() + "_f1"] = f1
        ret[cat.lower() + "_precision"] = precision
        ret[cat.lower() + "_recall"] = recall

    if not is_multi_label:
        _, _, micro_f1, _ = precision_recall_fscore_support(
            labels, preds, average="micro"
        )
        ret["micro_f1"] = micro_f1
        ret["acc"] = accuracy_score(labels, preds)
    else:
        _, _, micro_f1, _ = precision_recall_fscore_support(
            labels, preds > 0, average="micro"
        )
        ret["micro_f1"] = micro_f1
        ret["emr"] = accuracy_score(labels, preds > 0)

    ret["macro_f1"] = torch.Tensor(f1s).mean()
    ret["macro_precision"] = torch.Tensor(precs).mean()
    ret["macro_recall"] = torch.Tensor(recalls).mean()

    return ret


def compute_metrics(preds):
    labels = preds.label_ids

    return get_metrics(preds.predictions, labels, id2label)
