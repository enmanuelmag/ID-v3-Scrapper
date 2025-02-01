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

logging.getLogger("mlflow.models.model").setLevel(logging.ERROR)

logging.basicConfig()

logger = logging.getLogger("Utils")
logger.setLevel(logging.INFO)

warnings.filterwarnings("ignore")

random_state = 42
set_seed(random_state)

id2label = {0: "NEG", 1: "NEU", 2: "POS"}
label2id = {v: k for k, v in id2label.items()}

preprocessing_args = {
    "url_token": "url",
    "user_token": "@usuario",
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
        self.name = "TransformerPlusBLSTM"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.transformer = transformer_model

        self.lstm = nn.LSTM(
            input_size=self.transformer.config.hidden_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

        self.intermediate_fc = nn.Linear(
            hidden_dim * 2 + self.transformer.config.hidden_size, 256
        )  # *2 por la bidireccionalidad de LSTM + hidden_size de transformer

        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        self.dropout = nn.Dropout(dropout)

        self.softmax = nn.Softmax(dim=1)

        self.loss_fn = loss_fn

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        label=None,
        **kwargs,
    ):
        # Obtener las representaciones de los tokens del transformer
        transformer_outputs = self.transformer.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Obtener la última capa oculta del transformer
        last_hidden_state = transformer_outputs[
            0
        ]  # Salida del transformer (sin el head de clasificación)

        # Pasar el input_ids a la BLSTM
        lstm_out, (hn, cn) = self.lstm(last_hidden_state)

        # Concatenar las dos direcciones de la BLSTM
        # final_hidden_state = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)

        # aplicar pooling promedio al final_hidden_state
        avg_lstm_out = torch.mean(lstm_out, dim=1)

        combined_output = torch.cat((avg_lstm_out, last_hidden_state[:, 0, :]), dim=1)

        # Pasar la salida combinada por una capa oculta adicional
        intermediate_output = self.dropout(
            torch.relu(self.intermediate_fc(combined_output))
        )

        # Pasar la salida de la capa oculta a la capa totalmente conectada
        logits = self.fc(intermediate_output)

        loss = None
        if label is not None:
            loss = self.loss_fn(logits, label)

        return logits, loss


class TransformerConv1DBLSTM(nn.Module):
    def __init__(
        self,
        transformer_model,
        conv_out_channels,
        conv_kernel_size,
        lstm_hidden_dim,
        lstm_num_layers,
        output_dim,
        dropout=0.5,
    ):
        super(TransformerConv1DBLSTM, self).__init__()
        self.transformer = transformer_model
        self.name = "TransformerConv1DBLSTM"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.conv1d = nn.Conv1d(
            in_channels=self.transformer.config.hidden_size,
            out_channels=conv_out_channels,
            kernel_size=conv_kernel_size,
        )
        self.lstm = nn.LSTM(
            input_size=conv_out_channels,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.fc = nn.Linear(
            lstm_hidden_dim * 2, output_dim
        )  # *2 por la bidireccionalidad de LSTM
        self.dropout = nn.Dropout(dropout)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self, input_ids, attention_mask=None, token_type_ids=None, label=None, **kwargs
    ):
        transformer_output = self.transformer.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )[0]

        x = transformer_output.transpose(1, 2)
        x = self.conv1d(x)
        x = x.transpose(1, 2)

        lstm_out, (hn, cn) = self.lstm(x)

        final_hidden_state = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)

        logits = self.fc(self.dropout(final_hidden_state))

        loss = None
        if label is not None:
            loss = self.loss_fn(logits, label)

        return logits, loss


class TransformerConv1D(nn.Module):
    def __init__(
        self,
        transformer_model,
        conv_out_channels,
        conv_kernel_size,
        hidden_dim,
        output_dim,
        dropout=0.5,
    ):
        super(TransformerConv1D, self).__init__()
        self.transformer = transformer_model
        self.name = "TransformerConv1D-V2"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.conv1d = nn.Conv1d(
            in_channels=self.transformer.config.hidden_size,
            out_channels=conv_out_channels,
            kernel_size=conv_kernel_size,
        )

        self.dropout = nn.Dropout(dropout)

        self.dense = nn.Linear(conv_out_channels, hidden_dim)

        self.fc = nn.Linear(hidden_dim, output_dim)

        self.softmax = nn.Softmax(dim=1)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self, input_ids, attention_mask=None, token_type_ids=None, label=None, **kwargs
    ):
        transformer_output = self.transformer.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )[0]

        x = transformer_output.transpose(1, 2)

        x = self.conv1d(x)

        x = x.transpose(1, 2)

        x = torch.mean(x, dim=1)

        x = self.dense(self.dropout(x))

        x = torch.tanh(x)

        x = self.dropout(x)

        x = self.fc(x)

        x = self.softmax(x)

        loss = None
        if label is not None:
            loss = self.loss_fn(x, label)

        return x, loss


def config_device(model):
    """
    Config device
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.debug(f"Using device {device.upper()}")

    model.to(device)

    return model


def load_model(
    base_model,
    train_arg,
    max_length=128,
    auto_class=AutoModelForSequenceClassification,
    problem_type=None,
    skip_device=False,
    custom_model=None,
):
    """
    Loads model and tokenizer
    """
    # assert base_model == BASE_MODEL, "Please provide a valid model name"

    logger.debug(f"Loading model {base_model}")

    if not custom_model:
        model = auto_class.from_pretrained(
            base_model,
            return_dict=True,
            num_labels=len(id2label),
            problem_type=problem_type,
        )
    else:
        model = custom_model

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.model_max_length = max_length

    label2id = {label: i for i, label in id2label.items()}

    special_tokens = list(preprocessing_args.values())

    tokenizer.add_tokens(special_tokens)

    model.resize_token_embeddings(len(tokenizer))
    model.config.id2label = id2label
    model.config.label2id = label2id

    if train_arg.get("blstm", False):
        logger.debug("Adding BLSTM layer")
        model = TransformerPlusBLSTM(
            model,
            hidden_dim=train_arg.get("lstm_hidden_dim", 128),
            num_layers=train_arg.get("lstm_num_layers", 2),
            loss_fn=nn.CrossEntropyLoss(),
        )
    elif train_arg.get("conv1d", False):
        logger.debug("Adding Conv1D and BLSTM layer")
        model = TransformerConv1D(
            transformer_model=model,
            conv_out_channels=train_arg.get("conv_out_channels", 256),
            conv_kernel_size=train_arg.get("conv_kernel_size", 3),
            hidden_dim=train_arg.get("hidden_dim", 128),
            output_dim=len(id2label),
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
        logger.debug(f"Loading dataset from {dataset_path.replace('.csv', '_dataset')}")
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
    logger.debug(f"Dataset: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    if preprocess:

        def preprocess_with_args(x):
            return preprocess_tweet(x, **preprocessing_args)

        logger.debug("Preprocessing train dataset")
        train_df["text"] = train_df["text"].apply(preprocess_with_args)
        logger.debug("Preprocessing test dataset")
        test_df["text"] = test_df["text"].apply(preprocess_with_args)
        logger.debug("Preprocessing val dataset")
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

    logger.debug(f"Dataset instances created")

    dataset_dict = DatasetDict(
        train=train_dataset, test=test_dataset, validation=val_dataset
    )

    if limit is not None:
        dataset_dict["train"] = dataset_dict["train"].select(range(limit))
        dataset_dict["test"] = dataset_dict["test"].select(range(limit))
        dataset_dict["validation"] = dataset_dict["validation"].select(range(limit))

    # save dataset
    logger.debug(f"Saving dataset to {dataset_path.replace('.csv', '_dataset')}")
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
