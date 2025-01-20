import torch
import logging
import transformers

from pysentimiento import preprocess_tweet
from datasets import Dataset
from torch.nn import functional as F

from .utils import load_model, preprocessing_args, id2label, label2id

logger = logging.getLogger("Analyzer")

# Set logging level to error
logger.setLevel(logging.WARNING)

transformers.logging.set_verbosity_error()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AnalyzerOutput:
    """
    Base class for classification output
    """

    def __init__(self, sentence, context, probas, is_multilabel=False):
        """
        Constructor
        """
        self.sentence = sentence
        self.probas = probas
        self.context = context
        self.is_multilabel = is_multilabel
        if not is_multilabel:
            self.output = max(probas.items(), key=lambda x: x[1])[0]
        else:
            self.output = [k for k, v in probas.items() if v > 0.5]

    def __repr__(self):
        ret = f"{self.__class__.__name__}"
        if not self.is_multilabel:
            formatted_probas = sorted(self.probas.items(), key=lambda x: -x[1])
        else:
            formatted_probas = list(self.probas.items())
        formatted_probas = [f"{k}: {v:.3f}" for k, v in formatted_probas]
        formatted_probas = "{" + ", ".join(formatted_probas) + "}"
        ret += f"(output={self.output}, probas={formatted_probas})"

        return ret


class TokenClassificationOutput:
    """
    Output for token classification
    """

    def __init__(self, sentence, tokens, labels, probas, entities=None):
        """
        Constructor
        """
        self.sentence = sentence
        self.tokens = tokens
        self.labels = labels
        self.entities = entities
        self.probas = probas

    def __repr__(self):
        ret = f"{self.__class__.__name__}"

        if self.entities:
            formatted_entities = ", ".join(
                f'{entity["text"]} ({entity["type"]})' for entity in self.entities
            )
            ret += f"(entities=[{formatted_entities}], tokens={self.tokens}, labels={self.labels})"
        else:
            ret += f"(tokens={self.tokens}, labels={self.labels})"

        return ret


class BaseAnalyzer:
    def __init__(
        self, model, tokenizer, preprocessing_args=preprocessing_args, batch_size=32
    ):
        """
        Constructor for SentimentAnalyzer class

        Arguments:

        model (nn.Module): HuggingFace model
        tokenizer (PretrainedTokenizer): HuggingFace tokenizer
        task (string): task to perform (string)
        preprocessing_args (dict): dict with preprocessing arguments used in the preprocess_tweet function
        batch_size: batch size for inference
        compile (bool): whether to compile the model or not using pytorch compile (default: True)


        """
        self.model = model

        self.tokenizer = tokenizer
        self.preprocessing_args = preprocessing_args
        self.batch_size = batch_size

        self.tokenizer.model_max_length = 128
        self.problem_type = None
        self.id2label = id2label

    def _tokenize(self, batch):
        # If context is present, use it
        if "context" in batch:
            inputs = [batch["text"], batch["context"]]
        else:
            inputs = [batch["text"]]
        return self.tokenizer(
            *inputs,
            padding=False,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        )


class AnalyzerForSequenceClassification(BaseAnalyzer):
    """
    Wrapper to use sentiment analysis models as black-box
    """

    @classmethod
    def from_model_name(
        cls, model_name, task, preprocessing_args={}, batch_size=32, **kwargs
    ):
        """
        Constructor for SentimentAnalyzer class

        Arguments:

        model_name: str or path
            Model name or
        """
        train_arg = {
            "blstm": True,
        }
        model, tokenizer = load_model(model_name, train_arg)
        return cls(model, tokenizer, task, preprocessing_args, batch_size, **kwargs)

    def _get_output(self, sentence, logits, context=None):
        """
        Get output from logits

        It takes care of the type of problem: single or multi label classification
        """
        if self.problem_type == "multi_label_classification":
            is_multilabel = True
            probs = torch.sigmoid(logits).view(-1)
        else:
            is_multilabel = False
            probs = torch.softmax(logits, dim=1).view(-1)

        probas = {self.id2label[i]: probs[i].item() for i in self.id2label}
        return AnalyzerOutput(
            sentence, probas=probas, is_multilabel=is_multilabel, context=context
        )

    def _predict_single(self, sentence, context, preprocess_context):
        """
        Predict single

        Do it this way (without creating dataset) to make it faster
        """
        sentence = preprocess_tweet(sentence, **self.preprocessing_args)
        inputs = [sentence]

        if context:
            if preprocess_context:
                context = preprocess_tweet(context, **self.preprocessing_args)
            inputs.append(context)
        idx = (
            torch.LongTensor(
                self.tokenizer.encode(
                    *inputs,
                    truncation=True,
                    max_length=self.tokenizer.model_max_length,
                )
            )
            .view(1, -1)
            .to(device)
        )
        output = self.model(idx)
        logits = output.logits
        return self._get_output(sentence, logits)

    def predict(self, inputs, context=None, target=None, preprocess_context=True):
        """
        Return most likely class for the sentence

        Arguments:
        ----------
        inputs: string or list of strings
            A single or a list of sentences to be predicted

        context: string or list of strings
            A single or a list of context to be used for the prediction

        target: string or list of strings
            A rename of context

        preprocess_context: bool
            Whether to preprocess the context or not

        Returns:
        --------
            List or single AnalyzerOutput
        """

        context = context or target

        # If single string => predict it single
        if isinstance(inputs, str):
            if context and not isinstance(context, str):
                raise ValueError("Context must be a string")
            return self._predict_single(
                inputs, context=context, preprocess_context=preprocess_context
            )

        data = {
            "text": [
                preprocess_tweet(sent, **self.preprocessing_args) for sent in inputs
            ]
        }

        if context:
            data["context"] = [
                (
                    preprocess_tweet(context, **self.preprocessing_args)
                    if preprocess_context
                    else context
                )
                for _ in range(len(inputs))
            ]

        dataset = Dataset.from_dict(data)
        dataset = dataset.map(self._tokenize, batched=True, batch_size=self.batch_size)

        # output = self.eval_trainer.predict(dataset)
        output = self.model(**dataset)
        logits = torch.tensor(output.predictions)

        if context is None:
            # Just to make this clearer
            data["context"] = [None] * len(data["text"])

        rets = [
            self._get_output(sent, logits_row.view(1, -1), context=context)
            for sent, context, logits_row in zip(data["text"], data["context"], logits)
        ]

        return rets


def create_analyzer_blstm(
    task=None,
    lang=None,
    model_name=None,
    preprocessing_args=preprocessing_args,
    **kwargs,
):
    """
    Create analyzer for the given task

    Arguments:
    ----------
    task: str
        Task name ("sentiment", "emotion", "hate_speech", "irony", "ner", "pos")
    lang: str
        Language code (accepts "en", "es", "it", "pt". See documentation for further information)
    model_name: str
        Model name or path
    preprocessing_args: dict
        Preprocessing arguments for `preprocess_tweet` function

    Returns:
    --------
        Analyzer object for the given task and language
    """
    if not (model_name or (lang and task)):
        raise ValueError("model_name or (lang and task) must be provided")

    preprocessing_args = preprocessing_args or {}

    return AnalyzerForSequenceClassification.from_model_name(
        model_name=model_name,
        task=task,
        preprocessing_args=preprocessing_args,
        lang=lang,
        **kwargs,
    )
