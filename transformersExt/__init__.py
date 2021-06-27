from collections import OrderedDict

from transformers import CONFIG_MAPPING, MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING, MODEL_NAMES_MAPPING, TOKENIZER_MAPPING
from transformers.convert_slow_tokenizer import SLOW_TO_FAST_CONVERTERS, BertConverter, XLMRobertaConverter
from transformers.models.auto.modeling_auto import auto_class_factory

from .models.layoutlm_ext import (
    LayoutLMExtConfig,
    LayoutLMExtModel,
    LayoutLMExtForQuestionAnswering,
    LayoutLMExtForSequenceClassification,
    LayoutLMExtForTokenClassification,
)

MODEL_NAMES_MAPPING.update(
    [
        ("layoutlm_ext", "LayoutLMEXT"),
    ]
)

CONFIG_MAPPING.update(
    [
        ("layoutlm_ext", LayoutLMExtConfig),
    ]
)

MODEL_MAPPING.update(
    [
        (LayoutLMExtConfig, LayoutLMExtModel),
    ]
)

MODEL_FOR_QUESTION_ANSWERING_MAPPING = OrderedDict(
    [
        (LayoutLMExtConfig, LayoutLMExtForQuestionAnswering),
    ]
)

MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING = OrderedDict(
    [
        (LayoutLMExtConfig, LayoutLMExtForSequenceClassification),
    ]
)

MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING.update(
    [
        (LayoutLMExtConfig, LayoutLMExtForTokenClassification),
    ]
)

AutoModel = auto_class_factory("AutoModel", MODEL_MAPPING)

AutoModelForTokenClassification = auto_class_factory(
    "AutoModelForTokenClassification", MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING, head_doc="token classification"
)

AutoModelForQuestionAnswering = auto_class_factory(
    "AutoModelForQuestionAnswering", MODEL_FOR_QUESTION_ANSWERING_MAPPING, head_doc="question answering"
)

AutoModelForSequenceClassification = auto_class_factory(
    "AutoModelForSequenceClassification", MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING, head_doc="sequence classification"
)
