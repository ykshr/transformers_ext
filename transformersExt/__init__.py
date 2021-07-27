from collections import OrderedDict

from transformers import CONFIG_MAPPING, MODEL_NAMES_MAPPING, MODEL_MAPPING
from transformers.models.auto.modeling_auto import auto_class_factory

from .models.layoutlm2d import (
    LayoutLM2dConfig,
    LayoutLM2dModel,
)

MODEL_NAMES_MAPPING.update(
    [
        ("layoutlm2d", "LayoutLM2d"),
    ]
)

CONFIG_MAPPING.update(
    [
        ("layoutlm2d", LayoutLM2dConfig),
    ]
)

MODEL_MAPPING.update(
    [
        (LayoutLM2dConfig, LayoutLM2dModel),
    ]
)

AutoModel = auto_class_factory("AutoModel", MODEL_MAPPING)
