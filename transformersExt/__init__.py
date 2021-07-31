from collections import OrderedDict

from transformers import CONFIG_MAPPING, MODEL_NAMES_MAPPING, MODEL_MAPPING

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
