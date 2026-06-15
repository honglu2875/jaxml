import importlib
import logging
import sys

import pytest

pytestmark = pytest.mark.critical


def test_inference_engine_exports_documented_entry_points():
    from jaxml.inference_engine import Engine, InferenceConfig, SamplingMethod
    from jaxml.inference_engine.engine import Engine as EngineImpl
    from jaxml.inference_engine.engine import InferenceConfig as InferenceConfigImpl
    from jaxml.inference_engine.sampling import SamplingMethod as SamplingMethodImpl

    assert Engine is EngineImpl
    assert InferenceConfig is InferenceConfigImpl
    assert SamplingMethod is SamplingMethodImpl


def test_package_root_exports_documented_entry_points():
    from jaxml import Engine, GenerationConfig, InferenceConfig, ModelConfig, SamplingMethod, TextGenerationPipeline
    from jaxml.config import ModelConfig as ModelConfigImpl
    from jaxml.inference_engine import Engine as EngineImpl
    from jaxml.inference_engine import InferenceConfig as InferenceConfigImpl
    from jaxml.inference_engine import SamplingMethod as SamplingMethodImpl
    from jaxml.text_generation import GenerationConfig as GenerationConfigImpl
    from jaxml.text_generation import TextGenerationPipeline as TextGenerationPipelineImpl

    assert Engine is EngineImpl
    assert GenerationConfig is GenerationConfigImpl
    assert InferenceConfig is InferenceConfigImpl
    assert ModelConfig is ModelConfigImpl
    assert SamplingMethod is SamplingMethodImpl
    assert TextGenerationPipeline is TextGenerationPipelineImpl


def test_models_package_exports_architecture_entry_points():
    import jaxml.models as models
    from jaxml.models.gemma3 import GemmaDecoder, GemmaMLP, GemmaModel, GemmaModelWithHead, GemmaRMSNorm
    from jaxml.models.gpt_neox import GPTNeoXDecoder, GPTNeoXMLP, GPTNeoXModel, GPTNeoXModelWithHead
    from jaxml.models.llama import LlamaDecoder, LlamaMLP, LlamaModel, LlamaModelWithHead

    assert models.__all__ == [
        "GPTNeoXDecoder",
        "GPTNeoXMLP",
        "GPTNeoXModel",
        "GPTNeoXModelWithHead",
        "GemmaDecoder",
        "GemmaMLP",
        "GemmaModel",
        "GemmaModelWithHead",
        "GemmaRMSNorm",
        "LlamaDecoder",
        "LlamaMLP",
        "LlamaModel",
        "LlamaModelWithHead",
    ]
    assert models.GPTNeoXDecoder is GPTNeoXDecoder
    assert models.GPTNeoXMLP is GPTNeoXMLP
    assert models.GPTNeoXModel is GPTNeoXModel
    assert models.GPTNeoXModelWithHead is GPTNeoXModelWithHead
    assert models.GemmaDecoder is GemmaDecoder
    assert models.GemmaMLP is GemmaMLP
    assert models.GemmaModel is GemmaModel
    assert models.GemmaModelWithHead is GemmaModelWithHead
    assert models.GemmaRMSNorm is GemmaRMSNorm
    assert models.LlamaDecoder is LlamaDecoder
    assert models.LlamaMLP is LlamaMLP
    assert models.LlamaModel is LlamaModel
    assert models.LlamaModelWithHead is LlamaModelWithHead


def test_nn_package_exports_component_entry_points():
    import jaxml.nn as nn
    from jaxml.nn.attention import Attention, AttentionWithRoPE
    from jaxml.nn.embedding import Embed
    from jaxml.nn.linear import DenseGeneral
    from jaxml.nn.module import Block, Module
    from jaxml.nn.norms import LayerNorm, RMSNorm
    from jaxml.nn.position import RotaryEmbedding

    assert nn.__all__ == [
        "Attention",
        "AttentionWithRoPE",
        "Block",
        "DenseGeneral",
        "Embed",
        "LayerNorm",
        "Module",
        "RMSNorm",
        "RotaryEmbedding",
    ]
    assert nn.Attention is Attention
    assert nn.AttentionWithRoPE is AttentionWithRoPE
    assert nn.Block is Block
    assert nn.DenseGeneral is DenseGeneral
    assert nn.Embed is Embed
    assert nn.LayerNorm is LayerNorm
    assert nn.Module is Module
    assert nn.RMSNorm is RMSNorm
    assert nn.RotaryEmbedding is RotaryEmbedding


def test_experimental_package_exports_entry_points():
    from jaxml.experimental import RNNDiscrete, RNNDiscreteConfig
    from jaxml.experimental import __all__ as experimental_all
    from jaxml.experimental.rnn_discrete import RNNDiscrete as RNNDiscreteImpl
    from jaxml.experimental.rnn_discrete import RNNDiscreteConfig as RNNDiscreteConfigImpl

    assert experimental_all == ["RNNDiscrete", "RNNDiscreteConfig"]
    assert RNNDiscrete is RNNDiscreteImpl
    assert RNNDiscreteConfig is RNNDiscreteConfigImpl


def test_importing_jaxml_does_not_configure_root_logging():
    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    original_level = root_logger.level
    original_jaxml = sys.modules.pop("jaxml", None)
    root_logger.handlers.clear()
    root_logger.setLevel(logging.WARNING)

    try:
        importlib.import_module("jaxml")

        assert root_logger.handlers == []
        assert root_logger.level == logging.WARNING
    finally:
        sys.modules.pop("jaxml", None)
        if original_jaxml is not None:
            sys.modules["jaxml"] = original_jaxml
        root_logger.handlers[:] = original_handlers
        root_logger.setLevel(original_level)
