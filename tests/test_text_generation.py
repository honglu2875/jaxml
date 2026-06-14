import numpy as np
import pytest

from jaxml.inference_engine.engine import InferenceConfig
from jaxml.text_generation import GenerationConfig, TextGenerationPipeline


class DummyTokenizer:
    def __init__(self, include_attention_mask=True):
        self.include_attention_mask = include_attention_mask
        self.encode_calls = []
        self.decode_calls = []

    def __call__(self, prompts, return_tensors, **kwargs):
        self.encode_calls.append((prompts, return_tensors, kwargs))
        assert return_tensors == "np"
        lengths = [max(1, len(prompt.split())) for prompt in prompts]
        max_len = max(lengths)
        input_ids = np.zeros((len(prompts), max_len), dtype=np.int32)
        attention_mask = np.zeros_like(input_ids)
        for row, length in enumerate(lengths):
            input_ids[row, :length] = np.arange(1, length + 1)
            attention_mask[row, :length] = 1
        encoded = {"input_ids": input_ids}
        if self.include_attention_mask:
            encoded["attention_mask"] = attention_mask
        return encoded

    def batch_decode(self, tokens, **kwargs):
        self.decode_calls.append((np.array(tokens), kwargs))
        return [" ".join(map(str, row.tolist())) for row in np.array(tokens)]


class DummyEngine:
    def __init__(self):
        self.generate_calls = []

    def generate(self, input_ids, attention_mask=None, **kwargs):
        input_ids = np.array(input_ids)
        self.generate_calls.append((input_ids, None if attention_mask is None else np.array(attention_mask), kwargs))
        new_tokens = np.full((input_ids.shape[0], kwargs["max_new_tokens"]), 9, dtype=np.int32)
        if kwargs["include_prompt"]:
            return np.concatenate((input_ids, new_tokens), axis=1)
        return new_tokens


def test_generate_text_returns_string_for_single_prompt():
    tokenizer = DummyTokenizer()
    engine = DummyEngine()
    pipeline = TextGenerationPipeline(engine=engine, tokenizer=tokenizer)

    text = pipeline.generate_text(
        "hello world",
        generation_config=GenerationConfig(max_new_tokens=2, temperature=0.0, include_prompt=True),
        decode_kwargs={"skip_special_tokens": False},
    )

    assert text == "1 2 9 9"
    assert len(tokenizer.encode_calls) == 1
    assert tokenizer.encode_calls[0][2] == {"padding": True}
    assert tokenizer.decode_calls[0][1] == {"skip_special_tokens": False}
    assert engine.generate_calls[0][2]["temperature"] == 0.0
    assert engine.generate_calls[0][2]["max_new_tokens"] == 2
    assert np.array_equal(engine.generate_calls[0][1], np.array([[1, 1]], dtype=np.int32))


def test_generate_tokens_handles_batches_and_generation_overrides():
    tokenizer = DummyTokenizer(include_attention_mask=False)
    engine = DummyEngine()
    pipeline = TextGenerationPipeline(engine=engine, tokenizer=tokenizer)

    tokens = pipeline.generate_tokens(
        ["one", "two words"],
        generation_config=GenerationConfig(max_new_tokens=3, include_prompt=True),
        tokenize_kwargs={"padding": "longest"},
        include_prompt=False,
    )

    assert tokens.shape == (2, 3)
    assert np.all(tokens == 9)
    assert engine.generate_calls[0][1] is None
    assert engine.generate_calls[0][2]["include_prompt"] is False
    assert tokenizer.encode_calls[0][2] == {"padding": "longest"}


def test_generate_tokens_rejects_empty_prompt_batch():
    tokenizer = DummyTokenizer()
    engine = DummyEngine()
    pipeline = TextGenerationPipeline(engine=engine, tokenizer=tokenizer)

    with pytest.raises(ValueError, match="at least one prompt"):
        pipeline.generate_tokens([])

    assert tokenizer.encode_calls == []
    assert engine.generate_calls == []


def test_from_hf_wires_loader_engine_and_tokenizer(monkeypatch):
    calls = {}

    class FakeEngine:
        def __init__(self, model, config, params, dtype, cache_stride):
            calls["engine_init"] = (model, config, params, dtype, cache_stride)

        def init_params(self, use_tpu):
            calls["use_tpu"] = use_tpu

    def fake_load_model_from_hf(name, architecture, dtype, **kwargs):
        calls["load_model"] = (name, architecture, dtype, kwargs)
        return "model", {"params": "params"}

    monkeypatch.setattr("jaxml.text_generation.Engine", FakeEngine)
    monkeypatch.setattr("jaxml.text_generation.load_model_from_hf", fake_load_model_from_hf)

    tokenizer = DummyTokenizer()
    inference_config = InferenceConfig(tp_size=2)
    pipeline = TextGenerationPipeline.from_hf(
        "some/model",
        architecture="llama",
        model_dtype="bfloat16",
        engine_dtype="engine-dtype",
        inference_config=inference_config,
        use_tpu=True,
        cache_stride=128,
        tokenizer=tokenizer,
        model_kwargs={"trust_remote_code": True},
    )

    assert pipeline.tokenizer is tokenizer
    assert calls["load_model"] == ("some/model", "llama", "bfloat16", {"trust_remote_code": True})
    assert calls["engine_init"] == ("model", inference_config, {"params": "params"}, "engine-dtype", 128)
    assert calls["use_tpu"] is True


def test_load_model_from_hf_dispatches_architecture(monkeypatch):
    import jaxml.hf_utils as hf_utils

    monkeypatch.setattr(hf_utils, "load_llama_from_hf", lambda *args, **kwargs: ("llama", args, kwargs))
    monkeypatch.setattr(hf_utils, "_infer_hf_architecture", lambda *args, **kwargs: "llama")

    assert hf_utils.load_model_from_hf("model", dtype="float16")[0] == "llama"
    assert hf_utils.load_model_from_hf("model", architecture="llama", dtype="float16")[2]["dtype"] == "float16"

    with pytest.raises(ValueError, match="Unsupported"):
        hf_utils.load_model_from_hf("model", architecture="unknown")
