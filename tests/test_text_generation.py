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


class StaticTokenizer:
    def __init__(self, encoded):
        self.encoded = encoded
        self.encode_calls = []
        self.decode_calls = []

    def __call__(self, prompts, return_tensors, **kwargs):
        self.encode_calls.append((prompts, return_tensors, kwargs))
        return self.encoded

    def batch_decode(self, tokens, **kwargs):
        self.decode_calls.append((np.array(tokens), kwargs))
        return ["decoded"]


@pytest.mark.parametrize("field_name", ["seed", "max_new_tokens", "top_k"])
@pytest.mark.parametrize("value", [True, np.bool_(True), 1.5])
def test_generation_config_rejects_non_integer_count_fields(field_name, value):
    with pytest.raises(TypeError, match=f"{field_name} must be an integer"):
        GenerationConfig(**{field_name: value})


def test_generation_config_rejects_negative_max_new_tokens():
    with pytest.raises(ValueError, match="max_new_tokens must be non-negative"):
        GenerationConfig(max_new_tokens=-1)


def test_generation_config_rejects_negative_top_k():
    with pytest.raises(ValueError, match="top_k must be non-negative"):
        GenerationConfig(top_k=-1)


@pytest.mark.parametrize("field_name", ["top_p", "min_p", "temperature"])
@pytest.mark.parametrize("value", [True, np.bool_(True), "1.0"])
def test_generation_config_rejects_non_real_sampling_fields(field_name, value):
    with pytest.raises(TypeError, match=f"{field_name} must be a real number"):
        GenerationConfig(**{field_name: value})


@pytest.mark.parametrize("field_name", ["top_p", "min_p", "temperature"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_generation_config_rejects_non_finite_sampling_fields(field_name, value):
    with pytest.raises(ValueError, match=f"{field_name} must be finite"):
        GenerationConfig(**{field_name: value})


@pytest.mark.parametrize("field_name", ["fuse_decoding", "include_prompt"])
@pytest.mark.parametrize("value", [1, "true"])
def test_generation_config_rejects_non_boolean_fields(field_name, value):
    with pytest.raises(TypeError, match=f"{field_name} must be a boolean"):
        GenerationConfig(**{field_name: value})


def test_generation_config_accepts_numpy_scalar_fields():
    config = GenerationConfig(
        seed=np.int64(1),
        max_new_tokens=np.int64(2),
        top_k=np.int64(3),
        top_p=np.float32(0.9),
        min_p=np.float32(0.1),
        temperature=np.float32(0.5),
        fuse_decoding=np.bool_(True),
        include_prompt=np.bool_(False),
    )

    assert config.seed == 1
    assert config.max_new_tokens == 2
    assert config.top_k == 3
    assert config.top_p == pytest.approx(0.9)
    assert config.min_p == pytest.approx(0.1)
    assert config.temperature == pytest.approx(0.5)
    assert config.fuse_decoding is True
    assert config.include_prompt is False


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


@pytest.mark.parametrize("prompts", [123, object(), (prompt for prompt in ["hello"])])
def test_generate_tokens_rejects_non_sequence_prompts(prompts):
    tokenizer = DummyTokenizer()
    engine = DummyEngine()
    pipeline = TextGenerationPipeline(engine=engine, tokenizer=tokenizer)

    with pytest.raises(TypeError, match="prompts must be a string or a sequence of strings"):
        pipeline.generate_tokens(prompts)

    assert tokenizer.encode_calls == []
    assert engine.generate_calls == []


@pytest.mark.parametrize("prompts", [["hello", 1], ("hello", None), [b"hello"]])
def test_generate_tokens_rejects_prompt_batches_with_non_strings(prompts):
    tokenizer = DummyTokenizer()
    engine = DummyEngine()
    pipeline = TextGenerationPipeline(engine=engine, tokenizer=tokenizer)

    with pytest.raises(TypeError, match="prompts must be a string or a sequence of strings"):
        pipeline.generate_tokens(prompts)

    assert tokenizer.encode_calls == []
    assert engine.generate_calls == []


def test_generate_tokens_rejects_invalid_generation_config_before_tokenizing():
    tokenizer = DummyTokenizer()
    engine = DummyEngine()
    pipeline = TextGenerationPipeline(engine=engine, tokenizer=tokenizer)

    with pytest.raises(TypeError, match="generation_config must be a GenerationConfig"):
        pipeline.generate_tokens("hello", generation_config={"max_new_tokens": 1})

    assert tokenizer.encode_calls == []
    assert engine.generate_calls == []


def test_generate_text_rejects_invalid_generation_config_before_tokenizing():
    tokenizer = DummyTokenizer()
    engine = DummyEngine()
    pipeline = TextGenerationPipeline(engine=engine, tokenizer=tokenizer)

    with pytest.raises(TypeError, match="generation_config must be a GenerationConfig"):
        pipeline.generate_text("hello", generation_config={"max_new_tokens": 1})

    assert tokenizer.encode_calls == []
    assert tokenizer.decode_calls == []
    assert engine.generate_calls == []


def test_generate_tokens_rejects_non_mapping_tokenize_kwargs_before_tokenizing():
    tokenizer = DummyTokenizer()
    engine = DummyEngine()
    pipeline = TextGenerationPipeline(engine=engine, tokenizer=tokenizer)

    with pytest.raises(TypeError, match="tokenize_kwargs must be a mapping"):
        pipeline.generate_tokens("hello", tokenize_kwargs=["padding"])

    assert tokenizer.encode_calls == []
    assert engine.generate_calls == []


def test_generate_text_rejects_non_mapping_decode_kwargs_before_generation():
    tokenizer = DummyTokenizer()
    engine = DummyEngine()
    pipeline = TextGenerationPipeline(engine=engine, tokenizer=tokenizer)

    with pytest.raises(TypeError, match="decode_kwargs must be a mapping"):
        pipeline.generate_text("hello", decode_kwargs=["skip_special_tokens"])

    assert tokenizer.encode_calls == []
    assert tokenizer.decode_calls == []
    assert engine.generate_calls == []


@pytest.mark.parametrize(
    "encoded,exception,match",
    [
        ({"input_ids": np.ones((2,), dtype=np.int32)}, ValueError, "input_ids must be a 2D array"),
        ({"input_ids": np.ones((1, 2), dtype=np.float32)}, TypeError, "integer token ids"),
        ({"input_ids": np.ones((1, 0), dtype=np.int32)}, ValueError, "at least one token"),
        (
            {"input_ids": np.ones((1, 2), dtype=np.int32), "attention_mask": np.ones((2,), dtype=np.int32)},
            ValueError,
            "attention_mask must be a 2D array",
        ),
        (
            {"input_ids": np.ones((1, 2), dtype=np.int32), "attention_mask": np.ones((1, 2), dtype=np.float32)},
            TypeError,
            "attention_mask must be boolean or integer",
        ),
        (
            {"input_ids": np.ones((1, 2), dtype=np.int32), "attention_mask": np.ones((1, 3), dtype=np.int32)},
            ValueError,
            "attention_mask shape must match",
        ),
        (
            {"input_ids": np.ones((1, 2), dtype=np.int32), "attention_mask": np.zeros((1, 2), dtype=np.int32)},
            ValueError,
            "at least one valid token",
        ),
    ],
)
def test_generate_tokens_rejects_invalid_tokenizer_arrays_before_generation(encoded, exception, match):
    tokenizer = StaticTokenizer(encoded)
    engine = DummyEngine()
    pipeline = TextGenerationPipeline(engine=engine, tokenizer=tokenizer)

    with pytest.raises(exception, match=match):
        pipeline.generate_tokens("hello")

    assert tokenizer.encode_calls
    assert tokenizer.decode_calls == []
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
        use_tpu=np.bool_(True),
        cache_stride=128,
        tokenizer=tokenizer,
        model_kwargs={"trust_remote_code": True},
    )

    assert pipeline.tokenizer is tokenizer
    assert calls["load_model"] == ("some/model", "llama", "bfloat16", {"trust_remote_code": True})
    assert calls["engine_init"] == ("model", inference_config, {"params": "params"}, "engine-dtype", 128)
    assert calls["use_tpu"] is True


def test_from_hf_rejects_non_boolean_use_tpu_before_loading(monkeypatch):
    calls = []

    def fake_load_model_from_hf(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("load_model_from_hf should not be called for invalid use_tpu.")

    monkeypatch.setattr("jaxml.text_generation.load_model_from_hf", fake_load_model_from_hf)

    with pytest.raises(TypeError, match="use_tpu must be a boolean"):
        TextGenerationPipeline.from_hf("some/model", tokenizer=DummyTokenizer(), use_tpu=1)

    assert calls == []


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"tokenizer_kwargs": ["padding"]}, "tokenizer_kwargs must be a mapping"),
        ({"model_kwargs": ["trust_remote_code"]}, "model_kwargs must be a mapping"),
    ],
)
def test_from_hf_rejects_non_mapping_kwargs_before_loading(monkeypatch, kwargs, match):
    calls = []

    def fake_load_model_from_hf(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("load_model_from_hf should not be called for invalid kwargs.")

    monkeypatch.setattr("jaxml.text_generation.load_model_from_hf", fake_load_model_from_hf)

    with pytest.raises(TypeError, match=match):
        TextGenerationPipeline.from_hf("some/model", tokenizer=DummyTokenizer(), **kwargs)

    assert calls == []


@pytest.mark.parametrize(
    "cache_stride,exception,match",
    [
        (True, TypeError, "cache_stride must be an integer"),
        (np.bool_(True), TypeError, "cache_stride must be an integer"),
        (1.5, TypeError, "cache_stride must be an integer"),
        (0, ValueError, "cache_stride must be positive"),
        (-1, ValueError, "cache_stride must be positive"),
    ],
)
def test_from_hf_rejects_invalid_cache_stride_before_loading(monkeypatch, cache_stride, exception, match):
    calls = []

    def fake_load_model_from_hf(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("load_model_from_hf should not be called for invalid cache_stride.")

    monkeypatch.setattr("jaxml.text_generation.load_model_from_hf", fake_load_model_from_hf)

    with pytest.raises(exception, match=match):
        TextGenerationPipeline.from_hf("some/model", tokenizer=DummyTokenizer(), cache_stride=cache_stride)

    assert calls == []


def test_load_model_from_hf_dispatches_architecture(monkeypatch):
    import jaxml.hf_utils as hf_utils

    monkeypatch.setattr(hf_utils, "load_llama_from_hf", lambda *args, **kwargs: ("llama", args, kwargs))
    monkeypatch.setattr(hf_utils, "load_neox_from_hf", lambda *args, **kwargs: ("neox", args, kwargs))
    monkeypatch.setattr(hf_utils, "load_gemma_from_hf", lambda *args, **kwargs: ("gemma", args, kwargs))
    monkeypatch.setattr(hf_utils, "_infer_hf_architecture", lambda *args, **kwargs: "llama")

    assert hf_utils.load_model_from_hf("model", dtype="float16")[0] == "llama"
    assert hf_utils.load_model_from_hf("model", architecture="llama", dtype="float16")[2]["dtype"] == "float16"
    assert hf_utils.load_model_from_hf("model", architecture="GPT-NeoX")[0] == "neox"
    assert hf_utils.load_model_from_hf("model", architecture="Gemma-3")[0] == "gemma"

    with pytest.raises(ValueError, match="Unsupported"):
        hf_utils.load_model_from_hf("model", architecture="unknown")
    with pytest.raises(TypeError, match="architecture must be a string"):
        hf_utils.load_model_from_hf("model", architecture=None)
