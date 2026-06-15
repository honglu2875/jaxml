from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from jaxml.inference_engine.engine import InferenceConfig
from jaxml.text_generation import GenerationConfig, TextGenerationPipeline

pytestmark = pytest.mark.critical


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


class StaticEngine:
    def __init__(self, output):
        self.output = output
        self.generate_calls = []

    def generate(self, input_ids, attention_mask=None, **kwargs):
        self.generate_calls.append((np.array(input_ids), None if attention_mask is None else np.array(attention_mask), kwargs))
        return self.output


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


class StaticDecodeTokenizer(DummyTokenizer):
    def __init__(self, decoded):
        super().__init__()
        self.decoded = decoded

    def batch_decode(self, tokens, **kwargs):
        self.decode_calls.append((np.array(tokens), kwargs))
        return self.decoded


class TokenizerWithoutDecode:
    def __call__(self, prompts, return_tensors, **kwargs):
        del prompts, return_tensors, kwargs
        return {"input_ids": np.ones((1, 1), dtype=np.int32)}


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


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"default_tokenize_kwargs": ["padding"]}, "default_tokenize_kwargs must be a mapping"),
        ({"default_decode_kwargs": ["skip_special_tokens"]}, "default_decode_kwargs must be a mapping"),
    ],
)
def test_pipeline_rejects_non_mapping_default_kwargs(kwargs, match):
    with pytest.raises(TypeError, match=match):
        TextGenerationPipeline(engine=DummyEngine(), tokenizer=DummyTokenizer(), **kwargs)


@pytest.mark.parametrize(
    "engine,tokenizer,match",
    [
        (object(), DummyTokenizer(), "engine must provide a callable generate method"),
        (DummyEngine(), object(), "tokenizer must be callable"),
        (DummyEngine(), TokenizerWithoutDecode(), "batch_decode"),
    ],
)
def test_pipeline_rejects_invalid_components(engine, tokenizer, match):
    with pytest.raises(TypeError, match=match):
        TextGenerationPipeline(engine=engine, tokenizer=tokenizer)


def test_pipeline_copies_default_kwargs_on_construction():
    tokenize_kwargs = {"padding": True}
    decode_kwargs = {"skip_special_tokens": True}
    pipeline = TextGenerationPipeline(
        engine=DummyEngine(),
        tokenizer=DummyTokenizer(),
        default_tokenize_kwargs=tokenize_kwargs,
        default_decode_kwargs=decode_kwargs,
    )
    tokenize_kwargs["padding"] = "longest"
    decode_kwargs["skip_special_tokens"] = False

    text = pipeline.generate_text(
        "hello",
        generation_config=GenerationConfig(max_new_tokens=1, include_prompt=False),
    )

    assert text == "9"
    assert pipeline.tokenizer.encode_calls[0][2] == {"padding": True}
    assert pipeline.tokenizer.decode_calls[0][1] == {"skip_special_tokens": True}


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


@pytest.mark.parametrize(
    "kwargs,exception,match",
    [
        ({"max_new_tokens": True}, TypeError, "max_new_tokens must be an integer"),
        ({"max_new_tokens": -1}, ValueError, "max_new_tokens must be non-negative"),
        ({"top_k": -1}, ValueError, "top_k must be non-negative"),
        ({"include_prompt": 1}, TypeError, "include_prompt must be a boolean"),
        ({"unknown_option": 1}, TypeError, "unexpected keyword argument"),
    ],
)
def test_generate_tokens_rejects_invalid_generation_overrides_before_tokenizing(kwargs, exception, match):
    tokenizer = DummyTokenizer()
    engine = DummyEngine()
    pipeline = TextGenerationPipeline(engine=engine, tokenizer=tokenizer)

    with pytest.raises(exception, match=match):
        pipeline.generate_tokens("hello", **kwargs)

    assert tokenizer.encode_calls == []
    assert engine.generate_calls == []


@pytest.mark.parametrize(
    "kwargs,exception,match",
    [
        ({"seed": True}, TypeError, "seed must be an integer"),
        ({"temperature": np.nan}, ValueError, "temperature must be finite"),
        ({"fuse_decoding": 1}, TypeError, "fuse_decoding must be a boolean"),
        ({"unknown_option": 1}, TypeError, "unexpected keyword argument"),
    ],
)
def test_generate_text_rejects_invalid_generation_overrides_before_tokenizing(kwargs, exception, match):
    tokenizer = DummyTokenizer()
    engine = DummyEngine()
    pipeline = TextGenerationPipeline(engine=engine, tokenizer=tokenizer)

    with pytest.raises(exception, match=match):
        pipeline.generate_text("hello", **kwargs)

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
        ({"input_ids": np.ones((0, 2), dtype=np.int32)}, ValueError, "at least one batch row"),
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


@pytest.mark.parametrize(
    "input_ids,attention_mask,exception,match",
    [
        (np.ones((2,), dtype=np.int32), None, ValueError, "input_ids must be a 2D array"),
        (np.ones((1, 2), dtype=np.float32), None, TypeError, "integer token ids"),
        (np.ones((0, 2), dtype=np.int32), None, ValueError, "at least one batch row"),
        (np.ones((1, 0), dtype=np.int32), None, ValueError, "at least one token"),
        (np.ones((1, 2), dtype=np.int32), np.ones((2,), dtype=np.int32), ValueError, "attention_mask must be a 2D array"),
        (
            np.ones((1, 2), dtype=np.int32),
            np.ones((1, 2), dtype=np.float32),
            TypeError,
            "attention_mask must be boolean or integer",
        ),
        (np.ones((1, 2), dtype=np.int32), np.ones((1, 3), dtype=np.int32), ValueError, "attention_mask shape must match"),
        (
            np.ones((1, 2), dtype=np.int32),
            np.zeros((1, 2), dtype=np.int32),
            ValueError,
            "at least one valid token",
        ),
    ],
)
def test_generate_tokens_from_arrays_rejects_invalid_arrays_before_engine(
    input_ids,
    attention_mask,
    exception,
    match,
):
    engine = DummyEngine()
    pipeline = TextGenerationPipeline(engine=engine, tokenizer=DummyTokenizer())

    with pytest.raises(exception, match=match):
        pipeline._generate_tokens_from_arrays(
            input_ids,
            attention_mask,
            generation_config=GenerationConfig(max_new_tokens=1),
        )

    assert engine.generate_calls == []


def test_generate_tokens_from_arrays_canonicalizes_integer_attention_mask():
    engine = DummyEngine()
    pipeline = TextGenerationPipeline(engine=engine, tokenizer=DummyTokenizer())

    tokens = pipeline._generate_tokens_from_arrays(
        np.array([[1, 2]], dtype=np.int32),
        np.array([[1, 0]], dtype=np.int32),
        generation_config=GenerationConfig(max_new_tokens=1, include_prompt=False),
    )

    assert tokens.shape == (1, 1)
    assert np.array_equal(engine.generate_calls[0][1], np.array([[True, False]]))


@pytest.mark.parametrize(
    "engine_output,exception,match",
    [
        (np.ones((3,), dtype=np.int32), ValueError, "2D token array"),
        (np.ones((1, 2), dtype=np.float32), TypeError, "integer token ids"),
    ],
)
def test_generate_text_rejects_invalid_engine_tokens_before_decode(engine_output, exception, match):
    tokenizer = DummyTokenizer()
    engine = StaticEngine(engine_output)
    pipeline = TextGenerationPipeline(engine=engine, tokenizer=tokenizer)

    with pytest.raises(exception, match=match):
        pipeline.generate_text("hello")

    assert engine.generate_calls
    assert tokenizer.decode_calls == []


def test_generate_text_rejects_engine_token_batch_mismatch_before_decode():
    tokenizer = DummyTokenizer()
    engine = StaticEngine(np.ones((2, 1), dtype=np.int32))
    pipeline = TextGenerationPipeline(engine=engine, tokenizer=tokenizer)

    with pytest.raises(ValueError, match="token batch size must match input batch size"):
        pipeline.generate_text("hello")

    assert engine.generate_calls
    assert tokenizer.decode_calls == []


@pytest.mark.parametrize(
    "decoded,exception,match",
    [
        ("decoded", TypeError, "sequence of strings"),
        (["first", "second"], ValueError, "output batch size must match token batch size"),
        ([1], TypeError, "sequence of strings"),
    ],
)
def test_generate_text_rejects_invalid_decoded_output(decoded, exception, match):
    tokenizer = StaticDecodeTokenizer(decoded)
    engine = StaticEngine(np.ones((1, 1), dtype=np.int32))
    pipeline = TextGenerationPipeline(engine=engine, tokenizer=tokenizer)

    with pytest.raises(exception, match=match):
        pipeline.generate_text("hello")

    assert engine.generate_calls
    assert tokenizer.decode_calls


def test_from_hf_wires_loader_engine_and_tokenizer(monkeypatch):
    calls = {}

    class FakeEngine:
        def __init__(self, model, config, params, dtype, cache_stride):
            calls["engine_init"] = (model, config, params, dtype, cache_stride)

        def init_params(self, use_tpu):
            calls["use_tpu"] = use_tpu

        def generate(self, input_ids, attention_mask=None, **kwargs):
            raise AssertionError("generate should not be called by from_hf.")

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
        engine_dtype="float32",
        inference_config=inference_config,
        use_tpu=np.bool_(True),
        cache_stride=128,
        tokenizer=tokenizer,
        model_kwargs={"trust_remote_code": True},
    )

    assert pipeline.tokenizer is tokenizer
    assert calls["load_model"] == ("some/model", "llama", "bfloat16", {"trust_remote_code": True})
    assert calls["engine_init"] == ("model", inference_config, {"params": "params"}, jnp.dtype("float32"), 128)
    assert calls["use_tpu"] is True


@pytest.mark.parametrize(
    "name,exception,match",
    [
        ("", ValueError, "non-empty"),
        ("   ", ValueError, "non-empty"),
        (123, TypeError, "string or path-like"),
    ],
)
def test_from_hf_rejects_invalid_name_before_tokenizer_or_model_loading(monkeypatch, name, exception, match):
    calls = []

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls.append(("tokenizer", args, kwargs))
            raise AssertionError("AutoTokenizer.from_pretrained should not be called for invalid name.")

    def fake_load_model_from_hf(*args, **kwargs):
        calls.append(("model", args, kwargs))
        raise AssertionError("load_model_from_hf should not be called for invalid name.")

    import transformers

    monkeypatch.setattr(transformers, "AutoTokenizer", FakeAutoTokenizer)
    monkeypatch.setattr("jaxml.text_generation.load_model_from_hf", fake_load_model_from_hf)

    with pytest.raises(exception, match=match):
        TextGenerationPipeline.from_hf(name)

    assert calls == []


@pytest.mark.parametrize(
    "kwargs,exception,match",
    [
        ({"architecture": None}, TypeError, "architecture must be a string"),
        ({"architecture": "unknown"}, ValueError, "Unsupported Hugging Face architecture"),
        ({"model_dtype": np.float32}, TypeError, "Expected dtype"),
        ({"model_dtype": "bad"}, ValueError, "Unsupported dtype"),
        ({"engine_dtype": None}, TypeError, "engine_dtype must be a valid JAX dtype"),
        ({"engine_dtype": "not-a-dtype"}, TypeError, "engine_dtype must be a valid JAX dtype"),
    ],
)
def test_from_hf_rejects_invalid_model_selectors_before_tokenizer_or_model_loading(monkeypatch, kwargs, exception, match):
    calls = []

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls.append(("tokenizer", args, kwargs))
            raise AssertionError("AutoTokenizer.from_pretrained should not be called for invalid model selectors.")

    def fake_load_model_from_hf(*args, **kwargs):
        calls.append(("model", args, kwargs))
        raise AssertionError("load_model_from_hf should not be called for invalid model selectors.")

    import transformers

    monkeypatch.setattr(transformers, "AutoTokenizer", FakeAutoTokenizer)
    monkeypatch.setattr("jaxml.text_generation.load_model_from_hf", fake_load_model_from_hf)

    with pytest.raises(exception, match=match):
        TextGenerationPipeline.from_hf("some/model", **kwargs)

    assert calls == []


def test_from_hf_normalizes_pathlike_name_for_tokenizer_and_model_loader(monkeypatch):
    calls = {}

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            calls["tokenizer"] = (name, kwargs)
            return DummyTokenizer()

    class FakeEngine:
        def __init__(self, model, config, params, dtype, cache_stride):
            calls["engine_init"] = (model, config, params, dtype, cache_stride)

        def init_params(self, use_tpu):
            calls["use_tpu"] = use_tpu

        def generate(self, input_ids, attention_mask=None, **kwargs):
            raise AssertionError("generate should not be called by from_hf.")

    def fake_load_model_from_hf(name, architecture, dtype, **kwargs):
        calls["load_model"] = (name, architecture, dtype, kwargs)
        return "model", {"params": "params"}

    import transformers

    monkeypatch.setattr(transformers, "AutoTokenizer", FakeAutoTokenizer)
    monkeypatch.setattr("jaxml.text_generation.Engine", FakeEngine)
    monkeypatch.setattr("jaxml.text_generation.load_model_from_hf", fake_load_model_from_hf)

    pipeline = TextGenerationPipeline.from_hf(
        Path("local-model"),
        architecture="GPT-NeoX",
        tokenizer_kwargs={"local_files_only": True},
        model_kwargs={"trust_remote_code": True},
    )

    assert isinstance(pipeline.tokenizer, DummyTokenizer)
    assert calls["tokenizer"] == ("local-model", {"local_files_only": True})
    assert calls["load_model"] == ("local-model", "neox", "float32", {"trust_remote_code": True})
    assert calls["engine_init"][0] == "model"
    assert calls["engine_init"][3] == jnp.dtype("float32")
    assert calls["use_tpu"] is False


def test_from_hf_rejects_non_boolean_use_tpu_before_loading(monkeypatch):
    calls = []

    def fake_load_model_from_hf(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("load_model_from_hf should not be called for invalid use_tpu.")

    monkeypatch.setattr("jaxml.text_generation.load_model_from_hf", fake_load_model_from_hf)

    with pytest.raises(TypeError, match="use_tpu must be a boolean"):
        TextGenerationPipeline.from_hf("some/model", tokenizer=DummyTokenizer(), use_tpu=1)

    assert calls == []


@pytest.mark.parametrize("inference_config", [{"tp_size": 1}, object()])
def test_from_hf_rejects_invalid_inference_config_before_loading(monkeypatch, inference_config):
    calls = []

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls.append(("tokenizer", args, kwargs))
            raise AssertionError("AutoTokenizer.from_pretrained should not be called for invalid inference_config.")

    def fake_load_model_from_hf(*args, **kwargs):
        calls.append(("model", args, kwargs))
        raise AssertionError("load_model_from_hf should not be called for invalid inference_config.")

    import transformers

    monkeypatch.setattr(transformers, "AutoTokenizer", FakeAutoTokenizer)
    monkeypatch.setattr("jaxml.text_generation.load_model_from_hf", fake_load_model_from_hf)

    with pytest.raises(TypeError, match="config must be an InferenceConfig"):
        TextGenerationPipeline.from_hf("some/model", inference_config=inference_config)

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
