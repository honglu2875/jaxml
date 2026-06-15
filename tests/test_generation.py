import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct
from flax.core import FrozenDict

from jaxml.inference_engine.engine import Engine, InferenceConfig
from jaxml.outputs import GenerationOutput

pytestmark = pytest.mark.critical


class FakeSharding:
    def __init__(self, spec):
        self.spec = spec


class FakeDevice:
    def __init__(self, platform):
        self.platform = platform


class FakeArrayWithDeviceProperty:
    shape = (1,)
    dtype = jnp.float32
    device = FakeDevice("tpu")


class FakeArrayWithDeviceMethod:
    shape = (1,)
    dtype = jnp.float32

    def device(self):
        return FakeDevice("gpu")


class FakeShard:
    def __init__(self, platform):
        self.device = FakeDevice(platform)


class FakeShardedArray:
    shape = (1,)
    dtype = jnp.float32
    addressable_shards = (FakeShard("tpu"),)


class InitShouldNotRunModel:
    def init(self, *args, **kwargs):
        raise AssertionError("model.init should not be called before rejecting invalid explicit weights.")


@struct.dataclass
class FakeModelConfig:
    num_layers: object


@struct.dataclass
class FakeCacheModel:
    config: object


def _prefill_fake_caches(kv_caches, batch_size=1):
    k = jnp.ones((batch_size, 1, 1, 2), dtype=jnp.float32)
    return tuple(
        cache if cache.k is not None else cache.update(k, k, mask=jnp.ones((batch_size, 1), dtype=bool)) for cache in kv_caches
    )


@pytest.mark.parametrize(
    "max_new_tokens,include_prompt,cache_stride,fuse_decoding,expected_length",
    [
        (10, True, 256, False, 20),
        (10, False, 8, False, 10),
        (3, True, 8, True, 13),
        (1, False, 256, False, 1),
        (0, True, 256, False, 10),
        (0, False, 256, False, 0),
    ],
)
def test_engine_generate_token_count_contract(
    llama_model_with_head,
    max_new_tokens,
    include_prompt,
    cache_stride,
    fuse_decoding,
    expected_length,
):
    bs, seq_len = 4, 10
    with jax.default_device(jax.devices("cpu")[0]):
        model, params = llama_model_with_head
        key = jax.random.PRNGKey(0)
        input_ids = jax.random.randint(key, (bs, seq_len), 0, model.config.vocab_size - 1, dtype=jnp.int32)

        engine = Engine(model, InferenceConfig(), params, cache_stride=cache_stride)
        engine.init_params(use_tpu=False)
        output = engine.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            include_prompt=include_prompt,
            fuse_decoding=fuse_decoding,
        )

    assert output.shape == (bs, expected_length)
    if include_prompt:
        assert np.array_equal(np.array(output[:, :seq_len]), np.array(input_ids))


def test_engine_generate_rejects_negative_max_new_tokens(llama_model_with_head):
    with jax.default_device(jax.devices("cpu")[0]):
        model, params = llama_model_with_head
        engine = Engine(model, InferenceConfig(), params)
        input_ids = jnp.ones((1, 4), dtype=jnp.int32)

        with pytest.raises(ValueError, match="max_new_tokens"):
            engine.generate(input_ids, max_new_tokens=-1)


def test_engine_generate_rejects_negative_max_new_tokens_before_prompt_or_sampling_validation(llama_model_with_head):
    with jax.default_device(jax.devices("cpu")[0]):
        model, params = llama_model_with_head
        engine = Engine(model, InferenceConfig(), params)

        with pytest.raises(ValueError, match="max_new_tokens must be non-negative"):
            engine.generate(jnp.ones((1, 2, 3), dtype=jnp.float32), max_new_tokens=-1, temperature=np.nan)


@pytest.mark.parametrize("max_new_tokens", [True, 1.5])
def test_engine_generate_rejects_non_integer_max_new_tokens(llama_model_with_head, max_new_tokens):
    with jax.default_device(jax.devices("cpu")[0]):
        model, params = llama_model_with_head
        engine = Engine(model, InferenceConfig(), params)
        input_ids = jnp.ones((1, 4), dtype=jnp.int32)

        with pytest.raises(TypeError, match="max_new_tokens must be an integer"):
            engine.generate(input_ids, max_new_tokens=max_new_tokens)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"seed": True}, "seed must be an integer"),
        ({"seed": np.bool_(True)}, "seed must be an integer"),
        ({"seed": 1.5}, "seed must be an integer"),
    ],
)
def test_engine_generate_rejects_non_integer_seed(llama_model_with_head, kwargs, match):
    with jax.default_device(jax.devices("cpu")[0]):
        model, params = llama_model_with_head
        engine = Engine(model, InferenceConfig(), params)
        input_ids = jnp.ones((1, 4), dtype=jnp.int32)

        with pytest.raises(TypeError, match=match):
            engine.generate(input_ids, max_new_tokens=0, **kwargs)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"include_prompt": 1}, "include_prompt must be a boolean"),
        ({"fuse_decoding": 1}, "fuse_decoding must be a boolean"),
    ],
)
def test_engine_generate_rejects_non_boolean_control_flags(llama_model_with_head, kwargs, match):
    with jax.default_device(jax.devices("cpu")[0]):
        model, params = llama_model_with_head
        engine = Engine(model, InferenceConfig(), params)
        input_ids = jnp.ones((1, 4), dtype=jnp.int32)

        with pytest.raises(TypeError, match=match):
            engine.generate(input_ids, max_new_tokens=0, **kwargs)


def test_engine_generate_accepts_numpy_scalar_control_arguments(llama_model_with_head):
    with jax.default_device(jax.devices("cpu")[0]):
        model, params = llama_model_with_head
        engine = Engine(model, InferenceConfig(), params)
        input_ids = jnp.ones((1, 4), dtype=jnp.int32)

        output = engine.generate(
            input_ids,
            seed=np.int64(0),
            max_new_tokens=np.int64(0),
            include_prompt=np.bool_(False),
            fuse_decoding=np.bool_(False),
        )

    assert output.shape == (1, 0)


def test_engine_init_params_rejects_non_dict_weights(llama_model_with_head):
    with jax.default_device(jax.devices("cpu")[0]):
        model, params = llama_model_with_head
        engine = Engine(model, InferenceConfig(), params)

        with pytest.raises(TypeError, match="weights must be a mapping"):
            engine.init_params(weights=("params", {}), use_tpu=False)


def test_engine_init_params_rejects_weights_without_params(llama_model_with_head):
    with jax.default_device(jax.devices("cpu")[0]):
        model, params = llama_model_with_head
        engine = Engine(model, InferenceConfig(), params)

        with pytest.raises(ValueError, match="'params'"):
            engine.init_params(weights={"cache": {}}, use_tpu=False)


def test_engine_init_params_accepts_frozen_dict_weights(llama_model_with_head):
    with jax.default_device(jax.devices("cpu")[0]):
        model, params = llama_model_with_head
        engine = Engine(model, InferenceConfig(), {"params": {}})
        weights = FrozenDict(params)

        engine.init_params(weights=weights, use_tpu=False)

    assert "params" in engine.params


def test_engine_init_params_rejects_frozen_dict_without_params(llama_model_with_head):
    with jax.default_device(jax.devices("cpu")[0]):
        model, params = llama_model_with_head
        engine = Engine(model, InferenceConfig(), params)

        with pytest.raises(ValueError, match="'params'"):
            engine.init_params(weights=FrozenDict({"cache": {}}), use_tpu=False)


def test_engine_init_params_rejects_empty_weights_mapping(llama_model_with_head):
    with jax.default_device(jax.devices("cpu")[0]):
        model, params = llama_model_with_head
        engine = Engine(model, InferenceConfig(), params)

        with pytest.raises(ValueError, match="'params'"):
            engine.init_params(weights={}, use_tpu=False)


@pytest.mark.parametrize(
    "weights,exception,match",
    [
        (("params", {}), TypeError, "weights must be a mapping"),
        ({}, ValueError, "'params'"),
    ],
)
def test_engine_init_params_rejects_invalid_explicit_weights_before_model_init(weights, exception, match):
    engine = Engine(InitShouldNotRunModel(), InferenceConfig(), {"params": {}})

    with pytest.raises(exception, match=match):
        engine.init_params(weights=weights, use_tpu=False)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"use_tpu": 1}, "use_tpu must be a boolean"),
        ({"reinit_weight": 1}, "reinit_weight must be a boolean"),
    ],
)
def test_engine_init_params_rejects_non_boolean_control_flags(llama_model_with_head, kwargs, match):
    with jax.default_device(jax.devices("cpu")[0]):
        model, params = llama_model_with_head
        engine = Engine(model, InferenceConfig(), params)

        with pytest.raises(TypeError, match=match):
            engine.init_params(**kwargs)


def test_engine_init_params_accepts_numpy_boolean_control_flags(llama_model_with_head):
    with jax.default_device(jax.devices("cpu")[0]):
        model, params = llama_model_with_head
        engine = Engine(model, InferenceConfig(), params)

        engine.init_params(use_tpu=np.bool_(False), reinit_weight=np.bool_(False))

    assert "params" in engine.params


def test_engine_prepare_input_converts_array_like_leaves_and_dtype(llama_model_with_head):
    model, params = llama_model_with_head
    engine = Engine(model, InferenceConfig(), params)

    prepared = engine.prepare_input({"input_ids": np.array([[1, 2, 3]], dtype=np.int32)}, dtype=jnp.float32)

    assert prepared["input_ids"].shape == (1, 3)
    assert prepared["input_ids"].dtype == jnp.float32


def test_engine_prepare_input_uses_configured_device_count(monkeypatch, llama_model_with_head):
    captured = {}
    available_devices = tuple(jax.devices())

    def fake_create_device_mesh(mesh_shape, devices=None):
        captured["mesh_shape"] = mesh_shape
        captured["devices"] = tuple(devices)
        return np.asarray(devices).reshape(mesh_shape)

    monkeypatch.setattr("jaxml.inference_engine.engine.mesh_utils.create_device_mesh", fake_create_device_mesh)

    model, params = llama_model_with_head
    engine = Engine(model, InferenceConfig(), params)

    engine.prepare_input({"input_ids": np.array([[1, 2, 3]], dtype=np.int32)}, dtype=jnp.float32)

    assert captured["mesh_shape"] == (1, 1)
    assert captured["devices"] == available_devices[:1]


def test_engine_prepare_input_accepts_batch_divisible_by_dp_size(monkeypatch, llama_model_with_head):
    available_devices = tuple(jax.devices())

    def fake_devices():
        return available_devices * 2

    def fake_create_device_mesh(mesh_shape, devices=None):
        return np.asarray(devices).reshape(mesh_shape)

    monkeypatch.setattr("jaxml.inference_engine.engine.jax.devices", fake_devices)
    monkeypatch.setattr("jaxml.inference_engine.engine.jax.device_count", lambda: 2)
    monkeypatch.setattr("jaxml.inference_engine.engine.jax.device_put", lambda inputs, sharding: inputs)
    monkeypatch.setattr("jaxml.inference_engine.engine.mesh_utils.create_device_mesh", fake_create_device_mesh)

    model, params = llama_model_with_head
    engine = Engine(model, InferenceConfig(dp_size=2), params)

    prepared = engine.prepare_input({"input_ids": np.array([[1, 2], [3, 4]], dtype=np.int32)}, dtype=jnp.float32)

    assert prepared["input_ids"].shape == (2, 2)
    assert prepared["input_ids"].dtype == jnp.float32


def test_engine_prepare_input_rejects_batch_not_divisible_by_dp_size(monkeypatch, llama_model_with_head):
    calls = []
    available_devices = tuple(jax.devices())

    def fake_devices():
        return available_devices * 2

    def fake_device_put(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("device_put should not be called for an invalid data-parallel batch size.")

    monkeypatch.setattr("jaxml.inference_engine.engine.jax.devices", fake_devices)
    monkeypatch.setattr("jaxml.inference_engine.engine.jax.device_count", lambda: 2)
    monkeypatch.setattr(jax, "device_put", fake_device_put)

    model, params = llama_model_with_head
    engine = Engine(model, InferenceConfig(dp_size=2), params)

    with pytest.raises(ValueError, match="divisible by dp_size=2"):
        engine.prepare_input({"input_ids": np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)}, dtype=jnp.float32)

    assert calls == []


def test_engine_prepare_input_rejects_empty_pytree_before_device_put(monkeypatch, llama_model_with_head):
    calls = []

    def fake_device_put(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("device_put should not be called for an empty input pytree.")

    monkeypatch.setattr(jax, "device_put", fake_device_put)

    model, params = llama_model_with_head
    engine = Engine(model, InferenceConfig(), params)

    with pytest.raises(ValueError, match="at least one array leaf"):
        engine.prepare_input({})

    assert calls == []


def test_engine_prepare_input_rejects_empty_leaf_batch_before_device_put(monkeypatch, llama_model_with_head):
    calls = []

    def fake_device_put(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("device_put should not be called for an empty input batch.")

    monkeypatch.setattr(jax, "device_put", fake_device_put)

    model, params = llama_model_with_head
    engine = Engine(model, InferenceConfig(), params)

    with pytest.raises(ValueError, match="at least one batch row"):
        engine.prepare_input({"input_ids": np.ones((0, 2), dtype=np.int32)})

    assert calls == []


def test_engine_prepare_input_rejects_empty_leaf_tokens_before_device_put(monkeypatch, llama_model_with_head):
    calls = []

    def fake_device_put(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("device_put should not be called for an empty input token axis.")

    monkeypatch.setattr(jax, "device_put", fake_device_put)

    model, params = llama_model_with_head
    engine = Engine(model, InferenceConfig(), params)

    with pytest.raises(ValueError, match="at least one token"):
        engine.prepare_input({"input_ids": np.ones((1, 0), dtype=np.int32)})

    assert calls == []


def test_engine_prepare_input_rejects_mismatched_leaf_batch_sizes_before_device_put(
    monkeypatch,
    llama_model_with_head,
):
    calls = []

    def fake_device_put(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("device_put should not be called for mismatched batch sizes.")

    monkeypatch.setattr(jax, "device_put", fake_device_put)

    model, params = llama_model_with_head
    engine = Engine(model, InferenceConfig(), params)

    with pytest.raises(ValueError, match="same batch size"):
        engine.prepare_input(
            {
                "input_ids": np.ones((2, 3), dtype=np.int32),
                "attention_mask": np.ones((1, 3), dtype=np.int32),
            }
        )

    assert calls == []


@pytest.mark.parametrize(
    "inputs,exception,match",
    [
        (object(), TypeError, "array-like"),
        (jnp.ones((3,), dtype=jnp.int32), ValueError, "2D arrays"),
        (jnp.ones((1, 2, 3), dtype=jnp.int32), ValueError, "2D arrays"),
    ],
)
def test_engine_prepare_input_rejects_invalid_leaves(llama_model_with_head, inputs, exception, match):
    model, params = llama_model_with_head
    engine = Engine(model, InferenceConfig(), params)

    with pytest.raises(exception, match=match):
        engine.prepare_input(inputs)


def test_engine_prepare_input_rejects_invalid_dtype_before_device_put(monkeypatch, llama_model_with_head):
    calls = []

    def fake_device_put(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("device_put should not be called for invalid dtype.")

    monkeypatch.setattr(jax, "device_put", fake_device_put)

    model, params = llama_model_with_head
    engine = Engine(model, InferenceConfig(), params)

    with pytest.raises(TypeError, match="valid JAX dtype"):
        engine.prepare_input(jnp.ones((1, 2), dtype=jnp.int32), dtype="not-a-dtype")

    assert calls == []


def test_engine_shard_params_rejects_object_without_spec(llama_model_with_head):
    model, params = llama_model_with_head
    engine = Engine(model, InferenceConfig(), params)

    with pytest.raises(TypeError, match="spec attribute"):
        engine._shard_params(jnp.ones((2, 3)), object())


def test_engine_shard_params_rejects_incompatible_spec_rank(llama_model_with_head):
    model, params = llama_model_with_head
    engine = Engine(model, InferenceConfig(), params)

    with pytest.raises(ValueError, match="does not match"):
        engine._shard_params(jnp.ones((2, 3, 4)), FakeSharding((None, None)))


def test_engine_generate_forwards_normalized_sampling_values(monkeypatch, llama_model_with_head):
    calls = []

    def fake_generate(
        params,
        eval_fn,
        prompt_tokens,
        attention_mask,
        kv_caches,
        call_hash,
        sampling_method,
        **kwargs,
    ):
        del params, eval_fn, prompt_tokens, attention_mask, call_hash, sampling_method
        calls.append(kwargs)
        return GenerationOutput(tokens=jnp.array([[1]], dtype=jnp.int32), kv_caches=kv_caches, rng=kwargs["rng"])

    monkeypatch.setattr("jaxml._generate.generate", fake_generate)

    with jax.default_device(jax.devices("cpu")[0]):
        model, params = llama_model_with_head
        engine = Engine(model, InferenceConfig(), params)
        engine.generate(
            jnp.ones((1, 4), dtype=jnp.int32),
            max_new_tokens=1,
            top_k=-3,
            top_p=1.5,
            min_p=-0.5,
            temperature=-1.0,
            include_prompt=False,
        )

    assert calls
    assert calls[0]["top_k"] == 0
    assert calls[0]["top_p"] == 1.0
    assert calls[0]["min_p"] == 0.0
    assert calls[0]["temperature"] == 0.0


def test_engine_generate_hash_includes_prompt_token_dtype(monkeypatch, llama_model_with_head):
    call_hashes = []

    def fake_generate(
        params,
        eval_fn,
        prompt_tokens,
        attention_mask,
        kv_caches,
        call_hash,
        sampling_method,
        **kwargs,
    ):
        del params, eval_fn, prompt_tokens, attention_mask, sampling_method
        call_hashes.append(call_hash)
        return GenerationOutput(tokens=jnp.array([[1]], dtype=jnp.int32), kv_caches=kv_caches, rng=kwargs["rng"])

    monkeypatch.setattr("jaxml._generate.generate", fake_generate)

    with jax.default_device(jax.devices("cpu")[0]):
        model, params = llama_model_with_head
        engine = Engine(model, InferenceConfig(), params)
        for dtype in (jnp.int32, jnp.uint32):
            engine.generate(
                jnp.ones((1, 4), dtype=dtype),
                max_new_tokens=1,
                temperature=0.0,
                include_prompt=False,
            )

    assert len(call_hashes) == 2
    assert call_hashes[0] != call_hashes[1]


def test_engine_generate_hash_includes_kv_cache_signature(monkeypatch, llama_model_with_head):
    call_hashes = []

    def fake_generate(
        params,
        eval_fn,
        prompt_tokens,
        attention_mask,
        kv_caches,
        call_hash,
        sampling_method,
        **kwargs,
    ):
        del params, eval_fn, prompt_tokens, attention_mask, sampling_method
        call_hashes.append(call_hash)
        return GenerationOutput(tokens=jnp.array([[1]], dtype=jnp.int32), kv_caches=kv_caches, rng=kwargs["rng"])

    monkeypatch.setattr("jaxml._generate.generate", fake_generate)

    with jax.default_device(jax.devices("cpu")[0]):
        model, params = llama_model_with_head
        for dtype in (jnp.float32, jnp.bfloat16):
            engine = Engine(model, InferenceConfig(), params, dtype=dtype)
            engine.generate(
                jnp.ones((1, 4), dtype=jnp.int32),
                max_new_tokens=1,
                temperature=0.0,
                include_prompt=False,
            )

    assert len(call_hashes) == 2
    assert call_hashes[0] != call_hashes[1]


def test_engine_platform_signature_accepts_device_property():
    signature = Engine._platform_signature(FakeArrayWithDeviceProperty())

    assert "'tpu'" in signature


def test_engine_platform_signature_accepts_device_method():
    signature = Engine._platform_signature(FakeArrayWithDeviceMethod())

    assert "'gpu'" in signature
    assert "method" not in signature


def test_engine_platform_signature_includes_addressable_shards():
    signature = Engine._platform_signature(FakeShardedArray())

    assert "'tpu'" in signature


def test_engine_generate_continues_rng_across_cache_resize_chunks(monkeypatch, llama_model_with_head):
    calls = []

    def fake_generate(
        params,
        eval_fn,
        prompt_tokens,
        attention_mask,
        kv_caches,
        call_hash,
        sampling_method,
        **kwargs,
    ):
        del params, eval_fn, prompt_tokens, attention_mask, call_hash, sampling_method
        calls.append(kwargs)
        next_rng = kwargs["rng"]
        decode_steps = kwargs["max_new_tokens"] if kwargs["skip_prefill"] else kwargs["max_new_tokens"] - 1
        for _ in range(decode_steps):
            next_rng, _ = jax.random.split(next_rng)
        tokens = jnp.full((1, kwargs["max_new_tokens"]), len(calls), dtype=jnp.int32)
        return GenerationOutput(tokens=tokens, kv_caches=_prefill_fake_caches(kv_caches), rng=next_rng)

    monkeypatch.setattr("jaxml._generate.generate", fake_generate)

    with jax.default_device(jax.devices("cpu")[0]):
        model, params = llama_model_with_head
        engine = Engine(model, InferenceConfig(), params, cache_stride=4)
        output = engine.generate(
            jnp.ones((1, 4), dtype=jnp.int32),
            seed=7,
            max_new_tokens=6,
            temperature=0.0,
            include_prompt=False,
        )

    expected_second_rng = jax.random.PRNGKey(7)
    for _ in range(3):
        expected_second_rng, _ = jax.random.split(expected_second_rng)

    assert output.shape == (1, 6)
    assert len(calls) == 2
    assert calls[0]["max_new_tokens"] == 4
    assert calls[0]["skip_prefill"] is False
    assert calls[1]["max_new_tokens"] == 2
    assert calls[1]["skip_prefill"] is True
    assert np.array_equal(np.array(calls[1]["rng"]), np.array(expected_second_rng))


def test_engine_generate_rejects_uninitialized_internal_caches_before_continuation(monkeypatch, llama_model_with_head):
    def fake_generate(
        params,
        eval_fn,
        prompt_tokens,
        attention_mask,
        kv_caches,
        call_hash,
        sampling_method,
        **kwargs,
    ):
        del params, eval_fn, prompt_tokens, attention_mask, call_hash, sampling_method
        tokens = jnp.ones((1, kwargs["max_new_tokens"]), dtype=jnp.int32)
        return GenerationOutput(tokens=tokens, kv_caches=kv_caches, rng=kwargs["rng"])

    monkeypatch.setattr("jaxml._generate.generate", fake_generate)

    with jax.default_device(jax.devices("cpu")[0]):
        model, params = llama_model_with_head
        engine = Engine(model, InferenceConfig(), params, cache_stride=4)

        with pytest.raises(ValueError, match="prefilled KV caches"):
            engine.generate(
                jnp.ones((1, 4), dtype=jnp.int32),
                max_new_tokens=6,
                temperature=0.0,
                include_prompt=False,
            )


def test_engine_generate_only_passes_attention_mask_to_prefill_chunk(monkeypatch, llama_model_with_head):
    calls = []

    def fake_generate(
        params,
        eval_fn,
        prompt_tokens,
        attention_mask,
        kv_caches,
        call_hash,
        sampling_method,
        **kwargs,
    ):
        del params, eval_fn, call_hash, sampling_method
        calls.append((prompt_tokens, attention_mask, kwargs))
        tokens = jnp.full((prompt_tokens.shape[0], kwargs["max_new_tokens"]), len(calls), dtype=jnp.int32)
        return GenerationOutput(
            tokens=tokens,
            kv_caches=_prefill_fake_caches(kv_caches, batch_size=prompt_tokens.shape[0]),
            rng=kwargs["rng"],
        )

    monkeypatch.setattr("jaxml._generate.generate", fake_generate)

    with jax.default_device(jax.devices("cpu")[0]):
        model, params = llama_model_with_head
        engine = Engine(model, InferenceConfig(), params, cache_stride=4)
        input_ids = jnp.ones((2, 4), dtype=jnp.int32)
        attention_mask = jnp.array([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=jnp.int32)

        output = engine.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=6,
            temperature=0.0,
            include_prompt=False,
        )

    assert output.shape == (2, 6)
    assert len(calls) == 2
    assert calls[0][0].shape == (2, 4)
    assert np.array_equal(np.array(calls[0][1]), np.array(attention_mask, dtype=bool))
    assert calls[0][2]["skip_prefill"] is False
    assert calls[1][0].shape == (2, 1)
    assert calls[1][1] is None
    assert calls[1][2]["skip_prefill"] is True


@pytest.mark.parametrize(
    "step_output_factory,exception,match",
    [
        (lambda kwargs, kv_caches: object(), TypeError, "GenerationOutput"),
        (
            lambda kwargs, kv_caches: GenerationOutput(
                tokens=jnp.ones((1,), dtype=jnp.int32),
                kv_caches=kv_caches,
                rng=kwargs["rng"],
            ),
            ValueError,
            "2D array",
        ),
        (
            lambda kwargs, kv_caches: GenerationOutput(
                tokens=jnp.ones((2, 1), dtype=jnp.int32),
                kv_caches=kv_caches,
                rng=kwargs["rng"],
            ),
            ValueError,
            "batch size",
        ),
        (
            lambda kwargs, kv_caches: GenerationOutput(
                tokens=jnp.ones((1, 0), dtype=jnp.int32),
                kv_caches=kv_caches,
                rng=kwargs["rng"],
            ),
            ValueError,
            "at least one token",
        ),
        (
            lambda kwargs, kv_caches: GenerationOutput(
                tokens=jnp.ones((1, 2), dtype=jnp.int32),
                kv_caches=kv_caches,
                rng=kwargs["rng"],
            ),
            ValueError,
            "step limited",
        ),
        (
            lambda kwargs, kv_caches: GenerationOutput(
                tokens=jnp.ones((1, 1), dtype=jnp.float32),
                kv_caches=kv_caches,
                rng=kwargs["rng"],
            ),
            TypeError,
            "integer token ids",
        ),
        (
            lambda kwargs, kv_caches: GenerationOutput(
                tokens=jnp.array([[-1]], dtype=jnp.int32),
                kv_caches=kv_caches,
                rng=kwargs["rng"],
            ),
            ValueError,
            "within",
        ),
        (
            lambda kwargs, kv_caches: GenerationOutput(
                tokens=jnp.array([[1_000_000]], dtype=jnp.int32),
                kv_caches=kv_caches,
                rng=kwargs["rng"],
            ),
            ValueError,
            "within",
        ),
        (
            lambda kwargs, kv_caches: GenerationOutput(
                tokens=jnp.ones((1, 1), dtype=jnp.int32),
                kv_caches=kv_caches,
                rng=None,
            ),
            ValueError,
            "RNG key",
        ),
        (
            lambda kwargs, kv_caches: GenerationOutput(
                tokens=jnp.ones((1, 1), dtype=jnp.int32),
                kv_caches=kv_caches,
                rng=jnp.ones((1, 2), dtype=jnp.uint32),
            ),
            ValueError,
            "shape",
        ),
        (
            lambda kwargs, kv_caches: GenerationOutput(
                tokens=jnp.ones((1, 1), dtype=jnp.int32),
                kv_caches=kv_caches,
                rng=jnp.ones((2,), dtype=jnp.int32),
            ),
            TypeError,
            "uint32 key data",
        ),
        (
            lambda kwargs, kv_caches: GenerationOutput(
                tokens=jnp.ones((1, 1), dtype=jnp.int32),
                kv_caches=kv_caches,
                rng=jnp.ones((2,), dtype=jnp.float32),
            ),
            TypeError,
            "uint32 key data",
        ),
        (
            lambda kwargs, kv_caches: GenerationOutput(
                tokens=jnp.ones((1, 1), dtype=jnp.int32),
                kv_caches=object(),
                rng=kwargs["rng"],
            ),
            TypeError,
            "kv_caches",
        ),
        (
            lambda kwargs, kv_caches: GenerationOutput(
                tokens=jnp.ones((1, 1), dtype=jnp.int32),
                kv_caches=(),
                rng=kwargs["rng"],
            ),
            ValueError,
            "KV caches",
        ),
        (
            lambda kwargs, kv_caches: GenerationOutput(
                tokens=jnp.ones((1, 1), dtype=jnp.int32),
                kv_caches=tuple(object() for _ in kv_caches),
                rng=kwargs["rng"],
            ),
            TypeError,
            "KVCache instances",
        ),
        (
            lambda kwargs, kv_caches: GenerationOutput(
                tokens=jnp.ones((1, 1), dtype=jnp.int32),
                kv_caches=(
                    kv_caches[0].replace(
                        k=jnp.ones((1, kv_caches[0].max_seq_len, 1, 1), dtype=kv_caches[0].dtype),
                    ),
                    *kv_caches[1:],
                ),
                rng=kwargs["rng"],
            ),
            ValueError,
            "Invalid internal generation KV cache",
        ),
    ],
)
def test_engine_generate_rejects_invalid_internal_step_output(
    monkeypatch,
    llama_model_with_head,
    step_output_factory,
    exception,
    match,
):
    def fake_generate(
        params,
        eval_fn,
        prompt_tokens,
        attention_mask,
        kv_caches,
        call_hash,
        sampling_method,
        **kwargs,
    ):
        del params, eval_fn, prompt_tokens, attention_mask, call_hash, sampling_method
        return step_output_factory(kwargs, kv_caches)

    monkeypatch.setattr("jaxml._generate.generate", fake_generate)

    with jax.default_device(jax.devices("cpu")[0]):
        model, params = llama_model_with_head
        engine = Engine(model, InferenceConfig(), params)

        with pytest.raises(exception, match=match):
            engine.generate(jnp.ones((1, 4), dtype=jnp.int32), max_new_tokens=1, include_prompt=False)


@pytest.mark.parametrize(
    "kwargs,exception,match",
    [
        ({"top_p": True}, TypeError, "top_p must be a real number"),
        ({"min_p": "0.1"}, TypeError, "min_p must be a real number"),
        ({"temperature": np.nan}, ValueError, "temp must be finite"),
    ],
)
def test_engine_generate_rejects_invalid_sampling_values(llama_model_with_head, kwargs, exception, match):
    with jax.default_device(jax.devices("cpu")[0]):
        model, params = llama_model_with_head
        engine = Engine(model, InferenceConfig(), params)
        input_ids = jnp.ones((1, 4), dtype=jnp.int32)

        with pytest.raises(exception, match=match):
            engine.generate(input_ids, max_new_tokens=0, **kwargs)


def test_engine_generate_accepts_unbatched_prompt_tokens(llama_model_with_head):
    with jax.default_device(jax.devices("cpu")[0]):
        model, params = llama_model_with_head
        engine = Engine(model, InferenceConfig(), params)
        input_ids = jnp.arange(4, dtype=jnp.int32)

        output = engine.generate(input_ids, max_new_tokens=0, include_prompt=True)

    assert output.shape == (1, 4)
    assert np.array_equal(np.array(output[0]), np.array(input_ids))


def test_engine_generate_accepts_unbatched_attention_mask(llama_model_with_head):
    with jax.default_device(jax.devices("cpu")[0]):
        model, params = llama_model_with_head
        engine = Engine(model, InferenceConfig(), params)
        input_ids = jnp.arange(4, dtype=jnp.int32)
        attention_mask = jnp.ones((4,), dtype=bool)

        output = engine.generate(input_ids, attention_mask=attention_mask, max_new_tokens=0, include_prompt=True)

    assert output.shape == (1, 4)


def test_prepare_generation_inputs_canonicalizes_integer_attention_mask():
    input_ids = jnp.arange(4, dtype=jnp.int32)
    attention_mask = jnp.array([1, 1, 0, 0], dtype=jnp.int32)

    prepared_ids, prepared_mask = Engine._prepare_generation_inputs(input_ids, attention_mask)

    assert prepared_ids.shape == (1, 4)
    assert prepared_mask.dtype == jnp.bool_
    assert np.array_equal(np.array(prepared_mask), np.array([[True, True, False, False]]))


def test_prepare_generation_inputs_accepts_traced_attention_mask():
    input_ids = jnp.arange(4, dtype=jnp.int32)

    @jax.jit
    def prepare(attention_mask):
        _, prepared_mask = Engine._prepare_generation_inputs(input_ids, attention_mask)
        return prepared_mask

    prepared_mask = prepare(jnp.array([1, 1, 0, 0], dtype=jnp.int32))

    assert prepared_mask.dtype == jnp.bool_
    assert np.array_equal(np.array(prepared_mask), np.array([[True, True, False, False]]))


@pytest.mark.parametrize(
    "input_ids,match",
    [
        (jnp.ones((1, 2, 3), dtype=jnp.int32), "prompt_tokens must be a 1D or 2D array"),
        (jnp.ones((0, 1), dtype=jnp.int32), "at least one batch row"),
        (jnp.ones((1, 0), dtype=jnp.int32), "at least one token"),
    ],
)
def test_engine_generate_rejects_invalid_prompt_tokens(llama_model_with_head, input_ids, match):
    with jax.default_device(jax.devices("cpu")[0]):
        model, params = llama_model_with_head
        engine = Engine(model, InferenceConfig(), params)

        with pytest.raises(ValueError, match=match):
            engine.generate(input_ids, max_new_tokens=0)


def test_engine_generate_rejects_non_integer_prompt_tokens(llama_model_with_head):
    with jax.default_device(jax.devices("cpu")[0]):
        model, params = llama_model_with_head
        engine = Engine(model, InferenceConfig(), params)

        with pytest.raises(TypeError, match="integer token ids"):
            engine.generate(jnp.ones((1, 4), dtype=jnp.float32), max_new_tokens=0)


@pytest.mark.parametrize("token_id", [-1, "vocab_size"])
def test_engine_generate_rejects_prompt_tokens_outside_model_vocab(llama_model_with_head, token_id):
    with jax.default_device(jax.devices("cpu")[0]):
        model, params = llama_model_with_head
        engine = Engine(model, InferenceConfig(), params)
        if token_id == "vocab_size":
            token_id = model.config.vocab_size

        with pytest.raises(ValueError, match=r"prompt_tokens token ids must be within"):
            engine.generate(jnp.array([[token_id]], dtype=jnp.int32), max_new_tokens=0)


def test_engine_generate_rejects_mismatched_attention_mask(llama_model_with_head):
    with jax.default_device(jax.devices("cpu")[0]):
        model, params = llama_model_with_head
        engine = Engine(model, InferenceConfig(), params)
        input_ids = jnp.ones((2, 4), dtype=jnp.int32)
        attention_mask = jnp.ones((1, 4), dtype=bool)

        with pytest.raises(ValueError, match="attention_mask shape must match"):
            engine.generate(input_ids, attention_mask=attention_mask, max_new_tokens=0)


def test_engine_generate_rejects_non_boolean_or_integer_attention_mask(llama_model_with_head):
    with jax.default_device(jax.devices("cpu")[0]):
        model, params = llama_model_with_head
        engine = Engine(model, InferenceConfig(), params)
        input_ids = jnp.ones((1, 4), dtype=jnp.int32)
        attention_mask = jnp.ones((1, 4), dtype=jnp.float32)

        with pytest.raises(TypeError, match="attention_mask must be boolean or integer"):
            engine.generate(input_ids, attention_mask=attention_mask, max_new_tokens=0)


@pytest.mark.parametrize(
    "input_ids,attention_mask",
    [
        (jnp.ones((2, 4), dtype=jnp.int32), jnp.array([[True, False, False, False], [False, False, False, False]])),
        (jnp.ones((4,), dtype=jnp.int32), jnp.zeros((4,), dtype=bool)),
    ],
)
def test_engine_generate_rejects_attention_mask_without_valid_tokens(llama_model_with_head, input_ids, attention_mask):
    with jax.default_device(jax.devices("cpu")[0]):
        model, params = llama_model_with_head
        engine = Engine(model, InferenceConfig(), params)

        with pytest.raises(ValueError, match="at least one valid token"):
            engine.generate(input_ids, attention_mask=attention_mask, max_new_tokens=0)


def test_engine_init_cache_creates_one_cache_per_model_layer():
    engine = Engine(FakeCacheModel(FakeModelConfig(num_layers=np.int64(2))), InferenceConfig(), {"params": {}})

    caches = engine.init_cache(max_seq_len=np.int64(8))

    assert len(caches) == 2
    assert [cache.max_seq_len for cache in caches] == [8, 8]


@pytest.mark.parametrize("max_seq_len", [True, 1.5])
def test_engine_init_cache_rejects_non_integer_max_seq_len(max_seq_len):
    engine = Engine(FakeCacheModel(FakeModelConfig(num_layers=1)), InferenceConfig(), {"params": {}})

    with pytest.raises(TypeError, match="max_seq_len must be an integer"):
        engine.init_cache(max_seq_len=max_seq_len)


@pytest.mark.parametrize("max_seq_len", [0, -1])
def test_engine_init_cache_rejects_non_positive_max_seq_len(max_seq_len):
    engine = Engine(FakeCacheModel(FakeModelConfig(num_layers=1)), InferenceConfig(), {"params": {}})

    with pytest.raises(ValueError, match="max_seq_len must be positive"):
        engine.init_cache(max_seq_len=max_seq_len)


def test_engine_init_cache_rejects_model_without_num_layers():
    engine = Engine(FakeCacheModel(config=object()), InferenceConfig(), {"params": {}})

    with pytest.raises(TypeError, match="config.num_layers"):
        engine.init_cache(max_seq_len=8)


@pytest.mark.parametrize("num_layers", [True, 1.5])
def test_engine_init_cache_rejects_non_integer_model_num_layers(num_layers):
    engine = Engine(FakeCacheModel(FakeModelConfig(num_layers=num_layers)), InferenceConfig(), {"params": {}})

    with pytest.raises(TypeError, match="model.config.num_layers must be an integer"):
        engine.init_cache(max_seq_len=8)


@pytest.mark.parametrize("num_layers", [0, -1])
def test_engine_init_cache_rejects_non_positive_model_num_layers(num_layers):
    engine = Engine(FakeCacheModel(FakeModelConfig(num_layers=num_layers)), InferenceConfig(), {"params": {}})

    with pytest.raises(ValueError, match="model.config.num_layers must be positive"):
        engine.init_cache(max_seq_len=8)


def test_engine_rejects_non_positive_cache_stride(llama_model_with_head):
    model, params = llama_model_with_head

    with pytest.raises(ValueError, match="cache_stride"):
        Engine(model, InferenceConfig(), params, cache_stride=0)


@pytest.mark.parametrize("cache_stride", [True, 1.5])
def test_engine_rejects_non_integer_cache_stride(llama_model_with_head, cache_stride):
    model, params = llama_model_with_head

    with pytest.raises(TypeError, match="cache_stride must be an integer"):
        Engine(model, InferenceConfig(), params, cache_stride=cache_stride)


def test_engine_accepts_numpy_integer_cache_stride(llama_model_with_head):
    model, params = llama_model_with_head

    engine = Engine(model, InferenceConfig(), params, cache_stride=np.int64(8))

    assert engine.cache_stride == 8


@pytest.mark.parametrize("dtype", [None, "not-a-dtype", object()])
def test_engine_rejects_invalid_dtype(llama_model_with_head, dtype):
    model, params = llama_model_with_head

    with pytest.raises(TypeError, match="dtype must be a valid JAX dtype"):
        Engine(model, InferenceConfig(), params, dtype=dtype)


@pytest.mark.parametrize("dtype", [jnp.float32, np.float32, "bfloat16"])
def test_engine_canonicalizes_dtype(llama_model_with_head, dtype):
    model, params = llama_model_with_head

    engine = Engine(model, InferenceConfig(), params, dtype=dtype)

    assert engine.dtype == jnp.dtype(dtype)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"tp_size": 0}, "tp_size"),
        ({"tp_size": -1}, "tp_size"),
        ({"dp_size": 0}, "dp_size"),
        ({"dp_size": -1}, "dp_size"),
    ],
)
def test_inference_config_rejects_non_positive_mesh_sizes(kwargs, match):
    with pytest.raises(ValueError, match=match):
        InferenceConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"tp_size": True}, "tp_size must be an integer"),
        ({"tp_size": 1.5}, "tp_size must be an integer"),
        ({"dp_size": True}, "dp_size must be an integer"),
        ({"dp_size": 1.5}, "dp_size must be an integer"),
    ],
)
def test_inference_config_rejects_non_integer_mesh_sizes(kwargs, match):
    with pytest.raises(TypeError, match=match):
        InferenceConfig(**kwargs)


def test_inference_config_accepts_numpy_integer_mesh_sizes():
    config = InferenceConfig(tp_size=np.int64(1), dp_size=np.int64(1))

    assert config.tp_size == 1
    assert config.dp_size == 1


@pytest.mark.parametrize("config", [None, {"tp_size": 1}, object()])
def test_engine_rejects_invalid_inference_config(llama_model_with_head, config):
    model, params = llama_model_with_head

    with pytest.raises(TypeError, match="config must be an InferenceConfig"):
        Engine(model, config, params)


def test_engine_rejects_mesh_larger_than_available_devices(llama_model_with_head):
    model, params = llama_model_with_head
    config = InferenceConfig(tp_size=jax.device_count() + 1)

    with pytest.raises(ValueError, match="requires .* devices"):
        Engine(model, config, params)
