import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxml.cache import KVCache
from jaxml.config import ModelConfig
from jaxml.models._utils import (
    cached_sequence_length,
    prepare_default_attention_mask,
    prepare_model_inputs,
    prepare_position_ids,
    should_use_default_attention_mask,
)
from jaxml.models.gemma3 import GemmaModel
from jaxml.models.gpt_neox import GPTNeoXDecoder, GPTNeoXModel
from jaxml.models.llama import LlamaDecoder, LlamaModel

pytestmark = pytest.mark.critical

MODEL_FIXTURES = ["llama_model", "neox_model", "gemma_model"]


def _kv(batch=1, seq_len=2, value=1.0):
    x = jnp.full((batch, seq_len, 1, 2), value, dtype=jnp.float32)
    return x, x + 1


def _small_model_config(**overrides):
    kwargs = {
        "hidden_size": 8,
        "head_dim": 4,
        "num_heads": 2,
        "num_layers": 1,
        "intermediate_ratio": (2, 1),
        "max_position_embeddings": 8,
        "vocab_size": 16,
        "attn_scale": 4**-0.5,
        "num_kv_heads": 2,
    } | overrides
    return ModelConfig(**kwargs)


def test_prepare_model_inputs_accepts_numpy_integer_num_layers():
    input_ids, attention_mask, kv_caches = prepare_model_inputs(
        jnp.ones((1, 2), dtype=jnp.int32),
        None,
        (None,),
        num_layers=np.int64(1),
    )

    assert input_ids.shape == (1, 2)
    assert attention_mask is None
    assert kv_caches == (None,)


def test_prepare_model_inputs_accepts_negative_mask_placeholder_ids_with_vocab_size():
    input_ids, attention_mask, kv_caches = prepare_model_inputs(
        jnp.array([[-100, 1]], dtype=jnp.int32),
        None,
        None,
        num_layers=1,
        vocab_size=np.int64(4),
    )

    assert input_ids.tolist() == [[-100, 1]]
    assert attention_mask is None
    assert kv_caches is None


def test_prepare_model_inputs_accepts_masked_negative_placeholder_ids():
    input_ids, attention_mask, kv_caches = prepare_model_inputs(
        jnp.array([[-100, 1]], dtype=jnp.int32),
        jnp.array([[0, 1]], dtype=jnp.int32),
        None,
        num_layers=1,
        vocab_size=4,
    )

    assert input_ids.tolist() == [[-100, 1]]
    assert attention_mask.tolist() == [[False, True]]
    assert kv_caches is None


def test_prepare_model_inputs_rejects_valid_negative_placeholder_ids():
    with pytest.raises(ValueError, match="must not mark negative placeholder"):
        prepare_model_inputs(
            jnp.array([[-100, 1]], dtype=jnp.int32),
            jnp.array([[1, 1]], dtype=jnp.int32),
            None,
            num_layers=1,
            vocab_size=4,
        )


@pytest.mark.parametrize(
    "vocab_size,exception,match",
    [
        (True, TypeError, "vocab_size must be an integer"),
        (np.bool_(True), TypeError, "vocab_size must be an integer"),
        (1.5, TypeError, "vocab_size must be an integer"),
        (0, ValueError, "vocab_size must be positive"),
    ],
)
def test_prepare_model_inputs_rejects_invalid_vocab_size(vocab_size, exception, match):
    with pytest.raises(exception, match=match):
        prepare_model_inputs(jnp.ones((1, 2), dtype=jnp.int32), None, None, num_layers=1, vocab_size=vocab_size)


def test_prepare_model_inputs_rejects_input_ids_outside_vocab_size():
    with pytest.raises(ValueError, match="less than vocab_size=4"):
        prepare_model_inputs(jnp.array([[0, 4]], dtype=jnp.int32), None, None, num_layers=1, vocab_size=4)


def test_prepare_model_inputs_validates_kv_cache_state():
    cache = KVCache(
        k=None,
        v=None,
        max_seq_len=4,
        mask=jnp.ones((1, 4), dtype=bool),
        pos_id=None,
    )

    with pytest.raises(ValueError, match="k is empty"):
        prepare_model_inputs(jnp.ones((1, 2), dtype=jnp.int32), None, (cache,), num_layers=1)


def test_prepare_model_inputs_accepts_populated_cache_attention_mask_shape():
    k, v = _kv(seq_len=2)
    cache = KVCache.init(4).update(k, v, mask=jnp.ones((1, 2), dtype=bool))

    input_ids, attention_mask, kv_caches = prepare_model_inputs(
        jnp.ones((1, 1), dtype=jnp.int32),
        cache.mask,
        (cache,),
        num_layers=1,
    )

    assert input_ids.shape == (1, 1)
    assert attention_mask.shape == (1, 4)
    assert kv_caches == (cache,)


def test_prepare_model_inputs_rejects_token_attention_mask_for_populated_cache():
    k, v = _kv(seq_len=2)
    cache = KVCache.init(4).update(k, v, mask=jnp.ones((1, 2), dtype=bool))

    with pytest.raises(ValueError, match="populated KV cache mask shape"):
        prepare_model_inputs(
            jnp.ones((1, 1), dtype=jnp.int32),
            jnp.ones((1, 1), dtype=bool),
            (cache,),
            num_layers=1,
        )


def test_prepare_model_inputs_rejects_inconsistent_populated_cache_mask_shapes():
    k, v = _kv(seq_len=2)
    short_cache = KVCache.init(4).update(k, v, mask=jnp.ones((1, 2), dtype=bool))
    long_cache = KVCache.init(5).update(k, v, mask=jnp.ones((1, 2), dtype=bool))

    with pytest.raises(ValueError, match="share attention mask shape"):
        prepare_model_inputs(jnp.ones((1, 1), dtype=jnp.int32), None, (short_cache, long_cache), num_layers=2)


def test_prepare_model_inputs_rejects_invalid_populated_kv_cache_state():
    cache = KVCache(
        k=jnp.zeros((1, 4, 1, 2), dtype=jnp.float32),
        v=jnp.zeros((1, 4, 1, 2), dtype=jnp.float32),
        max_seq_len=4,
        mask=jnp.ones((1, 4), dtype=jnp.int32),
        pos_id=jnp.zeros((1, 1), dtype=jnp.int32),
    )

    with pytest.raises(TypeError, match="Cached mask must be boolean"):
        prepare_model_inputs(jnp.ones((1, 2), dtype=jnp.int32), None, (cache,), num_layers=1)


@pytest.mark.parametrize("num_layers", [True, np.bool_(True), 1.5])
def test_prepare_model_inputs_rejects_non_integer_num_layers(num_layers):
    with pytest.raises(TypeError, match="num_layers must be an integer"):
        prepare_model_inputs(jnp.ones((1, 2), dtype=jnp.int32), None, None, num_layers=num_layers)


@pytest.mark.parametrize("num_layers", [0, -1])
def test_prepare_model_inputs_rejects_non_positive_num_layers(num_layers):
    with pytest.raises(ValueError, match="num_layers must be positive"):
        prepare_model_inputs(jnp.ones((1, 2), dtype=jnp.int32), None, None, num_layers=num_layers)


def test_prepare_position_ids_accepts_none():
    input_ids = jnp.ones((2, 4), dtype=jnp.int32)

    assert prepare_position_ids(None, input_ids) is None


def test_prepare_position_ids_accepts_broadcast_batch_axis():
    input_ids = jnp.ones((2, 4), dtype=jnp.int32)
    position_ids = jnp.arange(4, dtype=jnp.int32)[None]

    prepared = prepare_position_ids(position_ids, input_ids)

    assert prepared.shape == (1, 4)


def test_prepare_position_ids_rejects_invalid_max_position_embeddings():
    input_ids = jnp.ones((1, 4), dtype=jnp.int32)
    position_ids = jnp.arange(4, dtype=jnp.int32)[None]

    with pytest.raises(ValueError, match="max_position_embeddings must be positive"):
        prepare_position_ids(position_ids, input_ids, max_position_embeddings=0)


def test_prepare_position_ids_rejects_positions_at_or_above_max_position_embeddings():
    input_ids = jnp.ones((1, 4), dtype=jnp.int32)
    position_ids = jnp.array([[0, 1, 2, 8]], dtype=jnp.int32)

    with pytest.raises(ValueError, match="less than max_position_embeddings=8"):
        prepare_position_ids(position_ids, input_ids, max_position_embeddings=np.int64(8))


def test_default_attention_mask_uses_any_populated_cache_entry():
    k, v = _kv(seq_len=2)
    populated_cache = KVCache.init(4).update(k, v, mask=jnp.ones((1, 2), dtype=bool))

    assert should_use_default_attention_mask(None) is True
    assert should_use_default_attention_mask((None, KVCache.init(4))) is True
    assert should_use_default_attention_mask((None, populated_cache)) is False


def test_prepare_default_attention_mask_masks_negative_placeholder_ids():
    attention_mask = prepare_default_attention_mask(jnp.array([[-100, 1]], dtype=jnp.int32), None)

    assert attention_mask.tolist() == [[False, True]]


def test_prepare_default_attention_mask_rejects_rows_without_real_tokens():
    with pytest.raises(ValueError, match="at least one non-negative token"):
        prepare_default_attention_mask(jnp.array([[-100, -100]], dtype=jnp.int32), None)


def test_prepare_default_attention_mask_uses_populated_cache_mask_instead():
    k, v = _kv(seq_len=2)
    populated_cache = KVCache.init(4).update(k, v, mask=jnp.ones((1, 2), dtype=bool))

    assert prepare_default_attention_mask(jnp.array([[-100]], dtype=jnp.int32), (None, populated_cache)) is None


def test_cached_sequence_length_uses_first_populated_cache_entry():
    k, v = _kv(seq_len=2)
    populated_cache = KVCache.init(4).update(k, v, mask=jnp.ones((1, 2), dtype=bool))

    assert cached_sequence_length(None) is None
    assert cached_sequence_length((None, KVCache.init(4))) is None
    assert cached_sequence_length((None, populated_cache)) == 4


def test_cached_sequence_length_rejects_inconsistent_populated_cache_lengths():
    short_k, short_v = _kv(seq_len=2)
    long_k, long_v = _kv(seq_len=3)
    short_cache = KVCache.init(4).update(short_k, short_v, mask=jnp.ones((1, 2), dtype=bool))
    long_cache = KVCache.init(6).update(long_k, long_v, mask=jnp.ones((1, 3), dtype=bool))

    with pytest.raises(ValueError, match="share cached sequence length"):
        cached_sequence_length((short_cache, long_cache))


@pytest.mark.parametrize(
    "model_cls,config",
    [
        (LlamaModel, _small_model_config(use_rope=True)),
        (GPTNeoXModel, _small_model_config(use_rope=True, use_bias=True)),
        (
            GemmaModel,
            _small_model_config(
                hidden_size=8,
                use_rope=True,
                sliding_window=4,
                sliding_window_pattern=2,
            ),
        ),
    ],
)
def test_models_propagate_dtype_to_rotary_embedding_cache(model_cls, config):
    model = model_cls(config, dtype=jnp.bfloat16)
    input_ids = jnp.array([[0, 1]], dtype=jnp.int32)

    params = model.init(jax.random.PRNGKey(0), input_ids)

    assert params["cache"]["rotary_emb"]["cos_cached"].dtype == jnp.bfloat16
    assert params["cache"]["rotary_emb"]["sin_cached"].dtype == jnp.bfloat16
    if model_cls is GemmaModel:
        assert params["cache"]["rotary_emb_local"]["cos_cached"].dtype == jnp.bfloat16
        assert params["cache"]["rotary_emb_local"]["sin_cached"].dtype == jnp.bfloat16


@pytest.mark.parametrize(
    "model_cls,config,match",
    [
        (LlamaModel, _small_model_config(hidden_size=12, head_dim=4, num_heads=2), "Llama hidden_size"),
        (LlamaDecoder, _small_model_config(hidden_size=12, head_dim=4, num_heads=2), "Llama hidden_size"),
        (GPTNeoXModel, _small_model_config(hidden_size=12, head_dim=4, num_heads=2, use_bias=True), "GPT-NeoX hidden_size"),
        (GPTNeoXDecoder, _small_model_config(hidden_size=12, head_dim=4, num_heads=2, use_bias=True), "GPT-NeoX hidden_size"),
    ],
)
def test_classic_attention_architectures_reject_hidden_size_head_mismatch(model_cls, config, match):
    module = model_cls(config, dtype=jnp.float32)
    input_ids = jnp.array([[0, 1]], dtype=jnp.int32)
    hidden_states = jnp.ones((1, 2, config.hidden_size), dtype=jnp.float32)

    with pytest.raises(ValueError, match=match):
        if model_cls in (LlamaModel, GPTNeoXModel):
            module.init(jax.random.PRNGKey(0), input_ids)
        else:
            module.init(jax.random.PRNGKey(0), hidden_states)


def test_gemma_model_accepts_hidden_size_head_mismatch():
    config = _small_model_config(
        hidden_size=12,
        head_dim=4,
        num_heads=2,
        use_rope=True,
        sliding_window=4,
        sliding_window_pattern=2,
    )
    model = GemmaModel(config, dtype=jnp.float32)
    input_ids = jnp.array([[0, 1]], dtype=jnp.int32)

    params = model.init(jax.random.PRNGKey(0), input_ids)
    output = model.apply(params, input_ids)

    assert output.last_hidden_state.shape == (1, 2, config.hidden_size)


@pytest.mark.parametrize("fixture_name", MODEL_FIXTURES)
@pytest.mark.parametrize(
    "input_ids,exception,match",
    [
        (jnp.arange(4, dtype=jnp.int32), ValueError, "input_ids must be a 2D array"),
        (jnp.ones((1, 2, 1), dtype=jnp.int32), ValueError, "input_ids must be a 2D array"),
        (jnp.ones((1, 4), dtype=jnp.float32), TypeError, "integer token ids"),
        (jnp.ones((0, 4), dtype=jnp.int32), ValueError, "at least one batch row"),
        (jnp.ones((1, 0), dtype=jnp.int32), ValueError, "at least one token"),
    ],
)
def test_models_reject_invalid_input_ids(request, fixture_name, input_ids, exception, match):
    with jax.default_device(jax.devices("cpu")[0]):
        model, params = request.getfixturevalue(fixture_name)

        with pytest.raises(exception, match=match):
            model.apply(params, input_ids)


@pytest.mark.parametrize("fixture_name", MODEL_FIXTURES)
@pytest.mark.parametrize(
    "attention_mask,exception,match",
    [
        (jnp.ones((4,), dtype=bool), ValueError, "attention_mask must be a 2D array"),
        (jnp.ones((1, 2, 1), dtype=bool), ValueError, "attention_mask must be a 2D array"),
        (jnp.ones((1, 3), dtype=bool), ValueError, "attention_mask shape must match"),
        (jnp.ones((1, 4), dtype=jnp.float32), TypeError, "attention_mask must be boolean or integer"),
    ],
)
def test_models_reject_invalid_attention_mask(request, fixture_name, attention_mask, exception, match):
    with jax.default_device(jax.devices("cpu")[0]):
        model, params = request.getfixturevalue(fixture_name)
        input_ids = jnp.arange(4, dtype=jnp.int32)[None]

        with pytest.raises(exception, match=match):
            model.apply(params, input_ids, attention_mask=attention_mask)


@pytest.mark.parametrize("fixture_name", MODEL_FIXTURES)
def test_models_reject_attention_mask_without_valid_tokens(request, fixture_name):
    with jax.default_device(jax.devices("cpu")[0]):
        model, params = request.getfixturevalue(fixture_name)
        input_ids = jnp.tile(jnp.arange(4, dtype=jnp.int32)[None], (2, 1))
        attention_mask = jnp.array([[1, 1, 0, 0], [0, 0, 0, 0]], dtype=bool)

        with pytest.raises(ValueError, match="at least one valid token"):
            model.apply(params, input_ids, attention_mask=attention_mask)


@pytest.mark.parametrize("fixture_name", MODEL_FIXTURES)
def test_models_reject_input_ids_at_or_above_vocab_size(request, fixture_name):
    with jax.default_device(jax.devices("cpu")[0]):
        model, params = request.getfixturevalue(fixture_name)
        input_ids = jnp.array([[0, model.config.vocab_size]], dtype=jnp.int32)

        with pytest.raises(ValueError, match="less than vocab_size"):
            model.apply(params, input_ids)


@pytest.mark.parametrize("fixture_name", MODEL_FIXTURES)
def test_models_accept_negative_mask_placeholder_ids(request, fixture_name):
    with jax.default_device(jax.devices("cpu")[0]):
        model, params = request.getfixturevalue(fixture_name)
        input_ids = jnp.array([[-100, 1]], dtype=jnp.int32)

        output = model.apply(params, input_ids)

    assert output.last_hidden_state.shape[:2] == input_ids.shape


@pytest.mark.parametrize("fixture_name", MODEL_FIXTURES)
def test_models_accept_masked_negative_placeholder_ids(request, fixture_name):
    with jax.default_device(jax.devices("cpu")[0]):
        model, params = request.getfixturevalue(fixture_name)
        input_ids = jnp.array([[-100, 1]], dtype=jnp.int32)
        attention_mask = jnp.array([[0, 1]], dtype=jnp.int32)

        output = model.apply(params, input_ids, attention_mask=attention_mask)

    assert output.last_hidden_state.shape[:2] == input_ids.shape


@pytest.mark.parametrize("fixture_name", MODEL_FIXTURES)
def test_models_reject_valid_negative_placeholder_ids(request, fixture_name):
    with jax.default_device(jax.devices("cpu")[0]):
        model, params = request.getfixturevalue(fixture_name)
        input_ids = jnp.array([[-100, 1]], dtype=jnp.int32)
        attention_mask = jnp.array([[1, 1]], dtype=jnp.int32)

        with pytest.raises(ValueError, match="must not mark negative placeholder"):
            model.apply(params, input_ids, attention_mask=attention_mask)


@pytest.mark.parametrize("fixture_name", MODEL_FIXTURES)
def test_models_reject_all_negative_placeholder_rows_without_attention_mask(request, fixture_name):
    with jax.default_device(jax.devices("cpu")[0]):
        model, params = request.getfixturevalue(fixture_name)
        input_ids = jnp.array([[-100, -100]], dtype=jnp.int32)

        with pytest.raises(ValueError, match="at least one non-negative token"):
            model.apply(params, input_ids)


@pytest.mark.parametrize("fixture_name", MODEL_FIXTURES)
def test_models_canonicalize_integer_attention_mask(request, fixture_name):
    with jax.default_device(jax.devices("cpu")[0]):
        model, params = request.getfixturevalue(fixture_name)
        input_ids = jnp.arange(4, dtype=jnp.int32)[None]
        integer_mask = jnp.array([[1, 1, 0, 0]], dtype=jnp.int32)
        bool_mask = integer_mask.astype(bool)

        output = model.apply(params, input_ids, attention_mask=integer_mask)
        expected = model.apply(params, input_ids, attention_mask=bool_mask)

    assert np.allclose(output.last_hidden_state, expected.last_hidden_state, atol=1e-6)


@pytest.mark.parametrize("fixture_name", MODEL_FIXTURES)
@pytest.mark.parametrize(
    "position_ids,exception,match",
    [
        (jnp.arange(4, dtype=jnp.int32), ValueError, "position_ids must be a 2D array"),
        (jnp.ones((1, 4), dtype=jnp.float32), TypeError, "integer positions"),
        (jnp.ones((1, 3), dtype=jnp.int32), ValueError, "position_ids shape must be broadcastable"),
        (jnp.array([[-1, 0, 1, 2]], dtype=jnp.int32), ValueError, "non-negative positions"),
    ],
)
def test_models_reject_invalid_position_ids(request, fixture_name, position_ids, exception, match):
    with jax.default_device(jax.devices("cpu")[0]):
        model, params = request.getfixturevalue(fixture_name)
        input_ids = jnp.arange(4, dtype=jnp.int32)[None]

        with pytest.raises(exception, match=match):
            model.apply(params, input_ids, position_ids=position_ids)


@pytest.mark.parametrize("fixture_name", MODEL_FIXTURES)
def test_models_reject_position_ids_at_or_above_max_position_embeddings(request, fixture_name):
    with jax.default_device(jax.devices("cpu")[0]):
        model, params = request.getfixturevalue(fixture_name)
        input_ids = jnp.arange(4, dtype=jnp.int32)[None]
        position_ids = jnp.array([[0, 1, 2, model.config.max_position_embeddings]], dtype=jnp.int32)

        with pytest.raises(ValueError, match="less than max_position_embeddings"):
            model.apply(params, input_ids, position_ids=position_ids)


@pytest.mark.parametrize("fixture_name", MODEL_FIXTURES)
def test_models_accept_broadcast_position_ids(request, fixture_name):
    with jax.default_device(jax.devices("cpu")[0]):
        model, params = request.getfixturevalue(fixture_name)
        input_ids = jnp.arange(4, dtype=jnp.int32)[None]
        position_ids = jnp.arange(4, dtype=jnp.int32)[None]

        output = model.apply(params, input_ids, position_ids=position_ids)

    assert output.last_hidden_state.shape[:2] == input_ids.shape


@pytest.mark.parametrize("fixture_name", MODEL_FIXTURES)
def test_models_reject_wrong_number_of_kv_caches(request, fixture_name):
    with jax.default_device(jax.devices("cpu")[0]):
        model, params = request.getfixturevalue(fixture_name)
        input_ids = jnp.arange(4, dtype=jnp.int32)[None]

        with pytest.raises(ValueError, match="one cache per layer"):
            model.apply(params, input_ids, kv_caches=(), use_cache=True)


@pytest.mark.parametrize("fixture_name", MODEL_FIXTURES)
def test_models_reject_non_sequence_kv_caches(request, fixture_name):
    with jax.default_device(jax.devices("cpu")[0]):
        model, params = request.getfixturevalue(fixture_name)
        input_ids = jnp.arange(4, dtype=jnp.int32)[None]

        with pytest.raises(TypeError, match="kv_caches must be a sequence"):
            model.apply(params, input_ids, kv_caches=KVCache.init(4), use_cache=True)


@pytest.mark.parametrize("fixture_name", MODEL_FIXTURES)
def test_models_reject_invalid_kv_cache_entries(request, fixture_name):
    with jax.default_device(jax.devices("cpu")[0]):
        model, params = request.getfixturevalue(fixture_name)
        input_ids = jnp.arange(4, dtype=jnp.int32)[None]
        kv_caches = (None,) * (model.num_layers - 1) + (object(),)

        with pytest.raises(TypeError, match="KVCache instances or None"):
            model.apply(params, input_ids, kv_caches=kv_caches, use_cache=True)


@pytest.mark.parametrize("fixture_name", MODEL_FIXTURES)
def test_models_accept_empty_kv_cache_entries_without_attention_mask(request, fixture_name):
    with jax.default_device(jax.devices("cpu")[0]):
        model, params = request.getfixturevalue(fixture_name)
        input_ids = jnp.arange(4, dtype=jnp.int32)[None]
        kv_caches = (None,) * model.num_layers

        output = model.apply(params, input_ids, kv_caches=kv_caches, use_cache=True)

    assert output.last_hidden_state.shape[:2] == input_ids.shape
    assert output.kv_caches == kv_caches


@pytest.mark.parametrize("fixture_name", MODEL_FIXTURES)
def test_models_accept_populated_cache_attention_mask_shape(request, fixture_name):
    with jax.default_device(jax.devices("cpu")[0]):
        model, params = request.getfixturevalue(fixture_name)
        prompt_ids = jnp.array([[0, 1]], dtype=jnp.int32)
        kv_caches = tuple(KVCache.init(model.config.max_position_embeddings) for _ in range(model.num_layers))
        prefill = model.apply(params, prompt_ids, kv_caches=kv_caches, use_cache=True)
        input_ids = jnp.array([[2]], dtype=jnp.int32)
        attention_mask = prefill.kv_caches[0].mask

        output = model.apply(
            params,
            input_ids,
            attention_mask=attention_mask,
            kv_caches=prefill.kv_caches,
            use_cache=True,
        )

    assert output.last_hidden_state.shape[:2] == input_ids.shape
    assert output.kv_caches is not None


@pytest.mark.parametrize("fixture_name", MODEL_FIXTURES)
def test_models_reject_token_attention_mask_for_populated_cache(request, fixture_name):
    with jax.default_device(jax.devices("cpu")[0]):
        model, params = request.getfixturevalue(fixture_name)
        prompt_ids = jnp.array([[0, 1]], dtype=jnp.int32)
        kv_caches = tuple(KVCache.init(model.config.max_position_embeddings) for _ in range(model.num_layers))
        prefill = model.apply(params, prompt_ids, kv_caches=kv_caches, use_cache=True)
        input_ids = jnp.array([[2]], dtype=jnp.int32)

        with pytest.raises(ValueError, match="populated KV cache mask shape"):
            model.apply(
                params,
                input_ids,
                attention_mask=jnp.ones(input_ids.shape, dtype=bool),
                kv_caches=prefill.kv_caches,
                use_cache=True,
            )


@pytest.mark.parametrize("fixture_name", MODEL_FIXTURES)
@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"use_cache": 1}, "use_cache must be a boolean"),
        ({"use_cache": None}, "use_cache must be a boolean"),
        ({"output_attentions": 1}, "output_attentions must be a boolean"),
        ({"output_hidden_states": "true"}, "output_hidden_states must be a boolean"),
    ],
)
def test_models_reject_non_boolean_output_flags(request, fixture_name, kwargs, match):
    with jax.default_device(jax.devices("cpu")[0]):
        model, params = request.getfixturevalue(fixture_name)
        input_ids = jnp.arange(4, dtype=jnp.int32)[None]

        with pytest.raises(TypeError, match=match):
            model.apply(params, input_ids, **kwargs)


@pytest.mark.parametrize("fixture_name", MODEL_FIXTURES)
def test_models_accept_numpy_boolean_output_flags(request, fixture_name):
    with jax.default_device(jax.devices("cpu")[0]):
        model, params = request.getfixturevalue(fixture_name)
        input_ids = jnp.arange(4, dtype=jnp.int32)[None]

        output = model.apply(
            params,
            input_ids,
            use_cache=np.bool_(True),
            output_attentions=np.bool_(False),
            output_hidden_states=np.bool_(True),
        )

    assert output.kv_caches is not None
    assert output.hidden_states is not None
    assert output.attention_weights is None
