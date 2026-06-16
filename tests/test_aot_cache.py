import json
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxml.utils import (
    JAXML_CACHE_DIR_ENV,
    _aot_cache_metadata,
    _hash,
    _load_compiled_fn_from_path,
    compiled_fn_exist,
    compiled_fn_path,
    load_compiled_fn,
    load_if_exists,
    save_compiled_fn,
)

pytestmark = pytest.mark.milestone


def _write_aot_cache_entry(cache_entry, payload=b"compiled", in_tree="in-tree", out_tree="out-tree"):
    cache_entry.mkdir(parents=True)
    (cache_entry / "aot").write_bytes(payload)
    with (cache_entry / "in_out_spec").open("wb") as f:
        pickle.dump((in_tree, out_tree), f)
    (cache_entry / "metadata.json").write_text(json.dumps(_aot_cache_metadata(), sort_keys=True))


def test_compiled_fn_path_defaults_to_project_cache(monkeypatch):
    monkeypatch.delenv(JAXML_CACHE_DIR_ENV, raising=False)

    assert compiled_fn_path("decode", "abc") == Path(".jaxml") / "decode_abc"


def test_hash_frames_arguments_to_avoid_concatenation_collisions():
    assert _hash("ab", "c") != _hash("a", "bc")


def test_hash_rejects_non_string_arguments():
    with pytest.raises(TypeError, match="hash arguments must be strings"):
        _hash("decode", 1)


def test_compiled_fn_path_uses_cache_dir_env(monkeypatch, tmp_path):
    monkeypatch.setenv(JAXML_CACHE_DIR_ENV, str(tmp_path))

    assert compiled_fn_path("prefill", "abc") == tmp_path / "prefill_abc"


def test_compiled_fn_path_accepts_integer_hash(tmp_path):
    assert compiled_fn_path("prefill", 0, cache_dir=tmp_path) == tmp_path / "prefill_0"


def test_compiled_fn_path_accepts_numpy_integer_hash(tmp_path):
    assert compiled_fn_path("prefill", np.int64(7), cache_dir=tmp_path) == tmp_path / "prefill_7"


@pytest.mark.parametrize("name", [None, 123, np.int64(1), b"decode", object()])
def test_compiled_fn_path_rejects_non_string_names(tmp_path, name):
    with pytest.raises(TypeError, match="AOT cache name must be a string"):
        compiled_fn_path(name, "abc", cache_dir=tmp_path)


@pytest.mark.parametrize("hash_value", [None, True, np.bool_(True), b"abc", object()])
def test_compiled_fn_path_rejects_non_string_or_integer_hashes(tmp_path, hash_value):
    with pytest.raises(TypeError, match="AOT cache hash must be a string or integer"):
        compiled_fn_path("prefill", hash_value, cache_dir=tmp_path)


@pytest.mark.parametrize(
    "name,hash",
    [
        ("", "abc"),
        (".", "abc"),
        ("..", "abc"),
        ("../decode", "abc"),
        ("nested/decode", "abc"),
        ("nested\\decode", "abc"),
        (" decode", "abc"),
        ("decode ", "abc"),
        ("de code", "abc"),
        ("decode\n", "abc"),
        ("decode", ""),
        ("decode", "../abc"),
        ("decode", "nested\\abc"),
        ("decode", " abc"),
        ("decode", "ab c"),
    ],
)
def test_compiled_fn_path_rejects_unsafe_key_parts(tmp_path, name, hash):
    with pytest.raises(ValueError, match="AOT cache"):
        compiled_fn_path(name, hash, cache_dir=tmp_path)


def test_load_if_exists_rejects_unsafe_key_parts_before_wrapping():
    with pytest.raises(ValueError, match="AOT cache"):
        load_if_exists("../decode", "abc", log=False)


def test_save_compiled_fn_rejects_unsafe_key_parts_before_serializing(monkeypatch):
    serialize_calls = []
    monkeypatch.setattr(
        "jax.experimental.serialize_executable.serialize",
        lambda fn: serialize_calls.append(fn) or (b"compiled", "in-tree", "out-tree"),
    )

    with pytest.raises(ValueError, match="AOT cache"):
        save_compiled_fn(object(), "../decode", "abc", log=False)

    assert serialize_calls == []


def test_compiled_fn_exist_requires_payload_files(monkeypatch, tmp_path):
    monkeypatch.setenv(JAXML_CACHE_DIR_ENV, str(tmp_path))
    cache_entry = compiled_fn_path("decode", "abc")

    assert not compiled_fn_exist("decode", "abc")

    cache_entry.mkdir(parents=True)
    assert not compiled_fn_exist("decode", "abc")

    (cache_entry / "aot").write_bytes(b"compiled")
    assert not compiled_fn_exist("decode", "abc")

    (cache_entry / "in_out_spec").write_bytes(b"spec")
    assert not compiled_fn_exist("decode", "abc")

    (cache_entry / "metadata.json").write_text(json.dumps(_aot_cache_metadata(), sort_keys=True))
    assert compiled_fn_exist("decode", "abc")


@pytest.mark.parametrize("empty_payload_name", ["aot", "in_out_spec", "metadata.json"])
def test_compiled_fn_exist_rejects_empty_payload_files(monkeypatch, tmp_path, empty_payload_name):
    monkeypatch.setenv(JAXML_CACHE_DIR_ENV, str(tmp_path))
    cache_entry = compiled_fn_path("decode", "abc")
    _write_aot_cache_entry(cache_entry)
    (cache_entry / empty_payload_name).write_bytes(b"")

    assert not compiled_fn_exist("decode", "abc")


@pytest.mark.parametrize("empty_payload_name", ["aot", "in_out_spec", "metadata.json"])
def test_load_compiled_fn_rejects_empty_payload_files_before_deserializing(monkeypatch, tmp_path, empty_payload_name):
    monkeypatch.setenv(JAXML_CACHE_DIR_ENV, str(tmp_path))
    deserialize_calls = []
    monkeypatch.setattr(
        "jax.experimental.serialize_executable.deserialize_and_load",
        lambda *args: deserialize_calls.append(args),
    )
    _load_compiled_fn_from_path.cache_clear()
    cache_entry = compiled_fn_path("decode", "abc")
    _write_aot_cache_entry(cache_entry)
    (cache_entry / empty_payload_name).write_bytes(b"")

    with pytest.raises(ValueError, match="empty payload file"):
        load_compiled_fn("decode", "abc", log=False)

    assert deserialize_calls == []


def test_load_compiled_fn_rejects_metadata_mismatch_before_deserializing(monkeypatch, tmp_path):
    monkeypatch.setenv(JAXML_CACHE_DIR_ENV, str(tmp_path))
    deserialize_calls = []
    monkeypatch.setattr(
        "jax.experimental.serialize_executable.deserialize_and_load",
        lambda *args: deserialize_calls.append(args),
    )
    _load_compiled_fn_from_path.cache_clear()
    cache_entry = compiled_fn_path("decode", "abc")
    _write_aot_cache_entry(cache_entry)
    (cache_entry / "metadata.json").write_text(json.dumps(_aot_cache_metadata() | {"jax": "0.0.0"}, sort_keys=True))

    with pytest.raises(ValueError, match="metadata mismatch"):
        load_compiled_fn("decode", "abc", log=False)

    assert deserialize_calls == []


def test_save_compiled_fn_replaces_payloads_atomically(monkeypatch, tmp_path):
    monkeypatch.setenv(JAXML_CACHE_DIR_ENV, str(tmp_path))
    monkeypatch.setattr("jax.experimental.serialize_executable.serialize", lambda fn: (b"new-aot", "in-tree", "out-tree"))
    cache_entry = compiled_fn_path("decode", "abc")
    cache_entry.mkdir(parents=True)
    (cache_entry / "aot").write_bytes(b"old-aot")
    with (cache_entry / "in_out_spec").open("wb") as f:
        pickle.dump(("old-in", "old-out"), f)

    byte_count = save_compiled_fn(object(), "decode", "abc", log=False)

    expected_byte_count = len(b"new-aot")
    expected_byte_count += len(pickle.dumps(("in-tree", "out-tree")))
    expected_byte_count += len(json.dumps(_aot_cache_metadata(), sort_keys=True).encode("utf-8"))
    assert byte_count == expected_byte_count
    assert (cache_entry / "aot").read_bytes() == b"new-aot"
    with (cache_entry / "in_out_spec").open("rb") as f:
        assert pickle.load(f) == ("in-tree", "out-tree")
    assert json.loads((cache_entry / "metadata.json").read_text()) == _aot_cache_metadata()
    assert not list(cache_entry.glob("*.tmp"))


def test_save_compiled_fn_rejects_non_bytes_executable_payload(monkeypatch, tmp_path):
    monkeypatch.setenv(JAXML_CACHE_DIR_ENV, str(tmp_path))
    monkeypatch.setattr("jax.experimental.serialize_executable.serialize", lambda fn: ("not-bytes", "in-tree", "out-tree"))

    with pytest.raises(TypeError, match="Serialized AOT executable must be bytes"):
        save_compiled_fn(object(), "decode", "abc", log=False)

    cache_entry = compiled_fn_path("decode", "abc")
    assert not (cache_entry / "aot").exists()
    assert not (cache_entry / "in_out_spec").exists()
    assert not (cache_entry / "metadata.json").exists()


def test_save_compiled_fn_rejects_empty_executable_payload(monkeypatch, tmp_path):
    monkeypatch.setenv(JAXML_CACHE_DIR_ENV, str(tmp_path))
    monkeypatch.setattr("jax.experimental.serialize_executable.serialize", lambda fn: (b"", "in-tree", "out-tree"))

    with pytest.raises(ValueError, match="Serialized AOT executable must not be empty"):
        save_compiled_fn(object(), "decode", "abc", log=False)

    cache_entry = compiled_fn_path("decode", "abc")
    assert not (cache_entry / "aot").exists()
    assert not (cache_entry / "in_out_spec").exists()
    assert not (cache_entry / "metadata.json").exists()


def test_save_compiled_fn_invalidates_loaded_cache(monkeypatch, tmp_path):
    monkeypatch.setenv(JAXML_CACHE_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(
        "jax.experimental.serialize_executable.deserialize_and_load",
        lambda payload, in_tree, out_tree: (payload, in_tree, out_tree),
    )
    _load_compiled_fn_from_path.cache_clear()
    cache_entry = compiled_fn_path("decode", "abc")
    _write_aot_cache_entry(cache_entry, payload=b"old-aot", in_tree="old-in", out_tree="old-out")

    first = load_compiled_fn("decode", "abc", log=False)

    monkeypatch.setattr("jax.experimental.serialize_executable.serialize", lambda fn: (b"new-aot", "new-in", "new-out"))
    save_compiled_fn(object(), "decode", "abc", log=False)
    second = load_compiled_fn("decode", "abc", log=False)

    assert first == (b"old-aot", "old-in", "old-out")
    assert second == (b"new-aot", "new-in", "new-out")


def test_save_and_load_compiled_fn_round_trips_jax_executable(monkeypatch, tmp_path):
    if jax.default_backend() != "cpu":
        pytest.skip("Executable serialization round-trip is covered by the CPU milestone gate.")
    monkeypatch.setenv(JAXML_CACHE_DIR_ENV, str(tmp_path))
    _load_compiled_fn_from_path.cache_clear()

    def add_one(x):
        return x + 1

    with jax.default_device(jax.devices("cpu")[0]):
        compiled = jax.jit(add_one).lower(jnp.ones((4,), dtype=jnp.float32)).compile()
        byte_count = save_compiled_fn(compiled, "add_one", "abc", log=False)
        loaded = load_compiled_fn("add_one", "abc", log=False)
        result = loaded(jnp.arange(4, dtype=jnp.float32))

    assert byte_count > 0
    assert result.tolist() == [1.0, 2.0, 3.0, 4.0]


def test_load_compiled_fn_cache_key_includes_resolved_cache_path(monkeypatch, tmp_path):
    def write_cache_entry(root, payload):
        cache_entry = compiled_fn_path("decode", "abc", cache_dir=root)
        _write_aot_cache_entry(cache_entry, payload=payload)

    monkeypatch.setattr(
        "jax.experimental.serialize_executable.deserialize_and_load",
        lambda payload, in_tree, out_tree: (payload, in_tree, out_tree),
    )
    _load_compiled_fn_from_path.cache_clear()
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    write_cache_entry(first_root, b"first-payload")
    write_cache_entry(second_root, b"second-payload")

    monkeypatch.setenv(JAXML_CACHE_DIR_ENV, str(first_root))
    first = load_compiled_fn("decode", "abc", log=False)
    monkeypatch.setenv(JAXML_CACHE_DIR_ENV, str(second_root))
    second = load_compiled_fn("decode", "abc", log=False)

    assert first == (b"first-payload", "in-tree", "out-tree")
    assert second == (b"second-payload", "in-tree", "out-tree")


def test_load_compiled_fn_cache_key_resolves_relative_paths(monkeypatch, tmp_path):
    def write_cache_entry(cwd, payload):
        cache_entry = cwd / "cache" / "decode_abc"
        _write_aot_cache_entry(cache_entry, payload=payload)

    monkeypatch.setattr(
        "jax.experimental.serialize_executable.deserialize_and_load",
        lambda payload, in_tree, out_tree: (payload, in_tree, out_tree),
    )
    _load_compiled_fn_from_path.cache_clear()
    first_cwd = tmp_path / "first"
    second_cwd = tmp_path / "second"
    write_cache_entry(first_cwd, b"first-payload")
    write_cache_entry(second_cwd, b"second-payload")
    monkeypatch.setenv(JAXML_CACHE_DIR_ENV, "cache")

    monkeypatch.chdir(first_cwd)
    first = load_compiled_fn("decode", "abc", log=False)
    monkeypatch.chdir(second_cwd)
    second = load_compiled_fn("decode", "abc", log=False)

    assert first == (b"first-payload", "in-tree", "out-tree")
    assert second == (b"second-payload", "in-tree", "out-tree")


def test_load_compiled_fn_wraps_corrupt_spec_payload(monkeypatch, tmp_path):
    monkeypatch.setenv(JAXML_CACHE_DIR_ENV, str(tmp_path))
    _load_compiled_fn_from_path.cache_clear()
    cache_entry = compiled_fn_path("decode", "abc")
    _write_aot_cache_entry(cache_entry)
    (cache_entry / "in_out_spec").write_bytes(b"not-a-pickle")

    with pytest.raises(ValueError, match="Failed to load AOT cache entry") as exc_info:
        load_compiled_fn("decode", "abc", log=False)

    assert str(cache_entry.resolve()) in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, pickle.UnpicklingError)


def test_load_compiled_fn_wraps_deserializer_failures(monkeypatch, tmp_path):
    monkeypatch.setenv(JAXML_CACHE_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(
        "jax.experimental.serialize_executable.deserialize_and_load",
        lambda payload, in_tree, out_tree: (_ for _ in ()).throw(RuntimeError("bad executable")),
    )
    _load_compiled_fn_from_path.cache_clear()
    cache_entry = compiled_fn_path("decode", "abc")
    _write_aot_cache_entry(cache_entry)

    with pytest.raises(ValueError, match="Failed to load AOT cache entry") as exc_info:
        load_compiled_fn("decode", "abc", log=False)

    assert str(cache_entry.resolve()) in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, RuntimeError)


class FakeJit:
    def __init__(self, fn, compiled_fn):
        self.fn = fn
        self.compiled_fn = compiled_fn

    def lower(self, *args, **kwargs):
        return self

    def compile(self):
        return self.compiled_fn


def test_load_if_exists_recompiles_when_cached_load_fails(monkeypatch):
    compiled_calls = []

    def compiled_fn(x):
        compiled_calls.append(x)
        return x + 1

    monkeypatch.setattr("jaxml.utils.compiled_fn_exist", lambda name, hash: True)
    monkeypatch.setattr("jaxml.utils.load_compiled_fn", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad cache")))
    monkeypatch.setattr("jaxml.utils.save_compiled_fn", lambda *args, **kwargs: 1)
    monkeypatch.setattr("jaxml.utils.jax.jit", lambda fn: FakeJit(fn, compiled_fn))

    wrapped = load_if_exists("decode", "abc", log=False)(lambda x: x + 1)

    assert wrapped(2) == 3
    assert compiled_calls == [2]


def test_load_if_exists_recompiles_when_cached_input_tree_is_stale(monkeypatch):
    compiled_calls = []

    def stale_fn(x):
        raise TypeError("Function compiled with input pytree does not match the input pytree it was called with.")

    def compiled_fn(x):
        compiled_calls.append(x)
        return x + 1

    monkeypatch.setattr("jaxml.utils.compiled_fn_exist", lambda name, hash: True)
    monkeypatch.setattr("jaxml.utils.load_compiled_fn", lambda *args, **kwargs: stale_fn)
    monkeypatch.setattr("jaxml.utils.save_compiled_fn", lambda *args, **kwargs: 1)
    monkeypatch.setattr("jaxml.utils.jax.jit", lambda fn: FakeJit(fn, compiled_fn))

    wrapped = load_if_exists("prefill", "abc", log=False)(lambda x: x + 1)

    assert wrapped(2) == 3
    assert compiled_calls == [2]


def test_load_if_exists_recompiles_when_cached_argument_signature_is_stale(monkeypatch):
    compiled_calls = []

    def stale_fn(x):
        raise TypeError(
            "Argument types differ from the types for which this computation was compiled. "
            "The mismatches are: Argument 'x' compiled with float32[2] and called with float32[3]."
        )

    def compiled_fn(x):
        compiled_calls.append(x)
        return x + 1

    monkeypatch.setattr("jaxml.utils.compiled_fn_exist", lambda name, hash: True)
    monkeypatch.setattr("jaxml.utils.load_compiled_fn", lambda *args, **kwargs: stale_fn)
    monkeypatch.setattr("jaxml.utils.save_compiled_fn", lambda *args, **kwargs: 1)
    monkeypatch.setattr("jaxml.utils.jax.jit", lambda fn: FakeJit(fn, compiled_fn))

    wrapped = load_if_exists("prefill", "abc", log=False)(lambda x: x + 1)

    assert wrapped(2) == 3
    assert compiled_calls == [2]


def test_load_if_exists_preserves_non_stale_type_errors(monkeypatch):
    def broken_fn(x):
        raise TypeError("model bug")

    monkeypatch.setattr("jaxml.utils.compiled_fn_exist", lambda name, hash: True)
    monkeypatch.setattr("jaxml.utils.load_compiled_fn", lambda *args, **kwargs: broken_fn)

    wrapped = load_if_exists("prefill", "abc", log=False)(lambda x: x + 1)

    try:
        wrapped(2)
    except TypeError as e:
        assert str(e) == "model bug"
    else:
        raise AssertionError("Expected TypeError.")
