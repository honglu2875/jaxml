import pickle
from pathlib import Path

import pytest

from jaxml.utils import (
    JAXML_CACHE_DIR_ENV,
    _hash,
    _load_compiled_fn_from_path,
    compiled_fn_exist,
    compiled_fn_path,
    load_compiled_fn,
    load_if_exists,
    save_compiled_fn,
)


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


@pytest.mark.parametrize(
    "name,hash",
    [
        ("", "abc"),
        (".", "abc"),
        ("..", "abc"),
        ("../decode", "abc"),
        ("nested/decode", "abc"),
        ("nested\\decode", "abc"),
        ("decode", ""),
        ("decode", "../abc"),
        ("decode", "nested\\abc"),
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
    assert compiled_fn_exist("decode", "abc")


def test_save_compiled_fn_replaces_payloads_atomically(monkeypatch, tmp_path):
    monkeypatch.setenv(JAXML_CACHE_DIR_ENV, str(tmp_path))
    monkeypatch.setattr("jax.experimental.serialize_executable.serialize", lambda fn: (b"new-aot", "in-tree", "out-tree"))
    cache_entry = compiled_fn_path("decode", "abc")
    cache_entry.mkdir(parents=True)
    (cache_entry / "aot").write_bytes(b"old-aot")
    with (cache_entry / "in_out_spec").open("wb") as f:
        pickle.dump(("old-in", "old-out"), f)

    byte_count = save_compiled_fn(object(), "decode", "abc", log=False)

    assert byte_count == len(b"new-aot") + len(pickle.dumps(("in-tree", "out-tree")))
    assert (cache_entry / "aot").read_bytes() == b"new-aot"
    with (cache_entry / "in_out_spec").open("rb") as f:
        assert pickle.load(f) == ("in-tree", "out-tree")
    assert not list(cache_entry.glob("*.tmp"))


def test_load_compiled_fn_cache_key_includes_resolved_cache_path(monkeypatch, tmp_path):
    def write_cache_entry(root, payload):
        cache_entry = compiled_fn_path("decode", "abc", cache_dir=root)
        cache_entry.mkdir(parents=True)
        (cache_entry / "aot").write_bytes(payload)
        with (cache_entry / "in_out_spec").open("wb") as f:
            pickle.dump(("in-tree", "out-tree"), f)

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
