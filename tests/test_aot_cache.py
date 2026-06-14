from pathlib import Path
import pickle

from jaxml.utils import JAXML_CACHE_DIR_ENV, _load_compiled_fn_from_path, compiled_fn_exist, compiled_fn_path, load_compiled_fn


def test_compiled_fn_path_defaults_to_project_cache(monkeypatch):
    monkeypatch.delenv(JAXML_CACHE_DIR_ENV, raising=False)

    assert compiled_fn_path("decode", "abc") == Path(".jaxml") / "decode_abc"


def test_compiled_fn_path_uses_cache_dir_env(monkeypatch, tmp_path):
    monkeypatch.setenv(JAXML_CACHE_DIR_ENV, str(tmp_path))

    assert compiled_fn_path("prefill", "abc") == tmp_path / "prefill_abc"


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
