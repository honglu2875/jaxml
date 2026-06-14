from pathlib import Path

from jaxml.utils import JAXML_CACHE_DIR_ENV, compiled_fn_exist, compiled_fn_path


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
