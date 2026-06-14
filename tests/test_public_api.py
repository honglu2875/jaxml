import importlib
import logging
import sys


def test_inference_engine_exports_documented_entry_points():
    from jaxml.inference_engine import Engine, InferenceConfig, SamplingMethod
    from jaxml.inference_engine.engine import Engine as EngineImpl
    from jaxml.inference_engine.engine import InferenceConfig as InferenceConfigImpl
    from jaxml.inference_engine.sampling import SamplingMethod as SamplingMethodImpl

    assert Engine is EngineImpl
    assert InferenceConfig is InferenceConfigImpl
    assert SamplingMethod is SamplingMethodImpl


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
