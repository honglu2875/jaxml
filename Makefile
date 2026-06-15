SHELL=/bin/bash
LINT_PATHS=src/jaxml/ tests/

.PHONY: pytest pytest-cpu pytest-tpu lint format style verify-cpu verify-tpu

pytest:
	JAX_PLATFORMS=cpu pytest -m "not tpu" tests/

pytest-cpu:
	JAX_PLATFORMS=cpu pytest -m "not tpu" tests/

pytest-tpu:
	pytest -m "tpu" tests/

lint:
	flake8 ${LINT_PATHS} --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 ${LINT_PATHS} --count --ignore=E128,E501,E203 --exit-zero --statistics

format:
	isort ${LINT_PATHS}
	black -l 127 ${LINT_PATHS}

style: format lint

verify-cpu: lint pytest-cpu

verify-tpu: pytest-tpu
