SHELL=/bin/bash
LINT_PATHS=src/jaxml/ tests/ scripts/
CPU_TESTS=JAX_PLATFORMS=cpu pytest -m "not tpu" tests/
CRITICAL_CPU_TESTS=JAX_PLATFORMS=cpu pytest -m "critical and not tpu" tests/

.PHONY: pytest pytest-cpu pytest-critical-cpu pytest-tpu lint format format-check build-check lock-check dependency-check dependency-drift style verify-critical-cpu verify-cpu verify-milestone-cpu verify-tpu

pytest:
	${CPU_TESTS}

pytest-cpu:
	${CPU_TESTS}

pytest-critical-cpu:
	${CRITICAL_CPU_TESTS}

pytest-tpu:
	pytest -m "tpu" tests/

lint:
	flake8 ${LINT_PATHS} --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 ${LINT_PATHS} --count --ignore=E128,E501,E203 --exit-zero --statistics

format:
	isort ${LINT_PATHS}
	black ${LINT_PATHS}

format-check:
	isort --check-only ${LINT_PATHS}
	black --check ${LINT_PATHS}

build-check:
	uv build --clear --out-dir tmp/build-check

lock-check:
	uv lock --locked

dependency-check:
	uv pip check

dependency-drift:
	uv run --frozen --extra dev --extra tpu python scripts/check_dependency_drift.py

style: format lint

verify-critical-cpu: lock-check dependency-check lint format-check build-check pytest-critical-cpu

verify-cpu: verify-milestone-cpu

verify-milestone-cpu: lock-check dependency-check lint format-check pytest-cpu

verify-tpu: lock-check dependency-check pytest-tpu
