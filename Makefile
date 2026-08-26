.PHONY: test repro-check repro-run docs wheel

test:
	python -m pytest -q tests

repro-check:
	smith-repro check

repro-run:
	@for case in 02_regulatory_activity 03_ribomap_transfer 05_agent; do \
		smith-repro run $$case --output-dir outputs/reproducibility/$$case; \
	done

docs:
	sphinx-build -W -b html docs/source docs/_build/html

wheel:
	python -m build --wheel
