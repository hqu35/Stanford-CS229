.PHONY: check test verify data smoke

check:
	python scripts/preflight.py

# Initially, tests for student-owned TODOs will fail intentionally.
test:
	python -m pytest -q

verify:
	python scripts/verify_forward.py

data:
	python scripts/generate_data.py --overwrite

smoke:
	python scripts/train.py --run-name smoke --updates 2 --batch-size 16 --device cpu
