# Reproduces every artifact from raw seeds.
#   make all          full run (GPU recommended: DEVICE=cuda)
#   make pilot        10x-reduced smoke run (CPU, ~15 min)
DEVICE ?= cpu
PY ?= python3

all: artifacts/blind.jsonl
	$(PY) scripts/blind_baselines.py --manifest artifacts/blind.jsonl
	$(PY) scripts/surgery.py --pop artifacts/recipe.pt --device $(DEVICE)

artifacts/blind.jsonl:
	$(PY) scripts/run_all.py --device $(DEVICE) --out artifacts

pilot:
	$(PY) scripts/run_all.py --device $(DEVICE) --out artifacts_pilot --quick
	$(PY) scripts/blind_baselines.py --manifest artifacts_pilot/blind.jsonl

test:
	$(PY) -m pytest tests/ -q 2>/dev/null || $(PY) tests/test_piv.py

.PHONY: all pilot test
