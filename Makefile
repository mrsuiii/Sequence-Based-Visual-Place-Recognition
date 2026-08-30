.PHONY: verify-data characterise decode describe align verify gt-sheets evaluate manifest report lint test all

PY := python -m routealign

verify-data:
	$(PY) verify-data

characterise:
	$(PY) characterise

decode:
	$(PY) decode

describe:
	$(PY) describe --feat $(FEAT)

align:
	$(PY) align

verify:
	$(PY) verify

gt-sheets:
	$(PY) gt-sheets

evaluate:
	$(PY) evaluate

manifest:
	$(PY) manifest

report:
	$(PY) report

lint:
	ruff format --check .
	ruff check .

test:
	pytest -q

all: verify-data decode characterise describe align verify gt-sheets evaluate manifest report
	@echo "make all: done"
