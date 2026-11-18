.PHONY: all data train report test

all: data train report

data:      ## download play-by-play and build game states (~1 min)
	python scripts/build_dataset.py

train:     ## train + evaluate models with a season-based split (~1 min)
	python scripts/train.py

report:    ## regenerate the dashboard at docs/index.html
	python scripts/build_report.py

test:
	python -m pytest -q
