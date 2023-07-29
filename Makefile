setup:
	pip install --upgrade pip
	pip install -r requirements.txt
	python setup.py develop


install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black *.py

lint:
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics&&flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

test:
	pytest

gen:
	ludwig export_torchscript -m=results/api_experiment_run/model -op=model/

all: install lint test