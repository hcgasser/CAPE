SHELL := /bin/bash

image:
	source ${CAPE}/setup/create_image.sh

venv:
	source ./setup/setup_conda.sh

external:
	${PF}/setup/setup_external.sh

format:
	find . -name '*.py' -exec black {} +

lint:
	# ignores py files in hidden directories errors
	find . -not -path '*/.*' -type f -name '*.py'  -a -not -name 'cape-eval.py' -exec pylint --rcfile=.pylintrc {} + || true

test:
	./tools/run_tests.sh

ci:	format lint test

