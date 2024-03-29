#!/usr/bin/env bash
set -e

find targetran -iname "*.py" | grep -v -e "__init__.py" | xargs -L 1 pylint --errors-only
find targetran -iname "*.py" | grep -v -e "__init__.py" | xargs -L 1 pylint --exit-zero
find targetran -iname "*.py" | grep -v -e "__init__.py" | xargs -L 1 mypy --strict --check-untyped-defs --disable-error-code empty-body
find tests -iname "*.py" | xargs -L 1 python3 -m unittest