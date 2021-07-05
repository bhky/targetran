#!/usr/bin/env bash
set -e

find targetran -iname "*.py" | grep -v -e "__init__.py" | xargs -L 1 pylint
find targetran -iname "*.py" | grep -v -e "__init__.py" | xargs -L 1 mypy --strict --check-untyped-defs
find targetran -iname "*.py" | grep -v -e "__init__.py" | xargs -L 1 python3 -m unittest