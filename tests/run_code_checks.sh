#!/usr/bin/env bash
set -e

find targetran -iname "*.py" | grep -v -e "__init__.py" | xargs -L 1 pylint
find targetran -iname "*.py" | grep -v -e "__init__.py" | xargs -L 1 mypy --strict
find targetran -iname "*.py" | grep -v -e "__init__.py" | xargs -L 1 python3 -m unittest