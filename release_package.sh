#!/usr/bin/env bash

python3 -m build
twine upload dist/*