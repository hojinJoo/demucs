#!/bin/bash

name=$1

python3 -m tools.test_pretrained --repo ./release_models -n ${name}