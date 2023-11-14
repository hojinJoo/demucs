#!/bin/bash
sig=$1
name=$2
dirname=${PWD}

python3 -m tools.export ${sig}
mv ${dirname}/release_models/${sig}.th ${dirname}/release_models/${name}.th