#!/usr/bin/env bash

# download mnist
mkdir -p data/mnist
cd data/mnist

BASE="https://web.archive.org/web/20220331225223/https://yann.lecun.com/exdb/mnist"

for f in train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz \
         t10k-images-idx3-ubyte.gz  t10k-labels-idx1-ubyte.gz
do
    curl -L -O "$BASE/$f"
    gunzip -f "$f"
done