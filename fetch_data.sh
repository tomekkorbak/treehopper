#!/usr/bin/env bash
echo "Download training set"
curl -O http://poleval.pl/task2/sentiment-treebank.tar.gz
tar -xvf sentiment-treebank.tar.gz
rm sentiment-treebank.tar.gz

echo "Download fastText word vectors for Polish"
curl -O https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.pl.zip
unzip wiki.pl.zip
rm wiki.pl.zip

echo "Download and build fastText from source"
git clone https://github.com/facebookresearch/fastText.git
make fastText

echo "Install Python requirements"
pip install -r requirements.txt
pip install http://download.pytorch.org/whl/cu80/torch-0.1.12.post2-cp35-cp35m-linux_x86_64.whl

mkdir tmp/