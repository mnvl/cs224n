#!/usr/bin/env bash
# Downloads raw data into ./download
# and saves preprocessed data into ./data

# download punkt, perluniprops
if [ ! -d "/usr/local/share/nltk_data/tokenizers/punkt" ]; then
    python2 -m nltk.downloader punkt
fi

# SQuAD preprocess is in charge of downloading
# and formatting the data to be consumed later
DATA_DIR=data
DOWNLOAD_DIR=download
mkdir -p $DATA_DIR
rm -rf $DATA_DIR
python2 ./preprocessing/squad_preprocess.py

# Download distributed word representations
python2 ./preprocessing/dwr.py

# Data processing for TensorFlow
python2 ./qa_data.py --glove_dim 100
