#!/usr/bin/env bash

if [ -f "data/en.txt" ]; then
  echo "File data/en.txt exists."
else
  wget https://object.pouta.csc.fi/OPUS-OpenSubtitles/v1/mono/en.txt.gz -O data/en.txt.gz
  gzip -d data/en.txt.gz
fi

python src/data/parse_opus_data.py data/en.txt
