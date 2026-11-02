#!/bin/sh
# Variables
IFS='
'
LOWPASS="4000" # where its cut

# Functions
help() {
    echo "$0 - get the dataset"
    echo "Usage:"
    echo "  $0 [-dir directory that contains the songs to train on]"
    printf "\n"
    echo "Note: Only wav, flac and m4a are supported"
}

# Program
if [ -z "$1" ]; then
    echo "You must specify one directory as input"
    help
    exit 1
fi

rm -fr dataset/*

mkdir -p dataset/original/
mkdir -p dataset/filter/
mkdir -p dataset/target/

set -- $(find "$1" -type f \( -name "*.flac" -o -name "*.m4a" -o -name "*.wav" \))

unset IFS

# original dataset
OG_COUNT=0
for file in "$@"; do
    ffmpeg -i "$file" -f s16le -acodec pcm_s16le -ar 44100 -ac 1 dataset/original/$OG_COUNT.raw >/dev/null 2>&1
    OG_COUNT=$((OG_COUNT + 1))
done

# out dataset
OUT_COUNT=0
for file in "$@"; do
    ffmpeg -i "$file" -f s16le -acodec pcm_s16le -ar 44100 -ac 1 -af "lowpass=f=$LOWPASS" dataset/filter/$OUT_COUNT.raw >/dev/null 2>&1
    OUT_COUNT=$((OUT_COUNT + 1))
done

# remainder, original - out
python3 make-target.py
