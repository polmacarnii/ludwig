# Ludwig
AI model for "undo" the low-pass filter(with 4kHz cut). Ludwig can also work for bad recordings

## How to use
### Get a copy of the project
```sh
git clone https://github.com/polmacarnii/ludwig.git
cd ludwig
```

### Get the dataset
The dataset could be, from example, your ~/Music directory
> [!NOTE]
> Only .flac, .wav and .m4a are supported, because they use to be lossless(atleast most of the time). Please don't convert mp3 to this formats to make it work, ensure your files are lossless(Like from CD-Rips) to get good results

### Prepare the dataset
```sh
sh prep-dataset.sh [dataset directory]
```

This will output dataset/original/, dataset/filter/ and dataset/target/ :
- on dataset/original/ there is the original audio
- on dataset/filter/ there is the audio with low pass effect
- on dataset/target/ there is the residual (original - filter), this is what Ludwig learns to predict

> [!NOTE]
> The audio is stored without any class of compression, so, your dataset directory may weight GBs

The audio is stored on PCM 16-bit little endian 44100 Hz Mono on .raw

### Setup python
```
python3 -m venv .venv
source .venv/bin/activate # if you use fish, .venv/bin/activate.fish
pip install -r requirements.txt
```

### Train
```sh
python -m src.train
```
Checkpoint is stored at checkpoint/last.pt and checkpoint/best.pt.

> [!NOTE]
> You can use "--resume checkpoint/last.pt" to continue

### Reconstruct a file
Once you have a trained checkpoint:
```sh
python -m src.main -i filtered.flac -o reconstructed.flac
```
- the `-i` and `-o` accepts anything ffmpeg doe
- you can pass the flag `-m` to specify a checkpoint
- Stereo is supported: on stereo each channel is processed as is own
- Everything is resampled to 44100 Hz, because thats the sample rate the model was trained on
- Needs ffmpeg and ffprobe on your PATH

# DISCLAMERS
- The output Ludwig produces its an aproximation, it does not recover information perfectly, it invents it
- The dataset you use is gonna define your output, if you train based on just voices, it would be good for voices, but pretty bad for songs, and vice-versa, if you train using whole songs, it may not work well for only voices

## License
The whole source code is at public domain, ¡Do whatever you please :)!
