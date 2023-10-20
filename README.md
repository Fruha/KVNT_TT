## Description
Тестовое задание, написание скрипта для пайплайна VAD+ASR

## Installation
### WINDOWS
Установить отдельно **ffmpeg**
```bash
git clone https://github.com/Fruha/KVNT_TT
cd KVNT_TT
conda create --name kvnt python==3.10.12 -y
conda activate kvnt
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install -r requirements.txt
```

### Linux
```bash
git clone https://github.com/Fruha/KVNT_TT 
cd KVNT_TT
conda create --name kvnt python==3.10.12 -y
conda activate kvnt
apt-get update && apt-get install -y libsndfile1 ffmpeg
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install -r requirements.txt
```

## Usage

```bash
conda activate kvnt
python main.py
```

Output
```bash
100%|████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:02<00:00,  1.14it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:02<00:00,  1.34it/s]
creating speech segments: 100%|██████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 18.57it/s]
Transcribing: 100%|██████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 10.03it/s]

filepath: 1.wav
Transcribition: in the second i tell what is understood of her on earth here my lady is desired

filepath: 2.wav
Transcribition: this second part is divided into two four in the one i speak of the eyes which are the beginning of love in the second i speak of the mouth which is the end of love

filepath: 3.wav
Transcribition: are you talking to me you and your city are scum and everyone in greece hates you
```

## Sorce
https://github.com/NVIDIA/NeMo/tree/main/examples/asr/asr_vad