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
pip install Cython
python -m pip install git+https://github.com/NVIDIA/NeMo.git@{BRANCH}#egg=nemo_toolkit[all]
```

### Linux
```bash
git clone https://github.com/Fruha/KVNT_TT 
cd KVNT_TT
conda create --name kvnt python==3.10.12
conda activate kvnt
apt-get update && apt-get install -y libsndfile1 ffmpeg
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install Cython
```

## Usage

```bash
conda activate kvnt
python main.py
```