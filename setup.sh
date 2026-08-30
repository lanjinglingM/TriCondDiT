conda create -n stable-audio python=3.10 -y
conda activate stable-audio

pip install -U pip setuptools wheel

pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu118

git clone https://github.com/Stability-AI/stable-audio-tools.git
cd stable-audio-tools

pip install -e .

pip uninstall flash-attn -y

huggingface-cli login
