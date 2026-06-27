conda deactivate
conda remove -n openmmlab --all -y
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
conda install pytorch torchvision -c pytorch 安装torch1.12.1cu113:
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 -ihttps://mirrors.aliyun.com/pypi/simple/--extra-index-urlhttps://download.pytorch.org/whl/cu113
pip install -U openmim -i https://mirrors.aliyun.com/pypi/simple/
mim install mmengine -ihttps://mirrors.aliyun.com/pypi/simple/
mim install "mmcv==2.1.0" -ihttps://mirrors.aliyun.com/pypi/simple/
cd mmdetection
pip install -v -e .  -ihttps://mirrors.aliyun.com/pypi/simple/
pip install psutil -ihttps://mirrors.aliyun.com/pypi/simple/