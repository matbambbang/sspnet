conda create -n sspnet python=3.6

conda activate sspnet
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
pip install tensorflow-gpu==1.9.0
pip install overrides
pip install https://github.com/bethgelab/foolbox/archive/master.zip
pip install tqdm
pip install tb-nightly
pip install future
