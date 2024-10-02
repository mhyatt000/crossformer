# INSTALL

## opendr

https://github.com/mattloper/opendr/issues/19#issuecomment-532509726

```
sudo apt-get install libglu1-mesa-dev freeglut3-dev mesa-common-dev
sudo apt-get install libosmesa6-dev
# pip install opendr
```

https://github.com/microsoft/MeshTransformer/issues/35#issuecomment-1784155692

    @ZhihuaLiuEd Acctually i have solved this issue:

    1. Clone the repo https://github.com/mattloper/opendr.git, and checkout to v0.78
    2. Download the OSMesa.Linux.x86_64.zip from link http://files.is.tue.mpg.de/mloper/opendr/osmesa/OSMesa.Linux.x86_64.zip and put this file in opendr/contexts directory.
    3. Modify the line which import _constants.py to absolute path (from opendr.contexts._constants import *) in opendr/contexts/ctx_base.pyx
    4. Finally run python setup.py build and python setup.py install

## pytorch3d

* install from source if you need to.
* only supports up to torch 2.3.1

```
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d && pip install -e .
```

## gcc

```
conda install -c conda-forge gcc=12.1.0
```

## tensorflow cant find CUDA_DIR
`sudo ln -s /usr/lib/cuda /usr/local/cuda`
