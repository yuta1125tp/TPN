# Installation

```shell
git clone https://github.com/decisionforce/TPN.git
```

## Requirements

- Linux
    - Python 3.5+
    - PyTorch 1.0+
    - CUDA 9.0+
    - NVCC 2+
    - GCC 4.9+
    - mmcv 0.2.10
- Windows (only checked CPU)
    - Python 3.5+
    - pytorch 1.3.1
    - torchvision 0.4.2
    - mmcv 0.2.10
    - opencv_python
    - cython
    - moviepy
    - future
    - easydict
    ```bat
    pip install torch===1.3.1 torchvision===0.4.2 -f https://download.pytorch.org/whl/torch_stable.html
    pip install mmcv==0.2.10 opencv_python cython moviepy future easydict
    ```

## Install MMAction
(a) Install Cython
```shell
pip install cython
```
(b) Install mmaction
```shell
python setup.py develop
```

