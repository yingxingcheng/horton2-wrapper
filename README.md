# horton2-wrapper
Copyright (C) 2020-2023 Yingxing Cheng

A horton2 wrapper for generating inputs files for the ACKS2⍵ model.

## Modules

- `srfunctionals`: The exchange-correlation functional for short-range DFT.


## Installation

You can install `horton2-wrapper` using pip or by cloning the GitHub repository.


### Install from pip

```bash
pip install git+https://github.com/yingxingcheng/horton2-wrapper.git
```

### Install from github (for developer)

```bash
git clone git@github.com:yingxingcheng/horton2-wrapper.git
cd horton2-wrapper
pip install [-e] .
```

### Dependencies
The installation process will automatically handle the following dependencies:

- horton>=2.1.0
- progress>=1.5
- numpy>=1.16.3
- scipy>=1.2.1
- pytest>=4.6.11