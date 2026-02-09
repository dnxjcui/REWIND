#!/bin/bash

export CONDA_ENV_NAME=dynhamr
conda init
source activate $CONDA_ENV_NAME

cd ~/Desktop/rewind/Dyn-HaMR

# export CUDAHOSTCXX="C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64\cl.exe"
