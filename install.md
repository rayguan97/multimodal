conda create --name fbmm python=3.8 -y
cd multimodal
pip install -e .
cd examples
pip install -r flava/requirements.txt
pip install mat4py IPython

huggingface-cli login



data:
SUN RGB-D: https://rgbd.cs.princeton.edu/

