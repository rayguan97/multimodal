conda create --name fbmm python=3.8 -y
cd multimodal
pip install -e .
cd examples
pip install -r flava/requirements.txt