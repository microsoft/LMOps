# git clone https://github.com/hkust-nlp/simpleRL-reason.git
# cd simpleRL-reason/eval
cd latex2sympy
pip install -e .
cd ..
pip install -r requirements.txt --user
pip install vllm==0.5.1 --no-build-isolation
pip install transformers==4.49.0