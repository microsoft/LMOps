# Evaluator for Break dataset on beaker

FROM python:3.7.6-slim-buster

ENV PYTHONPATH .

# set the working directory

WORKDIR /break-evaluator


# install python packages

ADD ./requirements.txt .

RUN pip3.7 install -r requirements.txt
RUN python3.7 -m spacy download en_core_web_sm


# add in the readme and evaluation scripts

ADD README.md .
ADD allennlp_preds_format.py .
COPY evaluation ./evaluation
COPY scripts ./scripts
COPY utils ./utils

RUN mkdir /results


# define the default command
# in this case a linux shell where we can run the eval script
CMD ["/bin/bash"]
