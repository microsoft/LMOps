#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from setuptools import setup

with open("README.md") as f:
    readme = f.read()

setup(
    name="dpr",
    version="1.0.0",
    description="Facebook AI Research Open Domain Q&A Toolkit",
    url="https://github.com/facebookresearch/DPR/",
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    long_description=readme,
    long_description_content_type="text/markdown",
    setup_requires=[
        "setuptools>=18.0",
    ],
    install_requires=[
        "faiss-cpu>=1.6.1",
        "jsonlines",
        "filelock",
        "numpy",
        "regex",
        "torch>=1.5.0",
        "transformers>=4.20.0",
        "tqdm>=4.27",
        "wget",
        "spacy>=2.1.8",
        "hydra-core>=1.0.0",
        "omegaconf>=2.0.1",
    ],
)