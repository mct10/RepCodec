# Copyright (c) ByteDance, Inc. and its affiliates.
# Copyright (c) Chutong Meng
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup

try:
    with open("README.md") as fp:
        long_description = fp.read()
except Exception:
    long_description = ""

setup(
    name="RepCodec",
    version="v1.0.0",
    description="A Speech Representation Codec for Speech Tokenization",
    long_description=long_description,
    url="https://github.com/mct10/RepCodec",
    packages=["repcodec", "repcodec.modules", "repcodec.layers"],
    package_data={
        "repcodec": ["configs/*.yaml"]
    },
    install_requires=["numpy", "tqdm", "torch", "tensorboardX", "PyYAML"],
    entry_points={
        'console_scripts': [
            "repcodec=repcodec.tokenize:cli"
        ]
    }
)
