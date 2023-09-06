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
    install_requires=["numpy", "tqdm", "torch", "PyYAML"],
    entry_points={
        'console_scripts': [
            "repcodec=repcodec.tokenize:cli"
        ]
    }
)
