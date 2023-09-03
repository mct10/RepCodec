from setuptools import setup, find_packages

try:
    with open("README.md") as fp:
        long_description = fp.read()
except Exception:
    long_description = ""

setup(
    name="RepCodec",
    description="A Speech Representation Codec for Speech Tokenization",
    long_description=long_description,
    url="https://github.com/mct10/RepCodec",
    packages=find_packages(),
    install_requires=["numpy", "torch"],
    entry_points={
        'console_scripts': [
            "repcodec=repcodec.tokenize:cli"
        ]
    }
)
