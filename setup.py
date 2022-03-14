from setuptools import setup, find_packages

setup(
    name="PyTorchSSL",
    version="0.1",
    author='Florian C.F. Schulz',
    packages=find_packages(),
    install_requires=[
        'torch>1.9',
        'torchvision>0.10',
        'numpy',
    ],
)