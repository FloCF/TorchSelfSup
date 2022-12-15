from setuptools import setup, find_packages

setup(
    name='TorchSelfSup',
    packages=find_packages(exclude=['data', 'examples', 'results']),
    version='0.0.1',
    license='MIT',
    description = 'TorchSelfSup - PyTorch implementation of different Self-Supervised Learning Methods',
    author='Florian C.F. Schulz',
    author_email = 'floriancf.schulz@gmail.com',
    url = 'https://github.com/FloCF/TorchSelfSup',
    keywords = [
        'self supervised learning',
        'deep learning',
        'unsupervised learning',
        'representational learning'
    ],
    install_requires=[
        'torch>1.9',
        'torchvision>0.10',
    ],
)