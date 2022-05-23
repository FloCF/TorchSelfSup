from setuptools import setup, find_packages

setup(
    name='SSL_pytorch',
    packages=find_packages(exclude=['examples']),
    version='0.0.3',
    license='MIT',
    description = 'SSL_pytorch - PyTorch implementation of different SSL Methods',
    author='Florian C.F. Schulz',
    author_email = 'floriancf.schulz@gmail.com',
    url = 'https://github.com/FloCF/SSL_pytorch',
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