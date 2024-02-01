from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='fasthgp',
    version='0.1',
    description='',
    author='--',
    author_email='',
    url='',
    packages=find_packages(),
    install_requires=requirements,
)
