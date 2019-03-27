from setuptools import setup, find_packages

setup(
    name="MyGAN",
    version="0.1",
    packages=find_packages(),
    install_requires=['numpy>=1.15.4',
                      'pandas>=0.23.4',
                      'scikit-learn>=0.20',
                      'matplotlib>=3.0.1',
                      'pillow>=3.0',
                      'tensorflow>=1.12']
)
