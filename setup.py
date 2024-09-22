from setuptools import find_packages, setup

setup(
    name='radgraph',
    version='0.1.4',
    author='Jean-Benoit Delbrouck',
    license='MIT',
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3 :: Only',
    ],
    install_requires=['torch',
                      'transformers',
                      'appdirs',
                      'jsonpickle',
                      'filelock',
                      'h5py',
                      'spacy',
                      'nltk',
                      'dotmap',
                      'pytest',
                      ],
    packages=find_packages(),
    zip_safe=False)
