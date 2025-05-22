import ast
import os
import re
from pathlib import Path

from setuptools import find_packages, setup

with open('README.md') as f:
    long_description = f.read()

# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))

PACKAGE_NAME = 'mmfreelm'

def get_package_version():
    with open(Path(this_dir) / 'mmfreelm' / '__init__.py') as f:
        version_match = re.search(r"^__version__\s*=\s*(.*)$", f.read(), re.MULTILINE)
    return ast.literal_eval(version_match.group(1))


setup(
    name=PACKAGE_NAME,
    version=get_package_version(),
    description='Implementation for Matmul-free LM',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='',
    author_email='',
    url='',
    packages=find_packages(),
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    python_requires='>=3.7',
    install_requires=[
        'transformers',
        'einops',
        'ninja'
    ]
)