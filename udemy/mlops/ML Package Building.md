

# finally what we need to do before packing up


### 1. create testing files
using pytest fixtures


### 2. create manifest file

MANIFEST.in

consists of commands , one per line , instructing setuptools to add or remove some set of files from the sdist

``` MANIFEST.in
include *.txt #requirements.txt
include *.md #README
include *.py

recursive-include ./path_to_dir/*

include ./path_to_dir/datasets/*.csv
include ./path_to_dir/models/*.pkl
include ./path_to_dir/models/VERSION

include ./tests/*


recursive-exclude * __pycache__
recursive-exclude * *.py[co]
```

### 3. create Version file


create a VERSION file
```VERSION

1.0.0 # major.minor.micro

```


 ```
inside "__init__.py"
```

```python

import os

from X.config import config

with open(os.path.join(config.PACKAGE_ROOT , VERSION)) as f:
	__version__ = f.read().strip()

```
### 4. create setup.py


using setuptools

in setup.py



```python

from setuptools import setup , find_packages
import os
import io
from pathlib import Path


NAME = "XYZ"
DESCRIPTION = "lorem ipsum"
URL = "https://github.com/USER"
EMAIL = "USER@ORG.com"
AUTHOR = "FIRSTNAME LASTNAME"
REQUIRES_PYTHON = '>=3.x.a'

pwd = os.path.abspath(os.path.dirname(__file__)) # 



#get list of other dependencies to be installed

def list_reqs(fname = 'requirements.txt'):
	with io.open(os.path.join(pwd , fname) , encoding = 'utf-8') as f:
		return f.read().splitlines()

try:
	with io.open(os.path.join(pwd,'README.md') , encoding = 'utf-8') as f:
		long_description = '\n' + f.read()

except FileNotFoundError:
	long_description = DESCRIPTION

ROOT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = ROOT_DIR / NAME
about = {}

with open(PACKAGE_DIR / 'VERSION') as f:
	_version = f.read().strip()
	about['__version__'] = _version


setup(
	name = NAME,
	version = about['__version__']
	description = DESCRIPTION,
	long_description = long_description , 
	author = AUTHOR , 
	author_email = EMAIL,
	python_requires = REQUIRES_PYTHON,
	url = URL,
    packages = find_packages(exclude = ('tests' , )),
    package_data = {'X':['VERSION']},
    install_requires = list_reqs(),
    extras_require = {},
	include_package_data = True,
	license = 'MIT',
	classifiers = [
	'Lisence :: OSI Approved :: MIT License',
	'Programming Language :: Python',
	'Programming Language :: Python :: 3',
	'Programming Language :: Python :: 3.x',
	'Programming Language :: Python :: Implementation :: CPython',
	'Programming Language :: Python :: Implementation :: PyPy',
	
	],


)


```
