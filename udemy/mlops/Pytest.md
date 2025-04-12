File : test_*.py or *_test.py
Class Naming : Test* * 
Method Naming : test_

in the test_*.py -> import pytest

create testing class or method

use assert 

pytest.fail() <- alert

### pytest fixtures


test_X.py
``` python
import pytest
from X import a , b , c

@pytest.fixture

def X_setup():
	print("Setting up env for X")
	return {}

def test_a(X_setup):
	assert a(arg1 , arg1) == Value

def test_b(X_setup):
	assert b(arg1 , arg1) == Value

def test_c(X_setup):
	assert type(c(arg1 , arg1)) == Datatype


```

``` bash

pytest test_X.py

//or

pytest test_X.py -s // display messages we wanted to print like for settting up etc

pytest test_X.py -v // display additional messages


```