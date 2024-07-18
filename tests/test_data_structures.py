import pytest

from data_structures.arrays import Stack
from data_structures.arrays import parenthesis_match

@pytest.fixture
def num_inputs():
    return [a for a in range(1, 10)]

@pytest.fixture
def par_inputs():
    return [
        '{{[([{}])]}}',
        '{{[[{])]}'
    ]

# arrays
## stacks
def test_stack(num_inputs):
    _stack = Stack()
    
    for _input in num_inputs:
        _stack.push(_input)
    
    assert _stack.top() == 9

    for _ in range(1, 10):
        _stack.pop()
    
    assert _stack.is_empty() == True

## stack parenthesis example
def test_stack_parenthesis_example(par_inputs):
    outputs = []
    for _input in par_inputs:
        outputs.append(parenthesis_match(_input))
    
    assert outputs[0] == True
    assert outputs[1] == False