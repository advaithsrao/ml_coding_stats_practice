import pytest
from data_structures import Node, LLNode, TreeNode
from data_structures.arrays import Stack, Queue, Deque
from data_structures.arrays import parenthesis_match
from data_structures.linked_lists import SinglyLinkedList, DoublyLinkedList

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

# queues
def test_queue(num_inputs):
    _queue = Queue()
    
    for _input in num_inputs[:5]:
        _queue.enqueue(_input)
    
    assert _queue.front == 0
    assert _queue.rear == 5
    assert _queue.size == 5
    assert _queue.capacity == 8

    for _input in range(2):
        _queue.dequeue()
    
    assert _queue.front == 2
    assert _queue.rear == 5
    assert _queue.size == 3
    assert _queue.capacity == 8

    for _input in num_inputs[5:]:
        _queue.enqueue(_input)
    
    assert _queue.front == 2
    assert _queue.rear == 1
    assert _queue.size == 7
    assert _queue.capacity == 8
    
    assert not _queue.is_empty()

# deques
def test_deque(num_inputs):
    _deque = Deque()
    
    assert _deque.is_empty()

    for _input in num_inputs[:5]:
        _deque.addFront(_input)
    
    assert _deque.items[-1] == 1
    assert _deque.items[0] == 5

    for _ in range(5):
        _deque.removeRear()
    
    assert _deque.is_empty()
    
    for _input in num_inputs[:5]:
        _deque.addRear(_input)
    
    assert _deque.items[-1] == 5
    assert _deque.items[0] == 1

# linked lists
## singly linked list
def test_singly_linked_list():
    nodes = [LLNode(i) for i in range(5)]
    sll = SinglyLinkedList()
    sll.head = nodes[0]
    
    for node_num in range(0,len(nodes)-1):
        nodes[node_num].next = nodes[node_num+1]

    assert sll.return_list() == [0,1,2,3,4]
    
    sll.delete_from_beginning()
    assert sll.return_list() == [1,2,3,4]
    
    sll.insert_at_beginning(LLNode(-1))
    assert sll.return_list() == [-1,1,2,3,4]
    
    sll.insert_at_end(LLNode(5))
    assert sll.return_list() == [-1,1,2,3,4,5]
    
    sll.insert_at_position(LLNode(-3), 2)
    assert sll.return_list() == [-1,1,-3,2,3,4,5]
    
    sll.insert_after_element(LLNode(-2), nodes[1])
    assert sll.return_list() == [-1,1,-2,-3,2,3,4,5]
    
    sll.delete_from_end()
    assert sll.return_list() == [-1,1,-2,-3,2,3,4]
    
    sll.delete_from_position(2)
    assert sll.return_list() == [-1,1,-3,2,3,4]
    
    sll.delete_after_element(nodes[1])
    assert sll.return_list() == [-1,1,2,3,4]

## doubly linked list
def test_doubly_linked_list():
    nodes = [LLNode(i) for i in range(5)]
    dll = DoublyLinkedList()
    dll.head = nodes[0]
    
    for node_num in range(0,len(nodes)-1):
        nodes[node_num].next = nodes[node_num+1]
        nodes[node_num+1].prev = nodes[node_num]

    assert dll.return_list() == [0,1,2,3,4]
    
    dll.delete_from_beginning()
    assert dll.return_list() == [1,2,3,4]
    
    dll.insert_at_beginning(LLNode(-1))
    assert dll.return_list() == [-1,1,2,3,4]
    
    dll.insert_at_end(LLNode(5))
    assert dll.return_list() == [-1,1,2,3,4,5]
    
    dll.insert_at_position(LLNode(-3), 2)
    assert dll.return_list() == [-1,1,-3,2,3,4,5]
    
    dll.insert_after_element(LLNode(-2), nodes[1])
    assert dll.return_list() == [-1,1,-2,-3,2,3,4,5]
    
    dll.delete_from_end()
    assert dll.return_list() == [-1,1,-2,-3,2,3,4]
    
    dll.delete_from_position(2)
    assert dll.return_list() == [-1,1,-3,2,3,4]
    
    dll.delete_after_element(nodes[1])
    assert dll.return_list() == [-1,1,2,3,4]
