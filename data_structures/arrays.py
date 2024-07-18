# Contents
# 1. Stacks
#     a. Parenthesis example
# 2. Queues

# Stacks
class Stack:
    def __init__(self) -> None:
        self.stack = []
    
    def push(self, n: int) -> None:
        self.stack.append(n)
        print(f'Pushed {n}')
    
    def pop(self) -> None:
        if self.is_empty():
            raise ValueError
        print(f"Removed last element {self.stack.pop(-1)}")
    
    def top(self) -> int:
        if self.is_empty():
            raise ValueError
        
        return self.stack[-1]
    
    def is_empty(self) -> bool:
        return not bool(self.stack)

# Example of stack
# In an expression with parenthesis, do the parenthesis resolve properly
def parenthesis_match(expression: str) -> bool:
    lefts = "{[("
    rights = "}])"
    
    _stack = Stack()

    for char in expression:
        if char in lefts:
            _stack.push(char)
        elif char in rights:
            if _stack.is_empty():
                return False
            elif rights.index(char) == lefts.index(_stack.top()):
                _stack.pop()
            else:
                return False
        else:
            return False
    
    return _stack.is_empty()

# Queues
class Queue:
    def __init__(self):
        self.front = self.rear = self.size = 0
        self.capacity = 4 #default
        self.queue = [None] * self.capacity
    
    def is_empty(self) -> bool:
        return not bool(self.queue)

    def first(self) -> int:
        return self.queue[self.front]
    
    def length(self) -> int:
        return self.size

    def enqueue(self, n: int) -> None:
        self.size += 1
        if self.size == self.capacity:
            self.resize()
        self.queue[self.rear] = n
        self.rear = (self.rear + 1) % self.capacity
        print(f'Added {n}')
    
    def dequeue(self) -> None:
        last = self.queue[self.front]
        self.queue[self.front] = None
        self.size -= 1
        self.front = (self.front + 1) % self.capacity
        print(f'Removed {last}')

    def resize(self) -> None:
        self.queue += [None] * self.capacity
        self.capacity *= 2


if __name__ == '__main__':
    # Stacks
    print("="*10)
    print("STACKS")
    s = Stack()
    s.push(1)
    s.push(2)
    s.push(3)
    s.push(4)
    s.push(5)
    s.pop()
    s.pop()
    s.pop()
    s.pop()
    s.pop()
    print("="*10)

    # Parenthesis example
    print("="*10)
    print("Parenthesis Example")
    print("Match for {{[([{}])]}}:", parenthesis_match('{{[([{}])]}}'))
    print("Match for {{[[{])]}:", parenthesis_match('{{[[{])]}'))
    print("="*10)

    
    # Queues
    print("="*10)
    print("QUEUES")
    q = Queue()
    q.enqueue(1)
    q.enqueue(2)
    q.enqueue(3)
    q.enqueue(4)
    q.enqueue(5)
    print('\n')
    "=== After Last Enqueue ==="
    print(f"Queue: {q.queue}")
    print(f"Size: {q.size}")
    print(f"Capacity: {q.capacity}")
    print(f"Front: {q.front}")
    print(f"Rear: {q.rear}")
    "=== After first Dequeue ==="
    q.dequeue()
    print(f"Queue: {q.queue}")
    print(f"Size: {q.size}")
    print(f"Capacity: {q.capacity}")
    print(f"Front: {q.front}")
    print(f"Rear: {q.rear}")
    print('\n')
    q.dequeue()
    q.dequeue()
    q.dequeue()
