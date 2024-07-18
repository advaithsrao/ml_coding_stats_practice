# Stacks
class Stack:
    def __init__(self) -> None:
        self.stack = []
    
    def push(self, n: int) -> None:
        self.stack.append(n)
        print(f'Pushed {n}')
    
    def pop(self) -> None:
        print(f"Removed last element {self.stack.pop(-1)}")
    
    def top(self) -> int:
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

