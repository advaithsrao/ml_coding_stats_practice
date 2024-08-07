from abc import ABC, abstractmethod

# Node base class
class Node(ABC):
    def __init__(self, val):
        self.value = val

class LLNode(Node):
    def __init__(self, val):
        super().__init__(val)
        self.value = val
        self.next = None

class TreeNode(Node):
    def __init__(self, val):
        super().__init__(val)
        self.value = val
        self.left = None
        self.right = None