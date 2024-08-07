from abc import ABC, abstractmethod
from node import TreeNode as Node

# Tree base class
class Tree(ABC):
    def __init__(self, root: int):
        super().__init__()
        self.root = root
    
    @abstractmethod
    def return_list(self):
        # returns elements as a list
        pass

    @abstractmethod
    def insert(self, val):
        # inserts new element
        pass

    @abstractmethod
    def delete(self, val):
        # deletes the element
        pass

    @abstractmethod
    def search(self, val):
        # searches the element
        pass

    @abstractmethod
    def inorder(self):
        # inorder traversal
        pass

    @abstractmethod
    def preorder(self):
        # preorder traversal
        pass

    @abstractmethod
    def postorder(self):
        # postorder traversal
        pass


class BinaryTree(Tree):
    def __init__(self, root: int):
        super().__init__(root)
        self.root = root
    
    def return_list(self):
        def _recurse(node):
            if not node:
                return []
            return _recurse(node.left) + [node.value] + _recurse(node.right)
        return _recurse(self.root)
