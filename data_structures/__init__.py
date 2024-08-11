from abc import ABC, abstractmethod

# Node base class
class Node(ABC):
    def __init__(self, val):
        # No need to call super().__init__() here
        self.value = val


class LLNode(Node):
    def __init__(self, val):
        # Initialize the Node base class with the value
        super().__init__(val)
        self.next = None


class TreeNode(Node):
    def __init__(self, val):
        # Initialize the Node base class with the value
        super().__init__(val)
        self.left = None
        self.right = None


# Abstract Base Class for Linked Lists
class LinkedList(ABC):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def return_list(self) -> list:
        # returns elements as a list
        pass

    @abstractmethod
    def insert_at_beginning(self, node: LLNode):
        # inserts new first element
        pass

    @abstractmethod
    def insert_at_end(self, node: LLNode):
        # inserts new end element
        pass
    
    @abstractmethod
    def insert_at_position(self, node: LLNode, pos: int):
        # inserts new node at Position
        # Position is the index here
        pass

    @abstractmethod
    def insert_after_element(self, node: LLNode, element: LLNode):
        # insert after element
        pass

    @abstractmethod
    def delete_from_beginning(self):
        # deletes the first element
        pass
    
    @abstractmethod
    def delete_from_end(self):
        # deletes the end element
        pass
    
    @abstractmethod
    def delete_from_position(self, pos: int):
        # deletes the a given position
        pass

    @abstractmethod
    def delete_after_element(self, element: LLNode):
        # deletes after element
        pass


# Abstract base class for Trees
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
