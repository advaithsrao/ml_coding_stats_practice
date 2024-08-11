from data_structures import LinkedList, LLNode


# LLs
class SinglyLinkedList(LinkedList):
    def __init__(self):
        self.head = None
        super().__init__()
    
    def return_list(self):
        result = []
        cursor = self.head
        print(f'Head: {cursor.value}')
        
        while cursor != None:
            result.append(cursor.value)
            cursor = cursor.next
        print(f'List: {result}')
        return result

    def insert_at_beginning(self, node: LLNode):
        cursor = self.head
        self.head = node
        node.next = cursor
        print(f'Inserted at beginning: {self.head.value}')
    
    def insert_at_end(self, node):
        cursor = self.head
        while cursor.next != None:
            cursor = cursor.next
        cursor.next = node
        node.next = None
        print(f'Inserted at end: {node.value}')
    
    def insert_at_position(self, node: LLNode, pos: int):
        cursor = self.head
        index = 0
        
        while index + 1 != pos and cursor != None:
            cursor = cursor.next
            index += 1
        
        if cursor == None:
            raise IndexError
        
        old = cursor.next
        cursor.next = node
        node.next = old

        print(f'Inserted at {pos}: {node.value}')
    
    def insert_after_element(self, node: LLNode, element: LLNode):
        cursor = self.head
        
        while cursor != element and cursor != None:
            cursor = cursor.next
        
        if cursor == None:
            raise IndexError
        
        old = cursor.next
        cursor.next = node
        node.next = old

        print(f'Inserted after {element.value}: {node.value}')
    
    def delete_from_beginning(self):
        print(f'Deleted at beginning: {self.head.value}')
        self.head = self.head.next
    
    def delete_from_end(self):
        cursor = self.head
        while cursor.next.next != None:
            cursor = cursor.next
        print(f'Deleted at end: {cursor.next.value}')
        cursor.next = None
    
    def delete_from_position(self, pos: int):
        cursor = self.head
        index = 0
        while index+1 != pos and cursor != None:
            cursor = cursor.next
            index += 1
        
        if cursor == None:
            raise IndexError
        
        old = cursor.next
        cursor.next = cursor.next.next
        print(f'Deleted at {pos}: {old.value}')
    
    def delete_after_element(self, element: LLNode):
        cursor = self.head
        while cursor != element and cursor != None:
            cursor = cursor.next
        if cursor == None:
            raise IndexError
        old = cursor.next
        cursor.next = cursor.next.next
        print(f'Deleted after {element.value}: {old.value}')


# DLL
class DoublyLinkedList(LinkedList):
    def __init__(self):
        self.head = None
        self.tail = None
        super().__init__()

    def return_list(self):
        result = []
        cursor = self.head
        print(f'Head: {cursor.value}')
        
        while cursor != None:
            result.append(cursor.value)
            cursor = cursor.next
        print(f'List: {result}')
        return result

    def insert_at_beginning(self, node: LLNode):
        cursor = self.head
        self.head = node
        node.next = cursor
        cursor.prev = node
        node.prev = None
        print(f'Inserted at beginning: {self.head.value}')
    
    def insert_at_end(self, node: LLNode):
        cursor = self.head
        while cursor.next != None:
            cursor = cursor.next
        cursor.next = node
        node.prev = cursor
        node.next = None
        print(f'Inserted at end: {node.value}')
    
    def insert_at_position(self, node: LLNode, pos: int):
        cursor = self.head
        index = 0
        
        while index + 1 != pos and cursor != None:
            cursor = cursor.next
            index += 1
        
        if cursor == None:
            raise IndexError
        
        old = cursor.next
        cursor.next = node
        node.prev = cursor
        node.next = old
        old.prev = node

        print(f'Inserted at {pos}: {node.value}')
    
    def insert_after_element(self, node: LLNode, element: LLNode):
        cursor = self.head
        
        while cursor != element and cursor != None:
            cursor = cursor.next
        
        if cursor == None:
            raise IndexError
        
        old = cursor.next
        cursor.next = node
        node.prev = cursor
        node.next = old
        old.prev = node

        print(f'Inserted after {element.value}: {node.value}')
    
    def delete_from_beginning(self):
        print(f'Deleted at beginning: {self.head.value}')
        self.head = self.head.next
        self.head.prev = None
    
    def delete_from_end(self):
        cursor = self.head
        while cursor.next.next != None:
            cursor = cursor.next
        print(f'Deleted at end: {cursor.next.value}')
        cursor.next = None
        self.tail = cursor
    
    def delete_from_position(self, pos: int):
        cursor = self.head
        index = 0
        while index+1 != pos and cursor != None:
            cursor = cursor.next
            index += 1
        
        if cursor == None:
            raise IndexError
        
        old = cursor.next
        cursor.next = cursor.next.next
        cursor.next.prev = cursor
        print(f'Deleted at {pos}: {old.value}')
    
    def delete_after_element(self, element: LLNode):
        cursor = self.head
        while cursor != element and cursor != None:
            cursor = cursor.next
        if cursor == None:
            raise IndexError
        old = cursor.next
        cursor.next = cursor.next.next
        cursor.next.prev = cursor
        print(f'Deleted after {element.value}: {old.value}')



if __name__ == '__main__':
    nodes = [LLNode(i) for i in range(1,5)]

    #singly ll
    print("="*10)
    print("Singly LL")
    sll = SinglyLinkedList()
    sll.head = nodes[0]
    
    # set next pointers
    for node_num in range(0,len(nodes)-1):
        nodes[node_num].next = nodes[node_num+1]
    
    # print list
    print('Before deleting head')
    _ = sll.return_list()

    sll.delete_from_beginning()

    # print list
    print('After deleting head')
    _ = sll.return_list()

    sll.insert_at_beginning(LLNode(-1))

    print('After inserting new head')
    _ = sll.return_list()

    sll.insert_at_end(LLNode(5))

    print('After inserting new end')
    _ = sll.return_list()

    sll.insert_at_position(LLNode(-3), 2)

    print('After inserting at position 2')
    _ = sll.return_list()

    sll.insert_after_element(LLNode(-2), nodes[1])

    print('After inserting after 2')
    _ = sll.return_list()

    sll.delete_from_end()

    print('After deleting end')
    _ = sll.return_list()

    sll.delete_from_position(2)

    print('After deleting at position 2')

    _ = sll.return_list()

    sll.delete_after_element(nodes[1])

    print('After deleting after 2')

    _ = sll.return_list()

    print("="*10)

    #doubly ll
    print("="*10)
    print("Doubly LL")

    dll = DoublyLinkedList()

    dll.head = nodes[0]
    dll.tail = nodes[-1]

    # set next pointers
    for node_num in range(0,len(nodes)-1):
        nodes[node_num].next = nodes[node_num+1]
        nodes[node_num+1].prev = nodes[node_num]

    # print list
    print('Before deleting head')
    _ = dll.return_list()

    dll.delete_from_beginning()

    # print list
    print('After deleting head')
    _ = dll.return_list()

    dll.insert_at_beginning(LLNode(-1))
    
    print('After inserting new head')
    _ = dll.return_list()

    dll.insert_at_end(LLNode(5))

    print('After inserting new end')
    _ = dll.return_list()

    dll.insert_at_position(LLNode(-3), 2)

    print('After inserting at position 2')
    _ = dll.return_list()

    dll.insert_after_element(LLNode(-2), nodes[1])

    print('After inserting after 2')
    _ = dll.return_list()

    dll.delete_from_end()

    print('After deleting end')
    _ = dll.return_list()

    dll.delete_from_position(2)

    print('After deleting at position 2')
    _ = dll.return_list()

    dll.delete_after_element(nodes[1])

    print('After deleting after 2')
    _ = dll.return_list()

    print("="*10)
