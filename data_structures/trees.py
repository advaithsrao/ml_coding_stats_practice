from data_structures import Tree, TreeNode


class BinaryTree(Tree):
    def __init__(self, root: int):
        super().__init__(root)
        self.root = root
    
    def return_list(self):
        def _recurse(node: TreeNode):
            if not node:
                return []
            return _recurse(node.left) + [node.value] + _recurse(node.right)
        return _recurse(self.root)
