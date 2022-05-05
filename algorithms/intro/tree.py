# binary tree

# node
class Node:
    def __init__(self, data) -> None:
        self.parent = None
        self.left = None
        self.right = None
        self.data = data
        self.size = 1

    def __repr__(self) -> str:
        print(self.data)

    def __str__(self) -> str:
        return "asdf"

    def insert(self, node):
        if node.data <= self.data:
            node.right = self
            self.left = node
        else:
            node.left = self
            self.right = node


if __name__ == "__main__":
    print("start")
