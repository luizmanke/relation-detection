import numpy as np


class Node:

    def __init__(self) -> None:
        self.parent = None
        self.n_children = 0
        self.children: list = []

    def add_child(self, child) -> None:
        child.parent = self
        self.n_children += 1
        self.children.append(child)


def head_to_tree(heads, length):

    heads = heads[:length].tolist()
    nodes = [Node() for _ in heads]
    for i in range(len(nodes)):
        head = heads[i]
        nodes[i].idx = i
        nodes[i].dist = -1  # just a filler
        if head == 0:
            root = nodes[i]
        else:
            nodes[head-1].add_child(nodes[i])

    return root


def tree_to_matrix(tree, length):

    indexes = []
    queue = [tree]
    matrix = np.zeros((length, length), dtype=np.float32)
    while len(queue) > 0:
        current_tree, queue = queue[0], queue[1:]
        indexes += [current_tree.idx]

        for child in current_tree.children:
            matrix[current_tree.idx, child.idx] = 1
        queue += current_tree.children

    matrix = matrix + matrix.T
    return matrix.reshape(1, length, length)
