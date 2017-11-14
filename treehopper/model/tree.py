import torch
import numpy as np


class Tree(object):
    def __init__(self):
        self.parent = None
        self.num_children = 0
        self.children = list()
        self.gold_label = None
        self.output = None

    def add_child(self, child):
        child.parent = self
        self.children.append(child)
        self.num_children += 1

    def size(self):
        if getattr(self, '_size'):
            return self._size
        if self.num_children == 0:
            self._size = 1
        else:
            self._size = sum(child.size() for child in self.children) + 1
        return self._size

    def depth(self):
        if getattr(self, '_depth'):
            return self._depth
        if self.num_children == 0:
            self._depth = 0
        else:
            self._depth = max(child.depth() for child in self.children) + 1
        return self._depth

    def compute_accuracy(self):

        def _compute_accuracy(tree, accuracies=None):
            """
            Recursively compute accuracies for every subtree
            """
            if accuracies is None:
                accuracies = []
            accuracies.append(1 if tree.get_output() == tree.gold_label else 0)
            for subtree in tree.children:
                _compute_accuracy(subtree, accuracies)
            return accuracies

        total_accuracies = _compute_accuracy(self)
        return np.mean(total_accuracies)

    def list_children_in_order(self):
        assert self.parent is None, 'This method should only be called on ' \
                                    'root nodes'
        tree_list = []

        def traverse(tree, tree_list):
            tree_list.append(tree)
            for subtree in tree.children:
                traverse(subtree, tree_list)
            return tree_list

        final_tree_list = traverse(self, tree_list)
        return sorted(final_tree_list, key=lambda tree: tree.idx)

    def get_predicted_labels(self):
        assert self.parent is None, 'This method should only be called on ' \
                                    'root nodes'

        return ' '.join(str(tree.get_output() - 1)
                        for tree in self.list_children_in_order())

    def get_output(self):
        assert self.output is not None, 'No predicted label'
        return torch.max(self.output, 1,keepdim=True)[1].data.cpu().numpy()[0][0]
