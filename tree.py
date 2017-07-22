import numpy as np
import torch

class Tree(object):
    def __init__(self):
        self.parent = None
        self.num_children = 0
        self.children = list()
        self.gold_label = None # node label for SST
        self.output = None # output node for SST

        # helper for visualization
        self._viz_all_children = None
        self._viz_sentence = None
        self._viz_relations = None
        self._viz_labels = None

    def add_child(self,child):
        child.parent = self
        self.children.append(child)
        self.num_children += 1

    def size(self):
        if getattr(self,'_size'):
            return self._size

        count = 1
        for i in range(len(self.children)):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        if getattr(self,'_depth'):
            return self._depth

        count = 0
        for i in range(len(self.children)):
            child_depth = self.children[i].depth()
            if child_depth>count:
                count = child_depth
        count += 1

        self._depth = count
        return self._depth

    def visualize(self, with_output=False):

        root = [{'text': 'root', 'tag': get_label(self, with_output)}]
        root_arc = {
            'start': 0,
            'end': self.idx,
            'dir': 'left',
            'label': self.relation
        }
        sentence = {
            'words': root + [{'text': subtree.word, 'tag': get_label(subtree, with_output)}
                             for _, subtree in sorted(self._viz_all_children.items())],
            'arcs': get_arcs(self) + [root_arc]
        }
        return sentence

    def compute_accuracy(self):

        def _compute_accuracy(tree, accuracies=None):
            """
            Recursively compute accuracies for every subtree
            """
            if accuracies is None:
                accuracies = []
            accuracies.append(1 if torch.max(tree.output, 1)[1].data.numpy()[0][0] == tree.gold_label else 0)
            for subtree in tree.children:
                _compute_accuracy(subtree, accuracies)
            return accuracies

        total_accuracies = _compute_accuracy(self)
        return np.mean(total_accuracies)


def get_arcs(tree, arcs_list=None):
    if arcs_list is None:
        arcs_list = []
    for subtree in tree.children:
        arc = {
            'start': subtree.idx,
            'end': tree.idx,
            'dir': 'left' if subtree.idx < tree.idx else 'right',
            'label': tree.relation
        }
        arcs_list.append(arc)
        get_arcs(subtree, arcs_list)
    return arcs_list

def get_label(tree, with_output=False):
    if with_output:
        if not tree.output:
            print('no ej')
            print(tree.children)
            print(tree.word)
            print(tree.gold_label)
        output_label = torch.max(tree.output, 1)[1].data.numpy()[0][0]-1

        sentiment_label = 'should {}/is {}'.format(
            str(tree.gold_label - 1), output_label)
    else:
        sentiment_label = str(tree.gold_label - 1)
    return sentiment_label





