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
        self.num_children += 1
        self.children.append(child)

    def size(self):
        if getattr(self,'_size'):
            return self._size
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        if getattr(self,'_depth'):
            return self._depth
        count = 0
        if self.num_children>0:
            for i in range(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth>count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth

    def visualize(self):
        root = [{'text': 'root', 'tag': self.gold_label-1}]
        root_arc = {
            'start': 0,
            'end': self.idx,
            'dir': 'left',
            'label': self.relation
        }
        sentence = {
            'words': root + [{'text': subtree.word, 'tag': subtree.gold_label-1}
                             for subtree in self._viz_all_children.values()],
            'arcs': get_arcs(self) + [root_arc]
        }
        return sentence


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





