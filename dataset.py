import os
from copy import deepcopy
from tqdm import tqdm
import torch
import torch.utils.data as data
from tree import Tree
from vocab import Vocab
import constants
import utils


# Dataset class for SICK dataset
class SSTDataset(data.Dataset):
    def __init__(self, path=None, vocab=None, num_classes=None, fine_grain=None, model_name=None):
        super(SSTDataset, self).__init__()

        self.num_classes = num_classes
        if not path and not vocab:
            return

        self.vocab = vocab
        self.num_classes = num_classes
        self.fine_grain = fine_grain
        self.model_name = model_name

        skladnica_sentences = self.read_sentences(os.path.join(path, 'sklad_sentence.txt'))
        reviews_sentences = self.read_sentences(os.path.join(path, 'rev_sentence.txt'))
        self.sentences = skladnica_sentences + reviews_sentences

        skladnica_trees = self.read_trees(
            filename_parents=os.path.join(path, 'sklad_parents.txt'),
            filename_labels=os.path.join(path, 'sklad_labels.txt'),
            filename_tokens=os.path.join(path, 'sklad_sentence.txt'),
            filename_relations=os.path.join(path, 'sklad_rels.txt'),
        )

        reviews_trees = self.read_trees(
            filename_parents=os.path.join(path, 'rev_parents.txt'),
            filename_labels=os.path.join(path, 'rev_labels.txt'),
            filename_tokens=os.path.join(path, 'rev_sentence.txt'),
            filename_relations=os.path.join(path, 'rev_rels.txt'),
        )

        self.trees = skladnica_trees + reviews_trees  # list concatenation
        self.labels = []

        for i in range(0, len(self.trees)):
            self.labels.append(self.trees[i].gold_label)
        self.labels = torch.Tensor(self.labels)  # let labels be tensor

        # shuffle
        from sklearn.utils import shuffle
        self.trees, self.sentences, self.labels = shuffle(self.trees, self.sentences, self.labels)

    def __len__(self):
        return len(self.trees)

    def __getitem__(self, index):
        tree = deepcopy(self.trees[index])
        sent = deepcopy(self.sentences[index])
        label = deepcopy(self.labels[index])
        return (tree, sent, label)

    def read_sentences(self, filename):
        with open(filename,'r', encoding='utf-8') as f:
            sentences = [self.read_sentence(line) for line in tqdm(f.readlines())]
        return sentences

    def read_sentence(self, line):
        indices = self.vocab.convertToIdx(line.split(), constants.UNK_WORD)
        return torch.LongTensor(indices)

    def read_trees(self, filename_parents, filename_labels, filename_tokens, filename_relations):
        parents_file = open(filename_parents, 'r', encoding='utf-8') # parent node
        labels_file = open(filename_labels, 'r', encoding='utf-8') # label of a node
        tokens_file = open(filename_tokens, 'r', encoding='utf-8')
        relations_file = open(filename_relations, 'r', encoding='utf-8')
        iterator = zip(parents_file.readlines(), labels_file.readlines(),
                       tokens_file.readlines(), relations_file.readlines())
        trees = [self.read_tree(parents, labels, tokens, relations)
                 for parents, labels, tokens, relations in tqdm(iterator)]

        return trees

    def parse_label(self, label):
        return int(label) + 1

    def read_tree(self, line_parents, line_label, line_words, line_relations):
        parents = list(map(int, line_parents.split()))
        labels = list(map(self.parse_label, line_label.split()))
        words = line_words.split()
        relations = line_relations.split()
        trees = dict()
        root = None

        for i in range(1, len(parents)+1):
            if i not in trees.keys():
                idx = i
                prev = None
                while True:
                    parent = parents[idx-1]
                    tree = Tree()
                    if prev:
                        tree.add_child(prev)
                    trees[idx] = tree
                    tree.idx = idx
                    tree.gold_label = labels[idx-1]
                    tree.word = words[idx-1]
                    tree.relation = relations[idx-1]
                    if parent in trees.keys():
                        trees[parent].add_child(tree)
                        break
                    elif parent==0:
                        root = tree
                        break
                    else:
                        prev = tree
                        idx = parent
        # helper for visualization
        root._viz_all_children = trees
        root._viz_sentence = words
        root._viz_relations = relations
        root._viz_labels = labels
        return root