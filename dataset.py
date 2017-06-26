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

        self.sentences = self.read_sentences(os.path.join(path,'rev_sentence.txt')) + self.read_sentences(os.path.join(path,'sklad_sentence.txt'))
        self.trees = self.read_trees(os.path.join(path,'rev_parents.txt'), os.path.join(path,'rev_labels.txt')) + self.read_trees(os.path.join(path,'sklad_parents.txt'), os.path.join(path,'sklad_labels.txt'))

        # self.labels = self.read_labels(os.path.join(path,'dlabels.txt'))
        self.labels = []

        # only get pos or neg
        # new_trees = []
        # new_sentences = []
        # for i in range(len(temp_trees)):
        #     if temp_trees[i].gold_label != 1: # 0 neg, 1 neutral, 2 pos
        #         new_trees.append(temp_trees[i])
        #         new_sentences.append(temp_sentences[i])
        # self.trees = new_trees
        # self.sentences = new_sentences

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

    def read_trees(self, filename_parents, filename_labels):
        pfile = open(filename_parents, 'r', encoding='utf-8') # parent node
        lfile = open(filename_labels, 'r', encoding='utf-8') # label node
        p = pfile.readlines()
        l = lfile.readlines()
        pl = zip(p, l) # (parent, label) tuple
        trees = [self.read_tree(p_line, l_line) for p_line, l_line in tqdm(pl)]

        return trees

    def parse_dlabel_token(self, x):
        if x == '#':
            return None
        else:
            if self.fine_grain: # -2 -1 0 1 2 => 0 1 2 3 4
                return int(x)+2
            else: # # -2 -1 0 1 2 => 0 1 2
                tmp = int(x)
                if tmp < 0:
                    return 0
                elif tmp == 0:
                    return 1
                elif tmp >0 :
                    return 2

    def read_tree(self, line, label_line):
        # FIXED: tree.idx, also tree dict() use base 1 as it was in dataset
        # parents is list base 0, keep idx-1
        # labels is list base 0, keep idx-1
        parents = list(map(int,line.split())) # split each number and turn to int
        trees = dict() # this is dict
        root = None
        labels = list(map(self.parse_dlabel_token, label_line.split()))
        for i in range(1,len(parents)+1):
            #if not trees[i-1] and parents[i-1]!=-1:
            if i not in trees.keys() and parents[i-1]!=-1:
                idx = i
                prev = None
                while True:
                    parent = parents[idx-1]
                    if parent == -1:
                        break
                    tree = Tree()
                    if prev is not None:
                        tree.add_child(prev)
                    trees[idx] = tree
                    tree.idx = idx # -1 remove -1 here to prevent embs[tree.idx -1] = -1 while tree.idx = 0
                    tree.gold_label = labels[idx-1] # add node label
                    #if trees[parent-1] is not None:
                    if parent in trees.keys():
                        trees[parent].add_child(tree)
                        break
                    elif parent==0:
                        root = tree
                        break
                    else:
                        prev = tree
                        idx = parent
        return root

    def read_labels(self, filename):
        # Not in used
        with open(filename,'r') as f:
            labels = list(map(lambda x: float(x), f.readlines()))
            labels = torch.Tensor(labels)
        return labels