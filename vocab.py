import os


def build_vocab(filenames, vocabfile):
    """Write unique words from a set of files to a new file"""
    if os.path.isfile(vocabfile):
        print('Loading existing vocabulary from', vocabfile)
        return
    vocab = set()
    for filename in filenames:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = line.rstrip('\n').lower().split()
                vocab |= set(tokens)
    with open(vocabfile, 'w', encoding='utf-8') as f:
        for token in vocab:
            f.write(token + '\n')


class Vocab(object):
    def __init__(self, filename=None):
        self.idx_to_token = {}
        self.token_to_idx = {}

        if filename:
            self.load_file(filename)

    def size(self):
        return len(self.idx_to_token)

    def load_file(self, filename):
        """Load entries from a file."""
        for line in open(filename, encoding='utf-8'):
            self.add_token(line.rstrip('\n'))

    def get_index(self, key, default=None):
        key = key.lower()
        try:
            return self.token_to_idx[key]
        except KeyError:
            return default

    def get_token(self, idx, default=None):
        try:
            return self.idx_to_token[idx]
        except KeyError:
            return default

    def add_token(self, label):
        label = label.lower()

        if label in self.token_to_idx:
            idx = self.token_to_idx[label]
        else:
            idx = len(self.idx_to_token)
            self.idx_to_token[idx] = label
            self.token_to_idx[label] = idx
        return idx

    def convert_to_idx(self, tokens, unk_word, bos_word=None, eos_word=None):
        """Convert tokens to indices. Use `unk_word` if token is not found. 
        Optionally insert `bos_word` and `eos_word` at the begging and at the 
        end of the list of indices.
        """
        vec = []

        if bos_word is not None:
            vec += [self.get_index(bos_word)]

        unk = self.get_index(unk_word)
        vec += [self.get_index(token, default=unk) for token in tokens]

        if eos_word is not None:
            vec += [self.get_index(eos_word)]

        return vec

    def convert_to_tokens(self, idx, stop):
        """Convert indices to tokens. If index `stop` is reached, convert it 
        and return.
        """
        tokens = []
        for i in idx:
            tokens += [self.get_token(i)]
            if i == stop:
                break
        return tokens
