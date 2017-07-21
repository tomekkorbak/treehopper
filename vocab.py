# vocab object from harvardnlp/opennmt-py
class Vocab(object):
    def __init__(self, filename=None, data=None, lower=False):
        self.idx_to_label = {}
        self.label_to_idx = {}
        self.lower = lower

        # Special entries will not be pruned.
        self.special = []

        if data: self.add_specials(data)
        if filename: self.load_file(filename)

    def size(self):
        return len(self.idx_to_label)

    # Load entries from a file.
    def load_file(self, filename):
        for line in open(filename, encoding='utf-8'):
            self.add_label(line.rstrip('\n'))

    def get_index(self, key, default=None):
        if self.lower:
            key = key.lower()
        try:
            return self.label_to_idx[key]
        except KeyError:
            return default

    def get_label(self, idx, default=None):
        try:
            return self.idx_to_label[idx]
        except KeyError:
            return default

    # Mark this `label` and `idx` as special
    def ads_special(self, label, idx=None):
        idx = self.add_label(label)
        self.special += [idx]

    # Mark all labels in `labels` as specials
    def add_specials(self, labels):
        for label in labels:
            self.ads_special(label)

    # Add `label` in the dictionary. Use `idx` as its index if given.
    def add_label(self, label):
        if self.lower:
            label = label.lower()

        if label in self.label_to_idx:
            idx = self.label_to_idx[label]
        else:
            idx = len(self.idx_to_label)
            self.idx_to_label[idx] = label
            self.label_to_idx[label] = idx
        return idx

    # Convert `labels` to indices. Use `unkWord` if not found.
    # Optionally insert `bosWord` at the beginning and `eosWord` at the .
    def convert_to_idx(self, labels, unk_word, bos_word=None, eos_word=None):
        vec = []

        if bos_word is not None:
            vec += [self.get_index(bos_word)]

        unk = self.get_index(unk_word)
        vec += [self.get_index(label, default=unk) for label in labels]

        if eos_word is not None:
            vec += [self.get_index(eos_word)]

        return vec

    # Convert `idx` to labels. If index `stop` is reached, convert it and return.
    def convert_to_labels(self, idx, stop):
        labels = []
        for i in idx:
            labels += [self.get_label(i)]
            if i == stop:
                break
        return labels
