import os
import torch
from gensim.models import KeyedVectors
from gensim.models.wrappers import FastText

from vocab import Vocab


def load_word_vectors(embeddings_path):
    if os.path.isfile(embeddings_path+ '.pth') and os.path.isfile(embeddings_path+ '.vocab'):
        print('==> File found, loading to memory')
        vectors = torch.load(embeddings_path + '.pth')
        vocab = Vocab(filename=embeddings_path + '.vocab')
        return vocab, vectors
    if os.path.isfile(embeddings_path + '.model'):
        model = KeyedVectors.load(embeddings_path + ".model")
    if os.path.isfile(embeddings_path + '.vec'):
        model = FastText.load_word2vec_format(embeddings_path + '.vec')
    list_of_tokens = model.vocab.keys()
    vectors = torch.zeros(len(list_of_tokens), model.vector_size)
    with open(embeddings_path + '.vocab', 'w', encoding='utf-8') as f:
        for token in list_of_tokens:
            f.write(token+'\n')
    vocab = Vocab(filename=embeddings_path + '.vocab')
    for index, word in enumerate(list_of_tokens):
        vectors[index, :] = torch.from_numpy(model[word])
    return vocab, vectors



# write unique words from a set of files to a new file
def build_vocab(filenames, vocabfile):
    vocab = set()
    for filename in filenames:
        with open(filename,'r', encoding='utf-8') as f:
            for line in f:
                tokens = line.rstrip('\n').split(' ')
                vocab |= set(tokens)
    with open(vocabfile,'w', encoding='utf-8') as f:
        for token in vocab:
            f.write(token+'\n')


def map_label_to_target_sentiment(label, num_classes = 3):
    # num_classes not use yet
    target = torch.LongTensor(1)
    target[0] = int(label) # nothing to do here as we preprocess data
    return target
