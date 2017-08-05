import os
import subprocess
import numpy as np

import torch
from torch.nn import Embedding
from gensim.models import KeyedVectors
from gensim.models.wrappers import FastText

from vocab import Vocab


def load_word_vectors(embeddings_path):
    if os.path.isfile(embeddings_path + '.pth') and \
            os.path.isfile(embeddings_path + '.vocab'):
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


def apply_not_known_words(emb,args, not_known,vocab):
    new_words = 'tmp/new_words.txt'
    f = open(new_words, 'w', encoding='utf-8')
    for item in not_known:
        f.write("%s\n" % item)
    cmd = " ".join(["./fastText/fasttext", "print-word-vectors",
                    args.emb_dir + "/" + args.emb_file + ".bin", "<", new_words])
    print(cmd)
    ps = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = ps.communicate()[0]
    new_words_embeddings = [x.split(" ")[:-1] for x in output.decode("utf-8").split("\n")]
    for word in new_words_embeddings:
        if args.input_dim == len(word[1:]):
            emb[vocab.get_index(word[0])] = torch.from_numpy(np.asarray(list(map(float, word[1:]))))
        else:
            print('Word embedding from subproccess has different length than expected')
    # os.remove(new_words)
    return emb


def load_embedding_model(args, vocab):
    embedding_model = Embedding(vocab.size(), args.input_dim)

    if args.cuda:
        embedding_model = embedding_model.cuda()
    emb_file = os.path.join(args.data, args.emb_dir.split("/")[-1]+"_"+args.emb_file + '_emb.pth')
    if os.path.isfile(emb_file) and torch.load(emb_file).size()[1] == args.input_dim:
        emb = torch.load(emb_file)
    else:
        glove_vocab, glove_emb = load_word_vectors(os.path.join(args.emb_dir,args.emb_file))
        print('==> GLOVE vocabulary size: %d ' % glove_vocab.size())

        emb = torch.zeros(vocab.size(), glove_emb.size(1))
        not_known = []
        for word in vocab.token_to_idx.keys():
            if glove_vocab.get_index(word):
                emb[vocab.get_index(word)] = glove_emb[glove_vocab.get_index(word)]
            else:
                not_known.append(word)
                emb[vocab.get_index(word)] = torch.Tensor(emb[vocab.get_index(word)].size()).normal_(-0.05, 0.05)
        # if args.calculate_new_words:
        #     emb = apply_not_known_words(emb, args, not_known, vocab)

        torch.save(emb, emb_file)

    if args.cuda:
        emb = emb.cuda()
    # plug these into embedding matrix inside model
    embedding_model.state_dict()['weight'].copy_(emb)
    return embedding_model
