from __future__ import print_function

import os
import torch.optim as optim
import gc

from model import *
from vocab import Vocab
from dataset import SSTDataset
from metrics import Metrics
from utils import load_word_vectors, build_vocab
from config import parse_args
from trainer import SentimentTrainer


def main():
    global args
    args = parse_args(type=1)
    args.input_dim = 300
    args.mem_dim = 168
    args.num_classes = 3  # -1 0 1
    args.cuda = args.cuda and torch.cuda.is_available()
    print(args)

    train_dir = 'training-treebank'
    token_files = os.path.join(train_dir, 'sents.toks')
    vocab_file = 'vocab.txt'
    build_vocab([
        'training-treebank/rev_sentence.txt',
        'training-treebank/sklad_sentence.txt'
    ], 'vocab.txt')
    vocab = Vocab(filename=vocab_file)
    dataset = SSTDataset(train_dir, vocab, args.num_classes,
                               args.fine_grain, args.model_name)

    # train/val split
    # TODO: refactor
    splitting_point = int(len(dataset)*0.9)
    train_dataset, dev_dataset = SSTDataset(num_classes=args.num_classes), SSTDataset(num_classes=args.num_classes)
    train_dataset.trees = dataset.trees[:splitting_point]
    train_dataset.sentences = dataset.sentences[:splitting_point]
    train_dataset.labels = dataset.labels[:splitting_point]
    dev_dataset.trees = dataset.trees[splitting_point:]
    dev_dataset.sentences = dataset.sentences[splitting_point:]
    dev_dataset.labels = dataset.labels[splitting_point:]



    # train_dir = os.path.join(args.data,'train/')
    # dev_dir = os.path.join(args.data,'dev/')
    # test_dir = os.path.join(args.data,'test/')
    #
    # # write unique words from all token files
    # token_files = [os.path.join(split, 'sents.toks') for split in [train_dir, dev_dir, test_dir]]
    # vocab_file = os.path.join(args.data,'vocab-cased.txt') # use vocab-cased
    #
    # # get vocab object from vocab file previously written
    # vocab = Vocab(filename=vocab_file)
    # print('==> SST vocabulary size : %d ' % vocab.size())
    #
    # # Load SST dataset splits
    #
    # # train
    # train_file = os.path.join(args.data,'sst_train.pth')
    # if os.path.isfile(train_file):
    #     train_dataset = torch.load(train_file)
    # else:
    #     train_dataset = SSTDataset(train_dir, vocab, args.num_classes, args.fine_grain, args.model_name)
    #     torch.save(train_dataset, train_file)
    #
    # # dev
    # dev_file = os.path.join(args.data,'sst_dev.pth')
    # if os.path.isfile(dev_file):
    #     dev_dataset = torch.load(dev_file)
    # else:
    #     dev_dataset = SSTDataset(dev_dir, vocab, args.num_classes, args.fine_grain, args.model_name)
    #     torch.save(dev_dataset, dev_file)
    #
    # # test
    # test_file = os.path.join(args.data,'sst_test.pth')
    # if os.path.isfile(test_file):
    #     test_dataset = torch.load(test_file)
    # else:
    #     test_dataset = SSTDataset(test_dir, vocab, args.num_classes, args.fine_grain, args.model_name)
    #     torch.save(test_dataset, test_file)

    criterion = nn.NLLLoss()
    # initialize model, criterion/loss_function, optimizer
    model = TreeLSTMSentiment(
                args.cuda, vocab.size(),
                args.input_dim, args.mem_dim,
                args.num_classes, args.model_name, criterion
            )

    embedding_model = nn.Embedding(vocab.size(), args.input_dim)

    if args.cuda:
        embedding_model = embedding_model.cuda()

    if args.cuda:
        model.cuda(), criterion.cuda()
    if args.optim=='adam':
        optimizer   = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd)
    elif args.optim=='adagrad':
        # optimizer   = optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd)
        optimizer = optim.Adagrad([
                {'params': model.parameters(), 'lr': args.lr}
            ], lr=args.lr, weight_decay=args.wd)
    metrics = Metrics(args.num_classes)

    utils.count_param(model)

    # for words common to dataset vocab and GLOVE, use GLOVE vectors
    # for other words in dataset vocab, use random normal vectors
    emb_file = os.path.join(args.data, 'sst_embed.pth')
    if os.path.isfile(emb_file):
        emb = torch.load(emb_file)
    else:
        # load glove embeddings and vocab
        glove_vocab, glove_emb = load_word_vectors(os.path.join(args.glove,'glove.twitter.27B.200d'))
        print('==> GLOVE vocabulary size: %d ' % glove_vocab.size())

        emb = torch.zeros(vocab.size(),glove_emb.size(1))

        for word in vocab.labelToIdx.keys():
            if glove_vocab.getIndex(word):
                emb[vocab.getIndex(word)] = glove_emb[glove_vocab.getIndex(word)]
            else:
                emb[vocab.getIndex(word)] = torch.Tensor(emb[vocab.getIndex(word)].size()).normal_(-0.05,0.05)
        torch.save(emb, emb_file)

    # plug these into embedding matrix inside model
    if args.cuda:
        emb = emb.cuda()

    # model.childsumtreelstm.emb.state_dict()['weight'].copy_(emb)
    # embedding_model.state_dict()['weight'].copy_(emb)

    # create trainer object for training and testing
    trainer     = SentimentTrainer(args, model, embedding_model ,criterion, optimizer)

    max_dev = 0
    max_dev_epoch = 0
    filename = args.name + '.pth'
    for epoch in range(args.epochs):
        train_loss = trainer.train(train_dataset)
        dev_loss, dev_acc, _ = trainer.test(dev_dataset)
        dev_acc = torch.mean(dev_acc)
        print('==> Train loss   : %f \t' % train_loss, end="")
        print('Epoch ', epoch, 'dev percentage ', dev_acc)
        torch.save(model, args.saved + str(epoch) + '_model_' + filename)
        torch.save(embedding_model, args.saved + str(epoch) + '_embedding_' + filename)
        # if dev_acc > max_dev:
        #     max_dev = dev_acc
        #     max_dev_epoch = epoch
        gc.collect()
    print('epoch ' + str(max_dev_epoch) + ' dev score of ' + str(max_dev))
    print('eva on test set ')
    model = torch.load(args.saved + str(max_dev_epoch) + '_model_' + filename)
    embedding_model = torch.load(args.saved + str(max_dev_epoch) + '_embedding_' + filename)
    trainer = SentimentTrainer(args, model, embedding_model, criterion, optimizer)
    # test_loss, test_pred = trainer.test(test_dataset)
    # test_acc = metrics.sentiment_accuracy_score(test_pred, test_dataset.labels)

    # print('Epoch with max dev:' + str(max_dev_epoch) + ' |test percentage ' + str(test_acc))
    # print('____________________' + str(args.name) + '___________________')

if __name__ == "__main__":
    main()