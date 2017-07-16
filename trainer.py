from __future__ import print_function

import os
import torch.optim as optim
import gc

from model import *
from metrics import Metrics
from utils import load_word_vectors
from sentiment_trainer import SentimentTrainer


def choose_optimizer(args, model):
    if args.optim =='adam':
        return optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd)
    elif args.optim=='adagrad':
        # optimizer   = optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd)
        return optim.Adagrad([
                {'params': model.parameters(), 'lr': args.lr}
            ], lr=args.lr, weight_decay=args.wd)


def load_embedding_model(args,vocab):
    embedding_model = nn.Embedding(vocab.size(), args.input_dim)

    if args.cuda:
        embedding_model = embedding_model.cuda()
    # for words common to dataset vocab and GLOVE, use GLOVE vectors
    # for other words in dataset vocab, use random normal vectors
    emb_file = os.path.join(args.data, 'sst_embed.pth')
    if os.path.isfile(emb_file):
        emb = torch.load(emb_file)
    else:
        # load glove embeddings and vocab
        glove_vocab, glove_emb = load_word_vectors(os.path.join(args.glove,'glove.twitter.27B.25d'))
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
    embedding_model.state_dict()['weight'].copy_(emb)
    return embedding_model


def train(train_dataset, dev_dataset,vocab,args):
    criterion = nn.NLLLoss()

    # initialize model, criterion/loss_function, optimizer

    model = TreeLSTMSentiment(
                args.cuda, vocab.size(),
                args.input_dim, args.mem_dim,
                args.num_classes, args.model_name, criterion
            )

    if args.cuda:
        model.cuda()
        criterion.cuda()

    optimizer = choose_optimizer(args,model)

    metrics = Metrics(args.num_classes)

    utils.count_param(model)

    embedding_model = load_embedding_model(args,vocab)


    # create trainer object for training and testing
    trainer     = SentimentTrainer(args, model, embedding_model ,criterion, optimizer)

    max_dev = 0
    max_dev_epoch = 0
    filename = args.name + '.pth'
    for epoch in range(args.epochs):
        train_loss = trainer.train(train_dataset)
        dev_loss, dev_pred, output_trees = trainer.test(dev_dataset)
        dev_acc = metrics.sentiment_accuracy_score(dev_pred, dev_dataset.labels)
        print('==> Train loss   : %f \t' % train_loss, end="")
        print('Epoch ', epoch, 'dev percentage ', dev_acc)
        torch.save(model, args.saved + str(epoch) + '_model_' + filename)
        torch.save(embedding_model, args.saved + str(epoch) + '_embedding_' + filename)
        if dev_acc > max_dev:
            max_dev = dev_acc
            max_dev_epoch = epoch
        gc.collect()
    print('epoch ' + str(max_dev_epoch) + ' dev score of ' + str(max_dev))
    print('eva on test set ')
    model = torch.load(args.saved + str(max_dev_epoch) + '_model_' + filename)
    embedding_model = torch.load(args.saved + str(max_dev_epoch) + '_embedding_' + filename)
    trainer = SentimentTrainer(args, model, embedding_model, criterion, optimizer)
    print('Epoch with max dev:' + str(max_dev_epoch) + ' |test percentage ' + str(max_dev))
    print('____________________' + str(args.name) + '___________________')
    # test_loss, test_pred = trainer.test(test_dataset)
    # test_acc = metrics.sentiment_accuracy_score(test_pred, test_dataset.labels)

    return max_dev_epoch, max_dev

