from __future__ import print_function

import gc
import os

import torch
import torch.optim as optim
from src.datas.embeddings import load_embedding_model
from src.model.model import TreeLSTMSentiment
from src.model.sentiment_trainer import SentimentTrainer
from torch import nn


def choose_optimizer(args, model):

    if args.optim =='adam':
        return optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd)
    elif args.optim=='adagrad':
        # optimizer   = optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd)
        return optim.Adagrad([
                {'params': model.parameters(), 'lr': args.lr}
            ], lr=args.lr, weight_decay=args.wd)


def train(train_dataset, dev_dataset, vocab, args):
    # Optionally reweight loss per class to the distribution of classes in
    # the public dataset
    weight = torch.Tensor([1/0.024, 1/0.820, 1/0.156]) if args.reweight else None
    criterion = nn.NLLLoss(weight=weight)

    # initialize model, criterion/loss_function, optimizer

    model = TreeLSTMSentiment(args=args, criterion=criterion)

    if args.cuda:
        model.cuda()
        criterion.cuda()

    optimizer = choose_optimizer(args,model)

    embedding_model = load_embedding_model(args,vocab)

    # create trainer object for training and testing
    trainer = SentimentTrainer(args, model, embedding_model ,criterion, optimizer)
    experiment_dir = os.path.join(os.getcwd(), args.saved, "models_" + args.name)
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
        open(experiment_dir+"/"+"config.txt", "w+").write(str(args))
    max_dev = 0
    max_dev_epoch = 0
    for epoch in range(args.epochs):
        train_loss = trainer.train(train_dataset)
        dev_loss, dev_acc, _, _ = trainer.test(dev_dataset)
        dev_acc = torch.mean(dev_acc)
        print('==> Train loss   : %f \t' % train_loss, end="")
        print('Epoch ', epoch, 'dev percentage ', dev_acc)
        model_filename = experiment_dir + '/' +'model_' + str(epoch) + '.pth'
        emb_filename = experiment_dir + '/' + 'embedding_' + str(epoch) +  '.pth'
        torch.save(model, model_filename)
        torch.save(embedding_model, emb_filename)
        if dev_acc > max_dev:
            max_dev = dev_acc
            max_dev_epoch = epoch
            max_model_filename = model_filename
        gc.collect()
    print('epoch ' + str(max_dev_epoch) + ' dev score of ' + str(max_dev))

    return max_dev_epoch, max_dev, max_model_filename


