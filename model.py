import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var

from zoneout import zoneout


class ChildSumTreeLSTM(nn.Module):
    def __init__(self, args, criterion, output_module):
        super(ChildSumTreeLSTM, self).__init__()
        self.cuda_flag = args.cuda
        self.in_dim = args.input_dim
        self.mem_dim = args.mem_dim
        self.recurrent_dropout_p = args.recurrent_dropout

        self.ix = nn.Linear(self.in_dim, self.mem_dim)
        self.ih = nn.Linear(self.mem_dim, self.mem_dim)

        self.fh = nn.Linear(self.mem_dim, self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)

        self.ux = nn.Linear(self.in_dim, self.mem_dim)
        self.uh = nn.Linear(self.mem_dim, self.mem_dim)

        self.ox = nn.Linear(self.in_dim, self.mem_dim)
        self.oh = nn.Linear(self.mem_dim, self.mem_dim)

        self.criterion = criterion
        self.output_module = output_module

    def node_forward(self, inputs, child_c, child_h, training):
        child_h_sum = F.torch.sum(torch.squeeze(child_h, 1), 0)

        i = F.sigmoid(self.ix(inputs)+self.ih(child_h_sum))
        o = F.sigmoid(self.ox(inputs)+self.oh(child_h_sum))
        u = F.tanh(self.ux(inputs)+self.uh(child_h_sum))

        # add extra singleton dimension
        fx = F.torch.unsqueeze(self.fx(inputs), 1)
        f = F.torch.cat([self.fh(child_hi) + fx for child_hi in child_h], 0)
        f = F.sigmoid(f)
        # removing extra singleton dimension
        f = F.torch.unsqueeze(f, 1)
        fc = F.torch.squeeze(F.torch.mul(f, child_c), 1)

        c = zoneout(
            current_input=F.torch.mul(i, u) + F.torch.sum(fc, 0),
            previous_input=F.torch.sum(fc, 0),
            p=self.recurrent_dropout_p,
            training=training
        )
        h = zoneout(
            current_input=F.torch.mul(o, F.tanh(c)),
            previous_input=child_h_sum,
            p=self.recurrent_dropout_p,
            training=training
        )

        return c, h

    def forward(self, tree, embs, training=False):
        loss = Var(torch.zeros(1))  # initialize loss with zero
        if self.cuda_flag:
            loss = loss.cuda()

        for idx in range(tree.num_children):
            _, child_loss = self.forward(tree.children[idx], embs, training)
            loss += child_loss
        child_c, child_h = self.get_children_states(tree)
        tree.state = self.node_forward(embs[tree.idx-1], child_c, child_h, training)

        output, output_softmax = self.output_module.forward(tree.state[1], training)
        tree.output_softmax = output_softmax
        tree.output = output
        if training and tree.gold_label is not None:
            target = Var(torch.LongTensor([tree.gold_label]))
            if self.cuda_flag:
                target = target.cuda()
            loss = loss + self.criterion(output, target)
        return tree.state, loss

    def get_children_states(self, tree):
        # add extra singleton dimension in middle
        if tree.num_children == 0:
            child_c = Var(torch.zeros(1, 1, self.mem_dim))
            child_h = Var(torch.zeros(1, 1, self.mem_dim))
            if self.cuda_flag:
                child_c, child_h = child_c.cuda(), child_h.cuda()
        else:
            child_c = Var(torch.Tensor(tree.num_children, 1, self.mem_dim))
            child_h = Var(torch.Tensor(tree.num_children, 1, self.mem_dim))
            if self.cuda_flag:
                child_c, child_h = child_c.cuda(), child_h.cuda()
            for idx in range(tree.num_children):
                child_c[idx], child_h[idx] = tree.children[idx].state
        return child_c, child_h


class SentimentModule(nn.Module):
    def __init__(self, args, dropout=0.5):
        super(SentimentModule, self).__init__()
        self.cuda_flag = args.cuda
        self.mem_dim = args.mem_dim
        self.num_classes = args.num_classes

        self.dropout = dropout
        self.linear_layer = nn.Linear(self.mem_dim, self.num_classes)
        self.logsoftmax = nn.LogSoftmax()
        self.softmax = nn.Softmax()
        if self.cuda_flag:
            self.linear_layer = self.linear_layer.cuda()

    def forward(self, vec, training=False):

        return self.logsoftmax(self.linear_layer(F.dropout(vec,
                                                           p=self.dropout,
                                                           training=training))),\
               self.softmax(self.linear_layer(F.dropout(vec,
                                                p=self.dropout,
                                                training=training)))


class TreeLSTMSentiment(nn.Module):
    def __init__(self, args, criterion):
        super(TreeLSTMSentiment, self).__init__()
        self.output_module = SentimentModule(args, dropout=0.5)
        self.tree_module = ChildSumTreeLSTM(args, criterion,
                                            output_module=self.output_module)

    def forward(self, tree, inputs, training=False):
        _, loss = self.tree_module(tree, inputs, training)
        return tree.output, loss, tree.compute_accuracy(), tree.output_softmax
