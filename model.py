import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var
import utils


class ChildSumTreeLSTM(nn.Module):
    def __init__(self, cuda, in_dim, mem_dim, criterion):
        super(ChildSumTreeLSTM, self).__init__()
        self.cudaFlag = cuda
        self.in_dim = in_dim
        self.mem_dim = mem_dim

        # self.emb = nn.Embedding(vocab_size,in_dim,
        #                         padding_idx=Constants.PAD)
        # torch.manual_seed(123)

        self.ix = nn.Linear(self.in_dim,self.mem_dim)
        self.ih = nn.Linear(self.mem_dim,self.mem_dim)

        self.fh = nn.Linear(self.mem_dim, self.mem_dim)
        self.fx = nn.Linear(self.in_dim,self.mem_dim)

        self.ux = nn.Linear(self.in_dim,self.mem_dim)
        self.uh = nn.Linear(self.mem_dim,self.mem_dim)

        self.ox = nn.Linear(self.in_dim,self.mem_dim)
        self.oh = nn.Linear(self.mem_dim,self.mem_dim)

        self.criterion = criterion
        self.output_module = None

    def set_output_module(self, output_module):
        self.output_module = output_module

    # def getParameters(self):
    #     """
    #     Get flatParameters
    #     note that getParameters and parameters is not equal in this case
    #     getParameters do not get parameters of output module
    #     :return: 1d tensor
    #     """
    #     params = []
    #     for m in [self.ix, self.ih, self.fx, self.fh, self.ox, self.oh, self.ux, self.uh]:
    #         # we do not get param of output module
    #         l = list(m.parameters())
    #         params.extend(l)
    #
    #     one_dim = [p.view(p.numel()) for p in params]
    #     params = F.torch.cat(one_dim)
    #     return params


    def node_forward(self, inputs, child_c, child_h):
        child_h_sum = F.torch.sum(torch.squeeze(child_h,1),0)

        i = F.sigmoid(self.ix(inputs)+self.ih(child_h_sum))
        o = F.sigmoid(self.ox(inputs)+self.oh(child_h_sum))
        u = F.tanh(self.ux(inputs)+self.uh(child_h_sum))

        # add extra singleton dimension
        fx = F.torch.unsqueeze(self.fx(inputs),1)
        f = F.torch.cat([self.fh(child_hi)+fx for child_hi in child_h], 0)
        f = F.sigmoid(f)
        # removing extra singleton dimension
        f = F.torch.unsqueeze(f,1)
        fc = F.torch.squeeze(F.torch.mul(f,child_c),1)

        c = F.torch.mul(i,u) + F.torch.sum(fc,0)
        h = F.torch.mul(o, F.tanh(c))

        return c,h

    def forward(self, tree, embs, training = False):
        # add singleton dimension for future call to node_forward
        # embs = F.torch.unsqueeze(self.emb(inputs),1)

        loss = Var(torch.zeros(1)) # init zero loss
        if self.cudaFlag:
            loss = loss.cuda()

        for idx in range(tree.num_children):
            _, child_loss = self.forward(tree.children[idx], embs, training)
            loss = loss + child_loss
        child_c, child_h = self.get_child_states(tree)
        tree.state = self.node_forward(embs[tree.idx-1], child_c, child_h)


        output = self.output_module.forward(tree.state[1], training)
        tree.output = output
        if training and tree.gold_label != None:
            target = Var(utils.map_label_to_target_sentiment(tree.gold_label))
            if self.cudaFlag:
                target = target.cuda()
            loss = loss + self.criterion(output, target)
        return tree.state, loss

    def get_child_states(self, tree):
        # add extra singleton dimension in middle...
        # because pytorch needs mini batches... :sad:
        if tree.num_children==0:
            child_c = Var(torch.zeros(1,1,self.mem_dim))
            child_h = Var(torch.zeros(1,1,self.mem_dim))
            if self.cudaFlag:
                child_c, child_h = child_c.cuda(), child_h.cuda()
        else:
            child_c = Var(torch.Tensor(tree.num_children,1,self.mem_dim))
            child_h = Var(torch.Tensor(tree.num_children,1,self.mem_dim))
            if self.cudaFlag:
                child_c, child_h = child_c.cuda(), child_h.cuda()
            for idx in range(tree.num_children):
                child_c[idx], child_h[idx] = tree.children[idx].state
        return child_c, child_h

# output module
class SentimentModule(nn.Module):
    def __init__(self, cuda, mem_dim, num_classes, dropout = False):
        super(SentimentModule, self).__init__()
        self.cudaFlag = cuda
        self.mem_dim = mem_dim
        self.num_classes = num_classes
        self.dropout = dropout
        # torch.manual_seed(456)
        self.l1 = nn.Linear(self.mem_dim, self.num_classes)
        self.logsoftmax = nn.LogSoftmax()
        if self.cudaFlag:
            self.l1 = self.l1.cuda()

    def forward(self, vec, training = False):
        if self.dropout:
            out = self.logsoftmax(self.l1(F.dropout(vec, training = training)))
        else:
            out = self.logsoftmax(self.l1(vec))
        return out

class TreeLSTMSentiment(nn.Module):
    def __init__(self, cuda, vocab_size, in_dim, mem_dim, num_classes, model_name, criterion):
        super(TreeLSTMSentiment, self).__init__()
        self.cudaFlag = cuda
        self.model_name = model_name
        self.tree_module = ChildSumTreeLSTM(cuda, in_dim, mem_dim, criterion)
        self.output_module = SentimentModule(cuda, mem_dim, num_classes, dropout=True)
        self.tree_module.set_output_module(self.output_module)

    def forward(self, tree, inputs, training = False):
        tree_state, loss = self.tree_module(tree, inputs, training)
        output = tree.output
        return output, loss