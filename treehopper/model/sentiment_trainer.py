from tqdm import tqdm
import torch
from torch.autograd import Variable as Var
import torch.nn.functional as F


class SentimentTrainer(object):
    """
    For Sentiment module
    """
    def __init__(self, args, model, criterion, optimizer, embedding_model = None):
        super(SentimentTrainer, self).__init__()
        self.args       = args
        self.model      = model
        if embedding_model:
            self.embedding_model = embedding_model
        else:
            self.embedding_model = self.model.embeddings
        self.criterion  = criterion
        self.optimizer  = optimizer
        self.epoch      = 0

    # helper function for training
    def train(self, dataset):
        self.model.train()
        self.embedding_model.train()
        self.embedding_model.zero_grad()
        self.optimizer.zero_grad()
        loss, k = 0.0, 0
        # torch.manual_seed(789)
        indices = torch.randperm(len(dataset))
        for idx in tqdm(range(len(dataset)),desc='Training epoch '+str(self.epoch+1)+''):
            tree, sent, label = dataset[indices[idx]]
            input = Var(sent)
            target = Var(torch.LongTensor([int(label)]))
            if self.args.cuda:
                input = input.cuda()
                target = target.cuda()
            emb = F.torch.unsqueeze(self.embedding_model(input), 1)
            output, err, _, _ = self.model.forward(tree, emb, training=True)
            #params = self.model.childsumtreelstm.getParameters()
            # params_norm = params.norm()
            err = err/self.args.batchsize # + 0.5*self.args.reg*params_norm*params_norm # custom bias
            loss += err.data[0] #
            err.backward()
            k += 1
            if k==self.args.batchsize:
                for f in self.embedding_model.parameters():
                    f.data.sub_(f.grad.data * self.args.emblr)
                self.optimizer.step()
                self.embedding_model.zero_grad()
                self.optimizer.zero_grad()
                k = 0
        self.epoch += 1
        return loss/len(dataset)

    # helper function for testing
    def test(self, dataset):
        self.model.eval()
        self.embedding_model.eval()
        loss = 0
        accuracies = torch.zeros(len(dataset))

        output_trees = []
        outputs = []
        for idx in tqdm(range(len(dataset)), desc='Testing epoch  '+str(self.epoch)+''):
            tree, sent, label = dataset[idx]
            input = Var(sent, volatile=True)
            target = Var(torch.LongTensor([int(label)]), volatile=True)
            if self.args.cuda:
                input = input.cuda()
                target = target.cuda()
            emb = F.torch.unsqueeze(self.embedding_model(input),1)
            output, _, acc, tree = self.model(tree, emb)
            err = self.criterion(output, target)
            loss += err.data[0]
            accuracies[idx] = acc
            output_trees.append(tree)
            outputs.append(tree.output_softmax.data.numpy())
            # predictions[idx] = torch.dot(indices,torch.exp(output.data.cpu()))
        return loss/len(dataset), accuracies, outputs, output_trees

    def predict(self, dataset):
        self.model.eval()
        self.embedding_model.eval()
        output_trees = []
        for idx in tqdm(range(len(dataset)), desc='Predcting results'):
            tree, sent, _ = dataset[idx]
            input = Var(sent, volatile=True)
            emb = F.torch.unsqueeze(self.embedding_model(input), 1)
            output, _, acc, tree = self.model(tree, emb)
            output_trees.append(tree)
        return output_trees

