import torch
from treehopper.data.vocab import Vocab
from treehopper.predict import load_best_model

from treehopper.data.dataset import SSTDataset
from treehopper import train

SAMPLE_MODEL = 'models/sample_model/model_0.pth'

def eval(model_filename):
    args = train.set_arguments({})
    vocab = train.create_full_dataset(args)[0]
    test_dataset = SSTDataset('test', Vocab(filename='tmp/vocab.txt'), num_classes=3)
    trainer_instance = load_best_model(model_filename, args)
    loss, accuracies, outputs, output_trees = trainer_instance.test(test_dataset)
    test_acc = torch.mean(accuracies)
    return test_acc

if __name__ == "__main__":
    print("Accuracy {}".format(eval(SAMPLE_MODEL)))