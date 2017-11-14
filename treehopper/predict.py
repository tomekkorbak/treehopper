from datetime import datetime

import torch
from torch import nn
from treehopper.data.vocab import Vocab
from treehopper.model.sentiment_trainer import SentimentTrainer

from treehopper.data.dataset import SSTDataset
from treehopper import train

SAMPLE_MODEL = 'models/sample_model/model_0.pth'

def predict(models_filename):
    args = train.set_arguments({})
    vocab = train.create_full_dataset(args)[0]
    test_dataset = SSTDataset('test', Vocab(filename='tmp/vocab.txt'), num_classes=3)
    trainer_instance = load_best_model(models_filename, args)
    test_trees = trainer_instance.predict(test_dataset)
    return test_trees

def save_submission(predictions, filename):
    with open(filename, 'w') as submission_file:
        for sentence in predictions:
            submission_file.write(sentence.get_predicted_labels() + '\n')
    print('Predictions saved in {}'.format(filename))



def load_best_model(model_filename, args):
    model = torch.load(model_filename)
    emb = torch.load(model_filename.replace("model_", "embedding_"))
    trainer = SentimentTrainer(args, model, emb,
                               criterion=nn.NLLLoss(), optimizer=None)
    return trainer
if __name__ == "__main__":

    predictions = predict(SAMPLE_MODEL)
    save_submission(predictions,  'submission.txt'.format(date=datetime.now()))