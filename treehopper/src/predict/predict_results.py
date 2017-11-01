from datetime import datetime

import torch
from torch import nn
from treehopper.src.data.vocab import Vocab
from treehopper.src.evaluate import sentiment
from treehopper.src.data.dataset import SSTDataset
from treehopper.src.model.sentiment_trainer import SentimentTrainer


def predict(models_filename):
    args = sentiment.set_arguments({})
    vocab = sentiment.create_full_dataset(args)[0]
    test_dataset = SSTDataset('test', Vocab(filename='tmp/vocab.txt'), num_classes=3)
    trainer_instance = load_best_models([models_filename], args)[0]
    test_trees = trainer_instance.predict(test_dataset)
    return test_trees

def save_submission(predictions, filename):
    with open(filename, 'w') as submission_file:
        for sentence in predictions:
            submission_file.write(sentence.get_predicted_labels() + '\n')
    print('Predictions saved in {}'.format(filename))



def load_best_models(models_filenames, args):
    models = []
    for model_filename in models_filenames:
        model = torch.load(model_filename)
        emb = torch.load(model_filename.replace("model_", "embedding_"))
        trainer = SentimentTrainer(args, model, emb,
                                   criterion=nn.NLLLoss(), optimizer=None)
        models.append(trainer)
    return models

if __name__ == "__main__":
    predictions = predict('models/sample_model/model_0.pth')
    save_submission(predictions,  'finals/submission.txt'.format(date=datetime.now()))