import argparse

import torch
from torch import nn
from model.sentiment_trainer import SentimentTrainer

from data.dataset import SSTDataset
import train


def predict(args):
    trainer_instance = load_best_model(args.model_path, args)
    test_dataset = SSTDataset.create_dataset_from_user_input(
        args.input_sentences, args.input_parents,
        trainer_instance.model.vocab,
        num_classes=3
    )
    test_trees = trainer_instance.predict(test_dataset)
    return test_trees


def save_predictions(predictions, filename):
    with open(filename, 'w') as submission_file:
        for sentence in predictions:
            submission_file.write(sentence.get_predicted_labels() + '\n')
    print('Predictions saved in {}'.format(filename))


def load_best_model(model_filename, args):
    model = torch.load(model_filename)
    trainer = SentimentTrainer(args, model,criterion=nn.NLLLoss(), optimizer=None)
    return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch TreeLSTM for Sentiment Analysis Trees - Predictions')
    parser.add_argument('--model_path', help='Path to saved model', required=True)
    parser.add_argument('--input_parents', help='Path to parent input directory', default="test/polevaltest_parents.txt")
    parser.add_argument('--input_sentences', help='Path to sentence input directory',
                        default="test/polevaltest_sentence.txt")
    parser.add_argument('--output', help='Path to file with predictions', default="predictions.txt")
    args = train.set_arguments({}, parser)
    predictions = predict(args)
    save_predictions(predictions, args.output)



