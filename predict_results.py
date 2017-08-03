from datetime import datetime

import sentiment
from dataset import SSTDataset
from ensemble import load_best_models
from vocab import build_vocab, Vocab


def predict(models_filename):
    train_dir = 'test'
    vocab_file = 'tmp/vocab_test.txt'
    build_vocab([
        'test/polevaltest_sentence.txt',
    ], 'tmp/vocab_test.txt')
    vocab = Vocab(filename=vocab_file)
    test_dataset = SSTDataset(train_dir, vocab, num_classes=3)
    args = sentiment.set_arguments({})
    trainer_instance = load_best_models([models_filename], args)[0]
    test_trees = trainer_instance.predict(test_dataset)
    return test_trees


def save_submission(predictions):
    filename = 'submission_{date:%Y%m%d_%H%M}.txt'.format(date=datetime.now())
    with open(filename, 'w') as submission_file:
        for sentence in predictions:
            submission_file.write(sentence.get_predicted_labels() + '\n')
    print('Good luck!')


predictions = predict("models/saved_model24_model_20170803_1608.pth")
save_submission(predictions)
