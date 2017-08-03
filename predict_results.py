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
    model = load_best_models([models_filename], args)[0]
    test_trees = model.predict(test_dataset)
    test_trees[0].get_predicted_labels()
    return test_trees

predictions = predict("models/saved_model24_model_20170803_1608.pth")
print(predictions[0].get_predicted_labels())
