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
    test_dataset = SSTDataset(train_dir, vocab,num_classes=3)
    model = load_best_models([models_filename])[0]
    test_trees = model.predict(test_dataset)

predict("models/saved_model14_model_20170731_2209.pth")