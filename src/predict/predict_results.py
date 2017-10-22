from datetime import datetime

import torch
from src.datas.dataset import SSTDataset
from src.datas.vocab import build_vocab, Vocab
from src.evaluate import sentiment
from src.evaluate.ensemble import load_best_models

def predict(models_filename):
    args = sentiment.set_arguments({})
    vocab = sentiment.create_full_dataset(args)[0]
    test_dataset = SSTDataset('test', vocab, num_classes=3)
    trainer_instance = load_best_models([models_filename], args)[0]
    loss, accuracies, outputs, output_trees = trainer_instance.test(test_dataset)
    test_acc = torch.mean(accuracies)
    print("\n-------\n")
    print(loss, test_acc)
    print("\n-------\n")
    return test_acc
    # test_trees = trainer_instance.predict(test_dataset)
    # return test_trees

def save_submission(predictions):
    filename = 'finals/submission_fast_emblr.txt'.format(date=datetime.now())
    with open(filename, 'w') as submission_file:
        for sentence in predictions:
            submission_file.write(sentence.get_predicted_labels() + '\n')
    print('Good luck!')


acc =[]
for i in range(0,25):
    acc.append(predict('models/saved_model/models_20171022_2010/model_'+str(i)+'.pth'))
print(str(acc))
# save_submission(predictions)

#predictions = predict('finals/2saved_model22_model_20170804_2312.pth')
#save_submission(predictions)