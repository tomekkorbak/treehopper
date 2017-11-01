from datetime import datetime

from treehopper.src.data.vocab import Vocab
from treehopper.src.evaluate import sentiment
from treehopper.src.evaluate.ensemble import load_best_models

from treehopper.src.data.dataset import SSTDataset


def predict(models_filename):
    args = sentiment.set_arguments({})
    vocab = sentiment.create_full_dataset(args)[0]
    test_dataset = SSTDataset('test', Vocab(filename='tmp/vocab.txt'), num_classes=3)
    trainer_instance = load_best_models([models_filename], args)[0]
    # loss, accuracies, outputs, output_trees = trainer_instance.test(test_dataset)
    # test_acc = torch.mean(accuracies)
    # print("\n-------\n")
    # print(loss, test_acc)
    # print("\n-------\n")
    # return test_acc
    test_trees = trainer_instance.predict(test_dataset)
    return test_trees

def save_submission(predictions):
    filename = 'finals/submission.txt'.format(date=datetime.now())
    with open(filename, 'w') as submission_file:
        for sentence in predictions:
            submission_file.write(sentence.get_predicted_labels() + '\n')
    print('Good luck!')


# acc =[]
# for i in range(0,1):
#     acc.append(predict('models/saved_model/models_20171022_2134/model_'+str(i)+'.pth'))
# print(str(acc))
# save_submission(predictions)

predictions = predict('models/saved_model/models_20171022_2222/model_4.pth')
save_submission(predictions)