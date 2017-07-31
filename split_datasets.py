from sklearn.model_selection import train_test_split


def split_dataset_simple(dataset,train_dataset,dev_dataset, test_dataset,split=(0.1,0.1)):
    # splitting_point = int(len(dataset) * split)
    dev_split_start = int((1 - split[0] - split[1])*len(dataset))
    test_split_start = int((1 - split[1]) * len(dataset))
    train_dataset.trees, dev_dataset.trees,test_dataset.trees = dataset.trees[:dev_split_start], dataset.trees[dev_split_start:test_split_start],dataset.trees[test_split_start:]
    train_dataset.sentences, dev_dataset.sentences,test_dataset.sentences = dataset.sentences[:dev_split_start], dataset.sentences[dev_split_start:test_split_start],dataset.sentences[test_split_start:]
    train_dataset.labels, dev_dataset.labels,test_dataset.labels = dataset.labels[:dev_split_start], dataset.labels[dev_split_start:test_split_start],dataset.labels[test_split_start:]
    return train_dataset, dev_dataset, test_dataset


def split_dataset_random(dataset, train_dataset, dev_dataset,test_size=0.1):
    X_train, X_dev, train_dataset.labels, dev_dataset.labels = train_test_split(
        [(x, y) for x, y in zip(dataset.trees, dataset.sentences)], dataset.labels, test_size=test_size, random_state=0)
    train_dataset.trees, dev_dataset.trees = [x[0] for x in X_train], [x[0] for x in X_dev]
    train_dataset.sentences, dev_dataset.sentences = [x[1] for x in X_train], [x[1] for x in X_dev]
    return train_dataset, dev_dataset


def split_dataset_kfold(X,y,train_index,test_index,train_dataset,dev_dataset):
    train_dataset.trees, dev_dataset.trees = [x[0] for x in X[train_index].tolist()], [x[0] for x in X[test_index].tolist()]
    train_dataset.sentences, dev_dataset.sentences = [x[1] for x in X[train_index].tolist()], [x[1] for x in
                                                                                               X[test_index].tolist()]
    train_dataset.labels, dev_dataset.labels = y[train_index].tolist(), y[test_index].tolist()
    return train_dataset, dev_dataset