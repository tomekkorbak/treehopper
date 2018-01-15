treehopper
==================================

treehopper is a Tree-LSTM-based dependency tree sentiment labeler, implemented in [PyTorch](https://github.com/pytorch/pytorch) and optimized for morphologically rich languages with relatively loose word order (such as Polish).

treehopper was originally developed as a submission for [PolEval 2017](http://poleval.pl/), a SemEval-inspired NLP evaluation contest for Polish. It scores 0.80 accuracy on PolEval task 2 evaluation dataset. For more details see paper accompanying this submission: [Fine-tuning Tree-LSTM for phrase-level sentiment classification on a Polish dependency treebank](https://arxiv.org/abs/1711.01985).

## What the heck are Tree-LSTMs and dependency tree sentiment labeling?

A dependency tree is a linguistic formalism used for describing the structure of sentences. They are parse trees just like constituency trees, but slightly more useful when dealing with languages with complex inflectional structure and relatively loose word order such as Czech, Turkish, or Polish.

Tree sentiment labeling is the task of labeling each phrase (subtree) of a parse tree with its sentiment. [Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment) is one famous dataset for this task, but using constituency trees as its underlying linguistic formalism of choice.

Tree-LSTMs ([Tai et al., 2015](https://arxiv.org/abs/1503.00075)) generalize LSTMs from chain-like to tree-like structures, enabling state-of-the-art tree sentiment labeling. treehopper implements a variant of Tree-LSTMs known as Child-Sum Tree-LSTM, where each node of a tree can have an unbounded number of children and there is no order over those children. This approach is particularly well-suited for dependency trees.

## How to use

First things first:

```bash
git clone git@github.com:tomekkorbak/treehopper.git
```

### Dependencies

Make sure to use Python>=3.5, PyTorch>=0.2 and a Unix-like operating system (sorry, Windows users).

We recommend managing your dependencies using [virtualenv](https://virtualenv.pypa.io/en/stable/) and pip. For instructions on installing an appropriate PyTorch version please refer to [its website](http://pytorch.org/). All other dependencies can be installed by running `pip install -r requirements.txt`.

### Inference using a pre-trained model

We provide a pre-trained model, trained on full PolEval training dataset (excluding evaluation dataset) with default hyperparameters (i.e. those described in the paper).

The script assumes the data to be tokenized and parsed. Specifically, `input_sentences` must be a list of tokenized sentences separated by a newline character. `input_parents` is a list of dependency trees in PolEval format (i.e. each token is assigned with an index of its parent).

```bash
cd treehopper/
curl -o model.pth <<URL WILL BE ADDED HERE>>
python predict --model_path model.pth \
               --input_parents test/polevaltest_parents.txt \
               --input_sentences test/polevaltest_sentence.txt \
               --output output.txt
```

### Evaluating a pre-trained model

By default, evaluation is against PolEval evaluation dataset.

```bash
cd treehopper/
./fetch_data.sh
python evaluate.py --model_path model.pth
```

### Training from scratch

By default, models trained are saved per epoch in `/models/saved_models/`.

```bash
cd treehopper/
./fetch_data.sh
python train.py
```

### Documentation

For a complete API documentation, please run `predict.py`, `train.py`, or `evaluate.py` with `--help` flag.

All flags default to hyperparameters described in the paper.

## Authors

Tomasz Korbak (tomasz.korbak@gmail.com)
Paulina Żak (paulina.zak1@gmail.com)

## How to cite

```
@article{korbakzak2017,
  author    = {Tomasz Korbak and
               Paulina \.Zak},
  title     = {Fine-tuning Tree-LSTM for phrase-level sentiment classification on
               a Polish dependency treebank. Submission to PolEval task 2},
  journal   = {Proceedings of the 8th Language & Technology Conference (LTC 2017)},
  year      = {2017},
  url       = {http://arxiv.org/abs/1711.01985}
}
```

## Acknowledgements

treehopper core code was loosely based on [TreeLSTMSentiment](https://github.com/ttpro1995/TreeLSTMSentiment), which was based on [Tree-LSTM's original Lua implementation](https://github.com/stanfordnlp/treelstm) of [Tai et al., 2015](https://arxiv.org/abs/1503.00075).

