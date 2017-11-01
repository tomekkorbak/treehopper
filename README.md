treehopper
==================================

treehopper is a Tree-LSTM-based dependency tree sentiment labeler, implemented in [PyTorch](https://github.com/pytorch/pytorch) and optimized for morphologically rich languages with relatively loose word order (such as Polish).

treehopper was originally developed as a submission for [PolEval 2017](http://poleval.pl/), a SemEval-inspired NLP evaluation contest for Polish. It scores 0.80 accuracy on PolEval task 2 dataset. For more details see paper accompanying this submission: Korbak & Żak, 2017, *Fine-tuning Tree-LSTM for phrase-level sentiment classification on a Polish dependency treebank. Submission to PolEval task 2*.

## What the heck are Tree-LSTMs and dependency tree sentiment labeling?

A dependency tree is a linguistic formalism used for described the structure of sentences. They are parse trees just like constituency trees, but slightly more useful when dealing with languages with complex inflectional structure and relatively loose word order such as Czech, Turkish, or Polish.

Tree sentiment labeling is the task of labeling each phrase (subtree) of a parse tree with its sentiment. [Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment) is one famous dataset for this task, but using constituency trees as its underlying linguistic formalism of choice.

Tree-LSTMs ([Tai et al., 2015](https://arxiv.org/abs/1503.00075)) generalize LSTMs from chain-like to tree-like structures, enabling state-of-the-art tree sentiment labeling. treehopper implements a variant of Tree-LSTMs known as Child-Sum Tree-LSTMwhere each node of a tree can have an unbounded number of children and there is no order over those children. This approach is particularily well-suited for dependency trees.

## Citation

TBA

## Acknowledgements

treehopper core code was loosely based on [TreeLSTMSentiment](https://github.com/ttpro1995/TreeLSTMSentiment), which was based on [Tree-LSTM's original Lua implementation](https://github.com/stanfordnlp/treelstm) of [Tai et al., 2015](https://arxiv.org/abs/1503.00075).

