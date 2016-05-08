# Deep Learning Resources
Resources for deep learning: papers, articles, courses

## Overview
[Schmidhuber, J. (2015). Deep learning in neural networks: An overview. Neural Networks, 61, 85–117.](http://arxiv.org/abs/1404.7828)

## Neural Networks

[Michael Nielsen book on NN](http://neuralnetworksanddeeplearning.com/chap1.html)

[Hacker's guide to Neural Networks. Andrej Karpathy blog](http://karpathy.github.io/neuralnets/)

[Visualize NN training](http://experiments.mostafa.io/public/ffbpann/)

## Backpropagation

[A Gentle Introduction to Backpropagation. Sathyanarayana (2014)](http://numericinsight.com/uploads/A_Gentle_Introduction_to_Backpropagation.pdf)

[Learning representations by back-propagating errors. Hinton et al, 1986](http://www.nature.com/nature/journal/v323/n6088/abs/323533a0.html)
Seminal paper by Hinton et al on back-propagation.

[The Backpropagation Algorithm](http://page.mi.fu-berlin.de/rojas/neural/chapter/K7.pdf)
Longer tutorial on the topic, 34 pages

[Overview of various optimization algorithms](http://sebastianruder.com/optimizing-gradient-descent/)
Good Blog article on different GD algorithms

## Recurrent Neural Network (RNN)
[Blog intro, tutorial](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/)

[Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. 2014. Cho et al, (Bengio group)](http://arxiv.org/abs/1406.1078)

[Character-Aware Neural Language Models. Kim et al, 2015.](http://arxiv.org/pdf/1508.06615.pdf)

[The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
Indepth, examples in vision and NLP. Provides code

[Sequence to Sequence Learning with Neural Networks. Sutskever et al (2014)](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)
Ground-breaking work on machine translation with RNN and LSTM

[Understanding Natural Language with Deep Neural Networks Using Torch (2015)](http://devblogs.nvidia.com/parallelforall/understanding-natural-language-deep-neural-networks-using-torch/)
See part on predicting next word with RNN.

[LSTM BASED RNN ARCHITECTURES FOR LARGE VOCABULARY SPEECH RECOGNITION](http://arxiv.org/pdf/1402.1128v1.pdf)
Google paper

[Awesome Recurrent Neural Networks](https://github.com/kjw0612/awesome-rnn#lectures)
Curated list of RNN resources

## Convolutional Neural Network (CNN, or ConvNet)

[Collobert. Natural Language Processing (Almost) from Scratch (2011)](http://dl.acm.org/citation.cfm?id=2078186)
Important paper that spurred interest in applying CNN to NLP.

[Multichannel Variable-Size Convolution for Sentence Classification. Yin, 2015](https://aclweb.org/anthology/K/K15/K15-1021.pdf)
Interesting, borrows multichannel from image CNN, where each channel is a different word embedding.

[A CNN for Modelling Sentences. Kalchbrenner et al, 2014](http://phd.nal.co/papers/Kalchbrenner_DCNN_ACL14)
Dynamic k-max pooling for variable length sentences. 

[Semantic Relation Classification via Convolutional Neural Networks with Simple Negative Sampling. Xu et al, 2015](http://arxiv.org/pdf/1506.07650v1.pdf)

[Text Understanding from Scratch. Zhang, LeCunn. (2015)](http://arxiv.org/abs/1502.01710)

[Kim. Convolutional Neural Networks for Sentence Classification (2014)](http://arxiv.org/pdf/1408.5882v2.pdf)
[Sensitivity Analysis of (And Practitioner's Guide to) CNN for Sentence Classification. Zhang, Wallace (2015)](http://arxiv.org/pdf/1510.03820v2.pdf)
[-Annotated](https://drive.google.com/open?id=0ByV7wn2NzevOY25JNlJQREVLZEU)

[Nguyen, Grishman. Relation Extraction: Perspective from Convolutional Neural Networks (2015)](http://www.cs.nyu.edu/~thien/pubs/vector15.pdf)
[Annotated version](https://drive.google.com/file/d/0ByV7wn2NzevObzAtV1QyUDl5X2M/view?usp=sharing)

[Convolutional Neural Network for Sentence Classification. Yahui Chen, 2015](https://uwspace.uwaterloo.ca/bitstream/handle/10012/9592/Chen_Yahui.pdf?sequence=3&isAllowed=y)
Chen's Master's thesis, University of Waterloo

## Deep Reinforcement Learning
[Playing Atari with Deep Reinforcement Learning. Mnih et al. (2014)](http://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
[Youtube Demo](https://www.youtube.com/watch?v=wfL4L_l4U9A)

## Other applications of DL
[Evolving Neural Networks through Augmenting Topologies. Stanley, Miikkulainen (2002)](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)

[Implementation of Evolutionary Algorithms for Deep Architectures. Sreenivas Sremath Tirumala (2014)](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.664.6933)

[DL in Finance](http://arxiv.org/pdf/1602.06561v2.pdf)

## General Topics

[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. Ioffe & Szegedy, 2015](http://arxiv.org/abs/1502.03167)
[Annotated](https://drive.google.com/open?id=0ByV7wn2NzevOSW9jVC14VEpSUHc)

[Dropout: A Simple Way to Prevent NNs from Overfitting. Srivastava, Hinton et al. 2014](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
Dropout is the most popular method to regularize CNN to prevent overfitting. A dropout layer will stochastically disable some neurons to force them to learn useful features individually, rather than relying on others. 

## Online Courses

[Deep Learning. Udacity, 2015](https://www.udacity.com/course/deep-learning--ud730)
This course is quite brief and does no go deeply into any topic. It is more about getting a feel for DL and specifically about using TensorFlow for DL.

[Convolutional Neural Networks for Visual Recognition. Stanford, 2016](http://cs231n.stanford.edu/)

[Neural Network Course. Université de Sherbrooke, 2013](http://info.usherbrooke.ca/hlarochelle/neural_networks/description.html)

[Machine Learning Course, University of Oxford(2014-2015)](https://www.cs.ox.ac.uk/people/nando.defreitas/machinelearning/)

[Deep Learning for NLP, Stanford (2015)](http://cs224d.stanford.edu/)
Click "syllabus" for full material

[Stanford Deep Learning tutorials](http://ufldl.stanford.edu/tutorial/)
From basics of Machine Learning, to DNN, CNN, and others. 
Includes code. 

Bengio 3 part lecture series on DL
Part [1](https://www.youtube.com/watch?v=JuimBuvEWBg), [2](https://www.youtube.com/watch?v=Fl-W7_z3w3o), [3](https://www.youtube.com/watch?v=_cohR7LAgWA)

## Books

[Yoshua Bengio, Ian Goodfellow and Aaron Courville (2015). Deep Learning. Book in preparation for MIT Press.](http://www.deeplearningbook.org)

## Lecture Notes
[Natural Language Understanding with Distributed Representation](http://arxiv.org/pdf/1511.07916v1.pdf)
Video Lectures: http://techtalks.tv/natural-language-processing-nyu/

[A Primer on Neural Network Models for Natural Language Processing] (http://u.cs.biu.ac.il/~yogo/nnlp.pdf)

## Other Reading Lists
[DeepLearning.net's list]
(http://deeplearning.net/reading-list/)

## Tools
[TensorFlow](https://www.tensorflow.org), [white paper](http://download.tensorflow.org/paper/whitepaper2015.pdf)

[Torch](http://torch.ch)
-[Learn Lua in 15 minutes](http://tylerneylon.com/a/learn-lua/)

[Deeplearning4j](http://deeplearning4j.org)

[Theano](http://deeplearning.net/software/theano/)
