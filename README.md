# Senior Thesis
##### Eddie Zhou ORFE '16 
This repository contains code relevant to technical implementation, analysis, and visualization of all my senior thesis work with Professor Han Liu at Princeton University.  The senior thesis work is submitted in partial fulfillment of the requirements of the Operations Research and Financial Engineering department, Statistics and Machine Learning certificate program, and Applications of Computing / Computer Science certificate program.

Each of the three projects is briefly explained below, and each contains a more thorough README in its respective subdirectory.

#### Project 1: Applying Nested Statistical Models for Temporal Affordance Learning in Vision-Based Autonomous Driving
In this work, we build off Chenyi Chen's work, [DeepDriving](http://deepdriving.cs.princeton.edu/), in building predictive models for vision-based autonomous driving.  Chen et al. validated the direct perception paradigm through the use of a convolutional neural network pipeline, and we look to investigate the potential for a model augmented by recurrency.  In other words, we apply rigorous statistical analysis to examine the role of history in the affordance indicator prediction problem.

Note: for brevity, this is the only work formally written into my senior thesis (see `thesis.pdf`).

#### Project 2: Extending LSTM Nets with Nonparametric Bases for Affordance Learning in Vision-Based Autonomous Driving
Chenyi later modified the DeepDriving pipeline to include LSTM (Long Short-Term Memory) modules, thereby implementing the recurrency investigated in Project 1.  In this work, we enhance the network by including various nonparametric basis additive factors into the rectified linear unit layers, hoping the model's increased flexibility can learn better.

#### Project 3: Creating an R package for Deep Learning, Bayesian Optimization, and Nonparametric Augmentation
The `project3` subdirectory contains the foundation for the source code of an R package that ties together deep learning, bayesian optimization, and nonparametric augmentation.  `deepbayes` ties together [Caffe](http://caffe.berkeleyvision.org/), a popular deep learning software package written in C++, [BayesOpt](https://github.com/rmcantin/bayesopt), and some nonparametric ideas as applied to deep neural networks.