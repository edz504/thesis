# Project 2
### Extending LSTM Nets with Nonparametric Bases for Affordance Learning in Vision-Based Autonomous Driving

#### Motivation
Chenyi recently modified LSTM implementations in Caffe used in previous work (TODO(Eddie): find authors) to make them suitable for our TORCS dataset.  As work from Prof. Liu has shown (paper pending), extending activation functions in deep networks with nonparametric bases greatly expands the model exploration space, and can enable much more predictively powerful models.  We look to apply these extensions to the LSTM pipeline created by Chenyi.

#### Implementation
We implement Nonparametric Rectified Linear Unit layers within Caffe with two different basis expansions: sigmoid and Fourier.  Specifically, this means a forward pass would consist of the following:

f1(x) = max(0, x) + beta * sigmoid(x)
f2(x) = max(0, x) + beta1 * sin(x) + beta2 * cos(x)

where the betas are learnable weights within the layer.  After implementing the forward and backward CPU and GPU versions of these layers, we proceed with the training.  This consists of a parametric initialization of the network for 50,000 iterations (until validation loss has converged), followed by another 50,000 iterations of the nonparametric version of the network, initialized with the weights from the first 50,000 parametric iterations.

#### Results
![Nonparametric Validation Loss][results]

Neither our sigmoid nor Fourier basis expansions provide massive drops in validation loss.  As Prof. Liu notes, however, they also do not result in an overfitting and increase in validation loss.  This simply indicates that these nonparametric expansions are not structurally beneficial for the TORCS data.

[results]: https://github.com/edz504/project2/results/np_together_50_init50.png "Nonparametric Validation Loss"


#### Code
Our modified version of the Caffe master directory is located in [a separate Github repo](https://github.com/edz504/lstm_dd/).  Specific file additions and modifications are listed in `changes.txt` within this repository, or `eddie_changes.txt` within `lstm_dd`.  Visualization code and results are provided in `results`.