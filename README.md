# Spike NN Conversion

This is a python, pytorch, and snntorch implementation of the spiking neural networks and network normalisation algorithms described in ["Fast-classifying, high-accuracy spiking deep networks through weight and threshold balancing"](https://ieeexplore.ieee.org/document/7280696). In their paper, the following procedure for adapting an analog neural network to a spiking one was proposed:

 1. Create a deep neural network with ReLU activators across all layers
 2. Set the biases to 0 across all layers and do not let the biases train
 3. Train the network using backpropagation
 4. Replace the ReLU activators with Integrate-and-Fire (IF) neurons
 5. (Optionally) perform weight normalisation to attain minimal loss in accuracy and faster convergence

Two different weight normalisation techniques were proposed: Model-based and data-based normalisation. These algorithms are implemented in `norms.py`. The techniques were used with two different models, which are replicated in the `mnist-snn.ipynb` and `mnist-scnn.ipynb` notebooks. The first model is a simple fully connected deep neural network with layers 784-1200-1200-10 and the second is a convolutional neural network with layers 28x28-12c5-2(avg-pooling)-64c5-2(avg-pooling)-10. Both models were applied to the MNIST dataset - a fairly simple classification task.

The original code was written in and for matlab, which I did not want to touch so I guess this is where we're at.

## How to use this code

I recommend creating a new python virtual environment for this project. I use pyenv but anything works fine. The steps are tested in python 3.13.1.

 1. Clone this repository somewhere using `git clone https://github.com/joshiteoshi/spike_nn_conversion.git`
 2. Go to the new directory `cd spike_nn_conversion`
 3. Make a new `params` directory using `mkdir params`
 4. Install the dependencies (ideally in a virtual environment) using `pip install requirements.txt`

From there, load up the notebooks in whatever environment you are familiar with.

## `mnist-snn` and `mnist-scnn`

The structure of each notebook is roughly the same. First is code to train the ordinary ReLU version of each network, followed by a basic spiking model without weight normalisation. Then the algorithms to perform model and data normalisation are used to get normalised versions of each model. So that the models do not have to be retrained or renormalised on every run, weights are stored to and loaded from a `params` directory.

Some implementation details:
 - The IF neurons used are snntorch's `Leaky` class with `beta=1`. This means membrane voltage does not decay.
 - I believe the original paper uses a simple SGD optimiser. However, this did not train well here. Thus, pytorch's `Adam` was used instead.