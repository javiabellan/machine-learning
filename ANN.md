# Artificial neural networks

## Functions

#### Loss functions

In order to train our model, we need to define what it means for the model to be good. Well, actually, in machine learning we typically define what it means for a model to be bad. We call this the cost, or the loss, and it represents how far off our model is from our desired outcome. We try to minimize that error, and the smaller the error margin, the better our model is.

 * **Cross-entropy**: [Read colah's blog](http://colah.github.io/posts/2015-09-Visual-Information/)

## Layers

#### Final layers

 * **Softmax**: Gives a list of values between 0 and 1, that add up to 1

## Models

 * **Softmax regression**: If you want to assign probabilities to an object being one of several different things, softmax is the thing to do, because softmax gives us a list of values between 0 and 1 that add up to 1.
 has two steps: first we add up the evidence of our input being in certain classes, and then we convert that evidence into probabilities.
