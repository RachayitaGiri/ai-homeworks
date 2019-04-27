import numpy as np
from random import shuffle 

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    # https://stats.stackexchange.com/questions/155088/gradient-for-hinge-loss-multiclass
    num_diff = 0 # the number of classes that do not meet the margin
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        num_diff += 1
        # since score = X[i].W, d(score)/dw = X[i], but this leaves j=y[i]!?
        dW[:,j] += X[i] 
        loss += margin
    # update the gradient for the correct(missing) class. HOW? Explained at the link above
    dW[:,y[i]] += -(num_diff) * X[i]
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  # Add regularization to the loss.
  loss = loss + reg * np.sum(W * W)
  # Average and then regularization of the gradient
  dW /= num_train
  dW += reg * np.sum(W)
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_train = X.shape[0]
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X @ W  # dimensions: N x C
  # use fancy indexing to access the score of the correct class
  margins = np.maximum(0, scores - scores[y] + 1) # also sets margin of y
  # now reset the margin of y to zero, because it has the correct classes for X
  margins[np.arange(num_train), y] = 0
  loss = np.sum(margins)/num_train + reg * np.sum(W*W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  qualified = margins
  qualified[margins > 0]=1
  num_diff = np.sum(qualified, axis=1)
  qualified[np.arange(num_train), y] = -num_diff.T
  dW = X.T @ qualified
  dW /= num_train
  dW += 2 * reg * W
  # https://mlxai.github.io/2017/01/06/vectorized-implementation-of-svm-loss-and-gradient-update.html
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
