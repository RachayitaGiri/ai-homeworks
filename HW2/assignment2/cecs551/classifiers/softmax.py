import numpy as np
from random import shuffle 

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  '''for i in range(num_train):
        scores = X[i] @ W
        scores -= np.max(scores)
        denom = np.sum(np.exp(scores))
        for j in range(num_classes):
            prob = np.exp(scores[j]) / denom  # normalized probability for j
            log_loss = -np.log(prob)          # log loss for class j
            if (j==y[i]):
                dW[:,y[i]] += (prob - 1) * X[i]   # gradient for correct class
            else:
                dW[:,j] += prob * X[i]            # gradient for incorrect class
        loss += log_loss    # Li - loss for ith sample, summed over all samples
  loss /= num_train                           
  loss += reg * np.sum(W * W)                 
  dW /= num_train
  dW += reg * np.sum(W)'''
  for i in range(num_train):
    # loss
    scores = X[i].dot(W)
    # shift values for 'scores' for numeric reasons (over-flow cautious)
    scores -= scores.max()
    scores_expsum = np.sum(np.exp(scores))
    cor_ex = np.exp(scores[y[i]])
    loss += - np.log( cor_ex / scores_expsum)
    # grad
    # for correct class
    dW[:, y[i]] += (-1) * (scores_expsum - cor_ex) / scores_expsum * X[i]
    for j in range(num_classes):
        # pass correct class gradient
        if j == y[i]:
            continue
        # for incorrect classes
        dW[:, j] += np.exp(scores[j]) / scores_expsum * X[i]
  loss /= num_train
  loss += reg * np.sum(W * W)
  dW /= num_train
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X @ W
  scores -= np.max(scores)
  pos = (range(num_train), y)
  temp1 = -scores[np.arange(num_train), y]
  temp2 = np.log(np.sum(np.exp(scores), axis=1))
  loss = np.sum(temp1) + np.sum(temp2)
  loss /= num_train
  loss += reg * np.sum(W*W)
    
  '''prob = np.exp(scores) / np.sum(np.exp(scores))
  prob[np.arange(num_train), y] += 1 
  dW = X.T @ prob
  dW /= num_train
  dW += reg * np.sum(W)'''
  
  '''exp_scores = np.exp(scores)
  sums = np.sum(exp_scores,axis=1)
  DlossDscores = exp_scores / (num_train * np.matrix(sums).T)
  DlossDscores[range(num_train),y] -= (1.0/num_train)
  dW = X.T @ DlossDscores
  dW /= num_train
  dW += reg * np.sum(W)'''
  scores = X.dot(W)
  scores -= scores.max()
  scores = np.exp(scores)
  scores_sums = np.sum(scores, axis=1)
  cors = scores[range(num_train), y]
  s = np.divide(scores, scores_sums.reshape(num_train, 1))
  s[range(num_train), y] = - (scores_sums - cors) / scores_sums
  dW = X.T.dot(s)
  dW /= num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

