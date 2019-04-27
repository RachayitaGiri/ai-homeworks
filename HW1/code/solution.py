import numpy as np 
from math import *
from helper import *
'''
Homework2: logistic regression classifier
'''

def logistic_regression(data, label, max_iter, learning_rate):
	'''
	The logistic regression classifier function.

	Args:
	data: train data with shape (1561, 3), which means 1561 samples and 
		  each sample has 3 features.(1, symmetry, average intensity)
	label: train data's label with shape (1561,1). 
		   1 for digit number 1 and -1 for digit number 5.
	max_iter: max iteration numbers
	learning_rate: learning rate for weight update
	
	Returns:
		w: the seperator with shape (3, 1). You must initialize it with w = np.zeros((d,1))
	'''
	d = data.shape[1]
	w = np.zeros((d, 1))
	N = label.shape[0]
	series_sum = 0

	print("Learning rate:", learning_rate)
	print("Total Iterations:", max_iter)
	print("Training on:", N, "samples")

	for t in tqdm(range(max_iter)):
		series_sum = sum( data[n] * label[n] / (1 + exp(label[n] * w.T @ data[n])) for n in range(N))
		gradient = -series_sum/N
		w = w - learning_rate * gradient.reshape((d,1))
	return w

def thirdorder(data):
	'''
	This function is used for a 3rd order polynomial transform of the data.
	Args:
	data: input data with shape (:, 3) the first dimension represents 
		  total samples (training: 1561; testing: 424) and the 
		  second dimesion represents total features.

	Return:
		result: A numpy array format new data with shape (:,10), which using 
		a 3rd order polynomial transformation to extend the feature numbers 
		from 3 to 10. 
		The first dimension represents total samples (training: 1561; testing: 424) 
		and the second dimesion represents total features.
	'''
	new_data = np.zeros((data.shape[0], 10))

	# for each row, calculate the new features
	for i in range(data.shape[0]):
		sym = data[i][1]
		intensity = data[i][2]
		new_features = [1, sym, intensity, sym**2, sym*intensity, intensity**2, sym**3, (sym**2)*intensity, sym*(intensity**2), intensity**3]
		new_data = np.append(new_data, [new_features], axis=0)
	new_data = new_data[1985:,:]
	return new_data

def accuracy(x, y, w):
    '''
    This function is used to compute accuracy of a logsitic regression model.
    
    Args:
    x: input data with shape (n, d), where n represents total data samples and d represents
        total feature numbers of a certain data sample.
    y: corresponding label of x with shape(n, 1), where n represents total data samples.
    w: the seperator learnt from logistic regression function with shape (d, 1),
        where d represents total feature numbers of a certain data sample.

    Return 
        accuracy: total percents of correctly classified samples. Set the threshold as 0.5,
        which means, if the predicted probability > 0.5, classify as 1; Otherwise, classify as -1.
    '''
    N = y.shape[0]				# number of data points
    correct = 0					# number of correctly classified points

    print("Testing on:", N, "samples")

    for i in range(N):
    	probability = np.dot(w.T, x[i])
    	predicted_class = 1 if (probability >= 0.5) else -1
    	if (predicted_class == y[i]):
    		correct += 1

    accuracy = correct*100/N
    return accuracy
    
