from helper import *
from solution import *
from math import *
import time

#Use for testing the training and testing processes of a model
def train_test_a_model(modelname, train_data, train_label, test_data, test_label, max_iter, learning_rate):
    '''
    you should try various number of max_iter and learning_rate
    '''
    if (modelname == "logistic_regression"):
        print("-------- Model: Logistic Regression ---------")
        
    else:
        print("-------- Model: Third Order Regression ---------")

    w = logistic_regression(train_data, train_label, max_iter, learning_rate)
    acc = accuracy(test_data, test_label, w)
    
    print("Accuracy:", acc, "\n")

def test_logistic_regression():
    '''
    you should try various number of max_iter and learning_rate
    '''
    # get training data
    df_train = load_features("../data/train.txt")
    train_data = df_train[0]
    train_label = df_train[1]
    # get test data
    df_test = load_features("../data/test.txt")
    test_data = df_test[0]
    test_label = df_test[1]
    # train and test the model
    train_test_a_model("logistic_regression", train_data, train_label, test_data, test_label, 1000, 0.01)
    train_test_a_model("logistic_regression", train_data, train_label, test_data, test_label, 10000, 0.01)
    train_test_a_model("logistic_regression", train_data, train_label, test_data, test_label, 1000, 0.5)
    train_test_a_model("logistic_regression", train_data, train_label, test_data, test_label, 10000, 0.5)
    train_test_a_model("logistic_regression", train_data, train_label, test_data, test_label, 1000, 1)
    train_test_a_model("logistic_regression", train_data, train_label, test_data, test_label, 10000, 1)

def test_thirdorder_logistic_regression():
    '''
    you should try various number of max_iter and learning_rate
    '''
    # get the data to be transformed
    df_train = load_features("../data/train.txt")
    df_test = load_features("../data/test.txt")

    # get the third order transformed data
    data = np.append(df_train[0], df_test[0], axis=0)
    data = thirdorder(data)

    # split the training and testing data and labels
    train_data = data[:1561,:]
    train_label = df_train[1]
    test_data = data[1561:, :]
    test_label = df_test[1]

    # train and test the model
    train_test_a_model("thirdorder_regression", train_data, train_label, test_data, test_label, 1000, 0.01)
    train_test_a_model("thirdorder_regression", train_data, train_label, test_data, test_label, 10000, 0.01)
    train_test_a_model("thirdorder_regression", train_data, train_label, test_data, test_label, 1000, 0.5)
    train_test_a_model("thirdorder_regression", train_data, train_label, test_data, test_label, 10000, 0.5)
    train_test_a_model("thirdorder_regression", train_data, train_label, test_data, test_label, 1000, 1)
    train_test_a_model("thirdorder_regression", train_data, train_label, test_data, test_label, 10000, 1)

if __name__ == '__main__':
	test_logistic_regression()
	test_thirdorder_logistic_regression()
