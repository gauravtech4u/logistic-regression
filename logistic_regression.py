from common import Common
from training_data import TrainingDataSet, Y
import decimal
import math
import copy

decimal.getcontext().prec = 10

THETA_LIST = [decimal.Decimal(-1.00), decimal.Decimal(-1.00), decimal.Decimal(-1.00), ]

ITERATIONS = 7000

LEARNING_RATE = 0.001


class LogisticRegression(object):

    def __init__(self, training_data, m, n):
        self.x = training_data
        self.m = m
        self.n = n
        self.theta_list = THETA_LIST
        self.theta_dict = {}
        self.rate = decimal.Decimal(LEARNING_RATE)
        self.y_list = TrainingDataSet.prepare_training_data()

    def train(self):
        for h_count in range(len(Y)):
            theta_list = copy.copy(self.theta_list)
            for i in range(ITERATIONS):
                    cost = decimal.Decimal(0.00)
                    for j in range(self.m):
                        hypothesis = decimal.Decimal(Common.sigmoid(theta_list, self.x[j]))
                        for k in range(self.n):
                            theta_list[k] = theta_list[k] - \
                                            self.rate*(hypothesis - decimal.Decimal(self.y_list[h_count][j])) \
                                            * decimal.Decimal(self.x[j][k])

                    cost += - decimal.Decimal(1)/self.m * (decimal.Decimal(self.y_list[h_count][j]) \
                            * decimal.Decimal(math.log(Common.sigmoid(theta_list, self.x[j]))) + \
                            (decimal.Decimal(1) - decimal.Decimal(self.y_list[h_count][j])) \
                            * decimal.Decimal(math.log(decimal.Decimal(1.00) - Common.sigmoid(theta_list, self.x[j]))))
                    # you can use this cost to check if gradient descent is properly converging
                    print "Running Iteration %s for Y %s & cost for this iteration is %s" % (i, h_count, cost)
            self.theta_dict[h_count] = copy.copy(theta_list)

    def classify(self, t_data, input_list):
        # predict the class in which input data resides
        t_data.append(input_list)
        scaled_data = Common.feature_scaling(t_data, self.m+1, self.n)
        predicted_value, predicted_class_val = decimal.Decimal(0.0), None
        for classified_value, theta_list in self.theta_dict.items():
            temp_value = Common.sigmoid(theta_list, scaled_data[-1])
            if temp_value > predicted_value:
                predicted_value = temp_value
                predicted_class_val = classified_value
        return predicted_class_val


if __name__ == '__main__':

    train_obj = TrainingDataSet()
    logistic_obj = LogisticRegression(train_obj.scaled_data, train_obj.m, train_obj.n)
    logistic_obj.train()
    for input_matrix in train_obj.prediction_list:
        print input_matrix
        training_data = copy.deepcopy(train_obj.training_data)
        predicted_class_value = logistic_obj.classify(training_data, input_matrix)
        predicted_class = train_obj.get_class(predicted_class_value)
        print predicted_class


