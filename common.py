import decimal
from math import exp

decimal.getcontext().prec = 10


class Common(object):

    @staticmethod
    def sigmoid(theta_list, X):
        z = decimal.Decimal(0.0)
        for i in range(len(theta_list)):
                z += decimal.Decimal(X[i]) * theta_list[i]
        return decimal.Decimal(1/(1 + exp(-z)))

    @staticmethod
    def feature_scaling(data_list, m, n):
        column_dict = {}
        for i in range(n):
            column_dict[i] = []
            for j in range(m):
                column_dict[i].append(data_list[j][i])
        for column_no, data in column_dict.items():
            for i in range(m):
                s = max(data) - min(data)
                avg = sum(data)/len(data)
                data_list[i][column_no] = float((data_list[i][column_no]-avg))/s
        return data_list
