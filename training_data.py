from common import Common
import copy


TRAINING_DATA = [
    [02, 23, 15, ],
    [04, 00, 17, ],
    [03, 02, 35, ],
    [05, 01, 20, ],
    [04, 22, 20, ],
    [02, 23, 15, ],
    [04, 00, 17, ],
    [03, 02, 35, ],
    [05, 01, 20, ],
    [04, 22, 20, ],
    [02, 22, 25, ],
    [10, 12, 30, ],
    [12, 13, 25, ],
    [05, 14, 18, ],
    [11, 12, 18, ],
    [02, 22, 25, ],
    [10, 12, 30, ],
    [12, 13, 25, ],
    [05, 14, 18, ],
    [11, 12, 18, ],
    [15, 15, 35, ],
    [7, 1, 20, ],
    [25, 13, 05, ],
    [15, 19, 15, ],
    [12, 15, 05, ],
    [8, 10, 07, ],
    [15, 15, 25, ],
    [10, 16, 15, ],
    [12, 20, 17, ],
    [9, 11, 28, ],
    [4, 10, 15, ],
    [14, 12, 5, ],
]

P = 0
L = 1
R = 2

Y_LIST = [P, P, P, P, P, P, P, P, P, P, L, L, L, L, L, L, L, L, L, L, L, P, R, R, R, R, R, R, R, R, R, R, ]

Y = [P, L, R]

CLASS_MAPPING = {0: 'Porn', 1: 'Learning', 2: 'Random'}

PREDICTION_DATA = [
    [4, 23, 23, ],
    [10, 12, 10],
    [15, 13, 5],
    [5, 01, 25],
]


class TrainingDataSet(object):

    def __init__(self):
        self.training_data = TRAINING_DATA
        self.m = len(self.training_data)
        self.n = len(self.training_data[0])
        self.scaled_data = Common.feature_scaling(copy.deepcopy(self.training_data), self.m, self.n)
        self.hypothesis_count = len(Y)
        self.prediction_list = PREDICTION_DATA

    @staticmethod
    def prepare_training_data():
        y_list = []
        for index in range(len(Y)):
            temp_list = []
            for ele in Y_LIST:
                new_value = 1
                if not ele == Y[index]:
                    new_value = 0
                temp_list.append(new_value)
            y_list.append(temp_list)
        return y_list

    @staticmethod
    def get_class(value):
        return CLASS_MAPPING[value]


if __name__ == '__main__':

    train_obj = TrainingDataSet()
    print train_obj.scaled_data