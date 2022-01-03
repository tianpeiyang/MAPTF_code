import os
import json
import numpy as np


class OutputJson:
    def __init__(self, data_field=[]):
        self.data_field = data_field
        self.data = {}
        for i in range(len(data_field)):
            if not isinstance(data_field[i], str):
                raise Exception('the data field must be type of string: ' + str(data_field[i]))

            self.data[data_field[i]] = []

    def update(self, value, key=None):
        if key is not None:
            if isinstance(value, bool):
                value = str(value)
            self.data[key].append(value)
            return
        if isinstance(value, tuple) or isinstance(value, list):
            if len(value) != len(self.data_field):
                raise Exception('Error in parameters size: ' + str(value))
            for i in range(len(value)):
                if type(value[i]) is np.bool_ or type(value[i]) is np.bool or type(value[i]) is bool:
                    self.data[self.data_field[i]].append(str(value[i]))
                else:
                    self.data[self.data_field[i]].append(value[i])

    def print_first(self):
        if self.data == {}:
            return
        for i, key in enumerate(self.data_field):
            print(key, ": %s, " % self.data[key][len(self.data[key]) - 1], end=' ')
        print()

    def print_by_key(self, key, index=None):
        if index is None:
            print(key, ": ", self.data[key])
        else:
            print(key, " ", index, ": ", self.data[key][index])

    def save(self, path, filename, field=None):
        if not os.path.exists(path):
            os.makedirs(path)
        if field is None:
            field = self.data_field
        out = {}
        for key in field:
            if len(self.data[key]) > 0 and type(self.data[key][0]) is np.ndarray:
                out[key] = [a.tolist() for a in self.data[key]]
            else:
                out[key] = self.data[key]
        with open(path + "/" + filename + ".json", "w") as f:
            json.dump(out, f)
