import numpy as np
import os
import json
from alg.muti_ptf_ppo.ppo import PPO

def action_equal(action1, action2, continuous_action=None):
    if not continuous_action or continuous_action is None:
        if (isinstance(action1, list) or isinstance(action1, np.ndarray)) and (isinstance(action2, list) or isinstance(action2, np.ndarray)):
            return (np.array(action1) == np.array(action2)).all()
        else:
            return False
    elif continuous_action:
        mean = action1[0]
        std = action1[1]
        for i in range(len(action2)):
            if action2[i] < mean[i] - std[i] or action2[i] > mean[i] + std[i]:
                return False
        return True


def build_source_actor(args, sess, policy_path, i=0):
    par_path = os.path.dirname(policy_path)
    file_name = ''
    for dirPath, dirNames, fileNames in os.walk(par_path):
        # print(fileNames)
        for fileName in fileNames:
            if fileName == 'args.json':
                file_name = fileName
                break
        if file_name != '':
            break
    file_path = par_path + "/" + file_name
    with open(file_path, 'r') as f:
        source_args = json.load(f)
    source_policy = 'ppo'#args['policy']''
    if source_policy == 'ppo':
        return PPO(args['action_dim'], args['features'], source_args, sess, logger=None, i=i)
    else:
        raise Exception('no such source_policy named: ' + str(source_policy))


class OptionToList:
    def __init__(self, num_agent):
        self.num_agent = num_agent
        self.option_list = []
        self.reset()

    def reset(self):
        self.option_list = []
        length = np.power(self.num_agent - 1, self.num_agent)
        for i in range(length):
            self.option_list.append(self.number_converter(i))

    # FIXME option网络输出option index，转换成union option的操作相当于进制转换，例如option_dim=3，option_index=26, union_option=[2, 2, 2]
    def number_converter(self, number):
        hex = self.num_agent
        res = np.zeros(hex)
        index = 0
        while True:
            s = number // (hex - 1)  # 商
            y = number % (hex - 1)  # 余数
            res[index] = y
            if s == 0:
                break
            number = s
            index += 1
        res = list(res)
        res.reverse()
        return res

    def get_option_list(self, i):
        if i >= len(self.option_list):
            assert 'out of option_list memory!'
        return self.option_list[i]
