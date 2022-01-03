import logging
import numpy
import time
import threading

R = threading.Lock()

class Logger:
    def __init__(self, log_name, graph_path, args):
        # 第一步，创建一个logger
        self.logger = self.build_log(log_name)
        self.build_tb_log(graph_path)
        self.args = args
        self.keys = dict()
        if 'summary_output_times' in self.args.keys():
            self.summary_times = self.args['summary_output_times']
        else:
            self.summary_times = 1


    def build_log(self, name):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)  # Log等级总开关
        # 第二步，创建一个handler，用于写入日志文件
        rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
        log_name = name + '/out.log'
        logfile = log_name
        fh = logging.FileHandler(logfile, mode='w')
        fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
        # 第三步，定义handler的输出格式
        formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
        fh.setFormatter(formatter)
        # 第四步，将logger添加到handler里面
        logger.addHandler(fh)
        return logger

    def build_tb_log(self, path):
        from tensorboard_logger import configure, log_value, log_histogram
        configure(path)
        self.tb_logger = log_value
        self.tb_h_logger = log_histogram

    def write_tb_log(self, key, value, t):
        if self.args['output_graph']:
            if t % self.summary_times != 0:
                return
            #print(key, value)
            if type(value) is numpy.ndarray or type(value) is list:
                R.acquire()
                if key not in self.keys.keys():
                    self.keys[key] = 0
                else:
                    self.keys[key] += 1
                #print(type(value), key, value, self.keys[key])
                self.tb_h_logger(key, value, self.keys[key])
                R.release()
            else:
                R.acquire()
                if key not in self.keys.keys():
                    self.keys[key] = 0
                else:
                    self.keys[key] += 1
                #print(key, value, self.keys[key])
                self.tb_logger(key, value, self.keys[key])
                R.release()
        else:
            return


    def write_log(self, msg, type='info'):
        if type == 'debug':
            self.logger.debug(msg)
        elif type == 'info':
            self.logger.log(msg)
        elif type == 'warning':
            self.logger.warning(msg)
        elif type == 'error':
            self.logger.error(msg)
        elif type == 'critical':
            self.logger.critical(msg)

