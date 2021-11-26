#-*-coding:utf8-*-
import logging


class Log():

    def __init__(self, save_dir):
        self.logger=logging.getLogger() #添加日志器
        self.logger.setLevel(level=logging.DEBUG) #设置日志级别
        self.f1 = logging.Formatter(fmt='[time]%(asctime)s %(message)s \n') #设置日志格式
        self.f2 = logging.Formatter(fmt='[time]%(asctime)s %(message)s \n') #设置日志格式
        self.save_dir = save_dir

    def add_StreamHandler(self):
        '''
        添加一个控制台处理器
        :return:
        '''
        self.hand=logging.StreamHandler() #console
        self.hand.setLevel(level=logging.DEBUG)
        #self.hand.setFormatter(self.f1)
        self.logger.addHandler(self.hand)

    def add_FileHandler(self):
        '''
        添加一个文件处理器
        :return:
        '''
        self.filehand=logging.FileHandler(filename='{}'.format(self.save_dir+'log.txt'),encoding='utf-8') #添加文件处理器
        self.filehand.setLevel(level=logging.DEBUG)
        #self.filehand.setFormatter(self.f2)
        self.logger.addHandler(self.filehand)

    def run(self):
        self.add_StreamHandler()
        self.add_FileHandler()
        return self.logger


if __name__=='__main__':
    import pprint
    import yaml
    with open('../config/superpoint_train.yaml', 'r') as fin:
        cfg = yaml.safe_load(fin)
    dict_cfg = pprint.pformat(cfg)
    log = Log('./').run()
    #log.info(f"{dict_cfg}")
    log.info('{}'.format(dict_cfg))
    log.info('hello world')
