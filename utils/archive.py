#-*-coding:utf8-*-

import pickle


def pickle_save(fname, data):
    with open(fname, 'wb') as fout:
        pickle.dump(data,fout)


def pickle_load(fname):
    data = None
    with open(fname, 'rb') as fin:
        data = pickle.load(fin)
    return data
