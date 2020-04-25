############################################################
# Evaluate part
# ----------------------------------------------------------
# This file is specifically used to find the relationship
# between peak_threshold set in address_config.py and F-Score
############################################################
from evaluation import evaluate
import address_config

import numpy as np
from os.path import join, exists
from pickle import load, dump
import matplotlib.pyplot as plt
from progress.bar import IncrementalBar


def max_score(data, argument):
    cache_address = join(address_config.cache_dir, 'cache_Fscore.pkl')
    if exists(cache_address):
        print('BEST SCORE FOUND. LOADING CACHE FROM %s' % cache_address)
        s = load(open(cache_address, 'rb'))
        max_index = s[1].index(max(s[1]))
        print('MAX F-SCORE %.5f APPROACH WHEN PEAK_THRESHOLD=%.3f'
              % (s[1][max_index],s[0][max_index]))
    else:
        print('BEST SCORE NO RECORD. GENERATING CACHE TO %s' % cache_address)
        f_score = []
        threshold = []
        bar = IncrementalBar('PROGRESS', max=100)
        for i in range(0, 100, 1):
            threshold.append(i / 100)
            address_config.peak_threshold = (threshold[i])
            f_score.append(evaluate(data, argument, print_flag=1))
            bar.next()
        bar.finish()
        s = [threshold, f_score]
        max_index = s[1].index(max(s[1]))
        print('MAX F-SCORE %.5f APPROACH WHEN PEAK_THRESHOLD=%.3f'%
              (s[1][max_index], s[0][max_index]))
        dump(s, open(cache_address, 'wb'), protocol=2)


    plt.plot(s[0], s[1])
    plt.xlabel('peaking threshold')
    plt.ylabel('F-score')
    plt.plot(s[0][max_index],s[1][max_index],'ro')
    plt.axis([0, 1, 0, 1])
    plt.savefig(address_config.figure_dir)
    return
