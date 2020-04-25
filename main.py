############################################################
# Main
# ----------------------------------------------------------
# This file is the main()
############################################################

import address_config
from evaluation_F_score import max_score
from argparse import ArgumentParser
from preprocessing import data_access
from evaluation import evaluate
from cnn import train
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# define the ArgumentParser type for split spring of
# training/evaluation data
def int_range(string):
    string1, string2 = string.split(':')
    return range(int(string1), int(string2))


def main():
    parser = ArgumentParser(description=
                            'CNN ONSET DETECTION BY YUXUAN YANG')
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='select training epochs'
    )
    fun_select = \
        parser.add_mutually_exclusive_group(required=True)
    fun_select.add_argument(
        '--train',
        type=int_range,
        help='select training models'
    )
    fun_select.add_argument(
        '--evaluate',
        type=int_range,
        help='select evaluation models'
    )
    arg = parser.parse_args()

    # import processed data
    # data type:
    # data[8] [files] [(AudioSample)]
    #
    data = data_access()

    print(
        'CREATED DATASET WITH SIZE %s.' % list(map(len, data)))

    # select function:
    if arg.evaluate:
        evaluate(data, arg.evaluate, print_flag=0)
        thresholdbuffer = address_config.peak_threshold
        max_score(data, arg.evaluate)
    else:
        train(data, arg.train, arg.epochs)
if __name__ == '__main__':
    main()
