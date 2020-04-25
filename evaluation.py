########################################################################################################################
# Evaluation
# ----------------------------------------------------------------------------------------------------------------------
# This file is the evaluation test
#
########################################################################################################################

from os.path import join
import numpy as np

from tensorflow.keras.models import load_model
from madmom.features.onsets import peak_picking
from madmom.evaluation.onsets import OnsetEvaluation
from madmom.utils import combine_events

import address_config


def map_in_audio_sample(d):
    """
    cut the evaluation audio to 15 frames/map
    """
    return np.array([d.pd[i:i + 15] for i in range(len(d.pd) - 14)])


def pickingpeck(y):
    """
    rather than use grid search we set treshold 0.54 and 5 frames for smooth
    :param y:
    :return:
    """
    onsets = peak_picking(y,
                          threshold=address_config.peak_threshold,
                          smooth=5,
                          pre_avg=0, post_avg=0,
                          pre_max=1.0, post_max=1.0)

    onsets = onsets.astype(np.float) / 100.0

    # this will allow a tolerance of 30%,
    # any onsets within 0.003 range will be considered as same
    onsets = combine_events(onsets, 0.003, 'left')
    return np.asarray(onsets)


def evaluate_audio_sample_map(model, d):
    """
    predict each map
    """
    x = map_in_audio_sample(d)
    y_predict = model.predict(x)
    y_predict = y_predict.squeeze()
    y_predict_peak = pickingpeck(y_predict)
    return OnsetEvaluation(y_predict_peak,
                           d.ra,
                           window=address_config.evaluationwindow,
                           combine=address_config.evaluationcombine)


def evaluate_each_slice(model, slice, print_flag):
    for audio_sample in slice:
        if print_flag == 0:
            print(audio_sample.an, end='\n', flush=True)
        yield evaluate_audio_sample_map(model, audio_sample)


def evaluate_all_slices(slices, int_range, print_flag):
    output_dir = address_config.model_dir
    evals = []
    n_slices = len(slices)
    model_file = 'model_best_val.h5'
    for i in int_range:
        if print_flag == 0:
            print('[%d/%d] EVALUATING,' % (i + 1, n_slices), end='', flush=True)
        model_path = join(output_dir, '%02d' % i, model_file)
        if print_flag == 0:
            print('LOADING %s,' % model_file, end='', flush=True)
        model = load_model(model_path)
        slice = slices[i]
        evals.extend(evaluate_each_slice(model, slice,print_flag))
        if print_flag == 0:
            print('EVALUATE DONE')
    return evals


def sum_evaluation(evals):
    """
    all predicted events of 8 slices
    """
    tp = sum(e.num_tp for e in evals)
    fp = sum(e.num_fp for e in evals)
    tn = sum(e.num_tn for e in evals)
    fn = sum(e.num_fn for e in evals)
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f_measure = 2 * prec * rec / (prec + rec)
    f_score = 'ALL %d FILES\n' % len(evals)
    f_score += ' TP+FP: %6d TP: %6d FP: %5d TN: %5d FN: %5d\n' \
               % (tp + fn, tp, fp, tn, fn)
    f_score += ' PREC: %.3f REC: %.3f F-SCORE: %.5f' \
               % (prec, rec, f_measure)
    return f_score, f_measure


def evaluate(data, int_range, print_flag):
    evals = evaluate_all_slices(data, int_range, print_flag)
    f_score_print, f_score = sum_evaluation(evals)
    if print_flag == 0:
        print(f_score_print)
    return f_score
