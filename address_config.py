########################################################################################################################
# Address_config
# ----------------------------------------------------------------------------------------------------------------------
# This file defined the address of the data_dir, cache_dir and models_dir
########################################################################################################################
data_dir = r'D:\projectRESEARCH\CNN\softonsetdetection\data\onsets_ISMIR_2012'
cache_dir = r'D:\projectRESEARCH\CNN\softonsetdetection\cache'
model_dir = r'D:\projectRESEARCH\CNN\softonsetdetection\models'
seed = '1234321'

# Used in madmom.evaluation.onsets.OnsetEvaluation

evaluationwindow = 0.025  # F-measure evaluation window
evaluationcombine = 0.03  # combined all onsets within such seconds
peak_threshold = 0.54