########################################################################################################################
# Address_config
# ----------------------------------------------------------------------------------------------------------------------
# This file defined the address of the data_dir, cache_dir and models_dir
########################################################################################################################
data_dir = r'D:\projectRESEARCH\CNN\softonsetdetection\data\music_Oxford_Fuchs'
"""
data1: \data\onsets_ISMIR_2012
data2: \data\music_Oxford_Fuchs
"""
cache_dir = r'D:\projectRESEARCH\CNN\softonsetdetection\cache\cache_music_Oxford_Fuchs'
"""
cache_dir1: 'cache\cache_onsets_ISMIR_2012'
cache_dir2: cache\cache_music_Oxford_Fuchs
"""
model_dir = r'D:\projectRESEARCH\CNN\softonsetdetection\models\model_music_Oxford_Fuchs'
"""
model_dir1: models\model_onsets_ISMIR_2012
model_dir2: models\model_music_Oxford_Fuchs
"""
figure_dir=r'D:\projectRESEARCH\CNN\softonsetdetection\figure\peak_threshold'
seed = '1234321'

# Used in madmom.evaluation.onsets.OnsetEvaluation

evaluationwindow = 0.025  # F-measure evaluation window
evaluationcombine = 0.03  # combined all onsets within such seconds
peak_threshold = 0.184
"""
threshold =0.54
"""