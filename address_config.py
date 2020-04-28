########################################################################################################################
# Address_config
# ----------------------------------------------------------------------------------------------------------------------
# This file defined the address of the data_dir, cache_dir and models_dir
########################################################################################################################
"""
music_Oxford_Fuchs
(above is the combined_pure_violin version without fuzzier)
onsets_ISMIR_2012
combined_all
combined_pure_violin
more options:
combined_pure_violin_FiveLayers
combined_pure_violin_NineLayers
combined_pure_violin_NoDense

"""
from os.path import join

current_class = 'music_Oxford_Fuchs'
model_class = 'music_Oxford_Fuchs'

"""

"""
data_dir = join(r'D:\projectRESEARCH\CNN\softonsetdetection\data', current_class)
cache_dir = join(r'D:\projectRESEARCH\CNN\softonsetdetection\cache', model_class)
model_dir = join(r'D:\projectRESEARCH\CNN\softonsetdetection\models', model_class)
eval_dir = join(r'D:\projectRESEARCH\CNN\softonsetdetection\models',model_class+'eval_cache.pkl')
figure_dir = join(r'D:\projectRESEARCH\CNN\softonsetdetection\figure\peak_threshold', model_class + '.png')

seed = '1234321'

# Used in madmom.evaluation.onsets.OnsetEvaluation

evaluationwindow = 0.025  # F-measure evaluation window
evaluationcombine = 0.03  # combined all onsets within such seconds
peak_threshold = 0.67
#0.7  0.78710
#0.3 0.87558
#0.2 0.86605
# Note that evaluate data is saved in model folder
# while the best F_score data is saved in cache folder
"""
combined_all Peak F 0.79006 threshold 0.78
combined_pure_violin Peak F 0.84218 threshold 0.67
combined_pure_violin_NineLayers Peak F 0.77951 threshold 0.500
combined_pure_violin_NoDense Peak F 0.60759 treshold 0.320
music_Oxford_Fuchs Peak F 0.83372 treshold 0.17

threshold =0.54
"""
