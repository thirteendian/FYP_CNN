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
current_class='combined_pure_violin'
model_class= 'combined_pure_violin_NoDense'

"""

"""
data_dir = join(r'D:\projectRESEARCH\CNN\softonsetdetection\data',current_class)
cache_dir = join(r'D:\projectRESEARCH\CNN\softonsetdetection\cache',model_class)
model_dir = join(r'D:\projectRESEARCH\CNN\softonsetdetection\models',model_class)

figure_dir= r'D:\projectRESEARCH\CNN\softonsetdetection\figure\peak_threshold'+model_class+'.png'


seed = '1234321'

# Used in madmom.evaluation.onsets.OnsetEvaluation

evaluationwindow = 0.025  # F-measure evaluation window
evaluationcombine = 0.03  # combined all onsets within such seconds
peak_threshold = 0.54
"""
threshold =0.54
"""