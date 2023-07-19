import numpy as np


def get_attribution_scores(preds):
    ld_score = np.sum(preds[0:10])
    sd_score = np.sum(preds[10:20])
    real_score = np.sum(preds[20:])

    attr_preds = [ld_score, sd_score, real_score]

    return attr_preds
