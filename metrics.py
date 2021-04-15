# import numpy as np
# import jiwer
# import jiwer.transforms as tr

# class SentencesToListOfCharacters(tr.AbstractTransform):
#     def process_string(self, s):
#         return list(s)

#     def process_list(self, inp):
#         chars = []
#         for sentence in inp:
#             chars.extend(self.process_string(sentence))
#         return chars

# class Bartospace(tr.AbstractTransform):
#     def process_string(self, s):
#         return s.replace("|", " ")

#     def process_list(self, inp):
#         chars = []
#         for sentence in inp:
#             chars.extend(self.process_string(sentence))
#         return chars


# class Metrics(object):
#     def __init__(self):
#         self.cer_transform = tr.Compose([Bartospace(), tr.RemoveMultipleSpaces(), tr.Strip(), SentencesToListOfCharacters()])
#         self.wer_transform = tr.Compose([tr.RemoveMultipleSpaces(), tr.Strip()])

#     def cer(self, predictions, references):
#         return np.mean([jiwer.wer(r, p, truth_transform=self.cer_transform, hypothesis_transform=self.cer_transform) for p, r in zip(predictions, references)])

#     def wer(self, predictions, references):
#         return np.mean([jiwer.wer(r, p, truth_transform=self.wer_transform, hypothesis_transform=self.wer_transform) for p, r in zip(predictions, references)])
"""
Copied from https://github.com/him4318/Transformer_ocr/blob/master/src/data/evaluation.py
Tool to metrics calculation through data and label (string and string).
 * Calculation from Optical Character Recognition (OCR) metrics with editdistance.
"""

import string
import unicodedata
import editdistance
import numpy as np


def ocr_metrics(predicts, ground_truth):
    """Calculate Character Error Rate (CER), Word Error Rate (WER) and Sequence Error Rate (SER)"""

    if len(predicts) == 0 or len(ground_truth) == 0:
        return (1, 1, 1)

    cer, wer, ser = [], [], []

    for (pd, gt) in zip(predicts, ground_truth):
        pd_cer, gt_cer = list(pd), list(gt)
        dist = editdistance.eval(pd_cer, gt_cer)
        cer.append(dist / (max(len(pd_cer), len(gt_cer))))

        pd_wer, gt_wer = pd.split('|'), gt.split('|')
        dist = editdistance.eval(pd_wer, gt_wer)
        wer.append(dist / (max(len(pd_wer), len(gt_wer))))

        # pd_ser, gt_ser = [pd], [gt]
        # dist = editdistance.eval(pd_ser, gt_ser)
        # ser.append(dist / (max(len(pd_ser), len(gt_ser))))

    metrics = [cer, wer]#, ser]
    metrics = np.mean(metrics, axis=1)

    return metrics