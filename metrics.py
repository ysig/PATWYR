import numpy as np
import jiwer
import jiwer.transforms as tr

class SentencesToListOfCharacters(tr.AbstractTransform):
    def process_string(self, s):
        return list(s)

    def process_list(self, inp):
        chars = []

        for sentence in inp:
            chars.extend(self.process_string(sentence))

        return chars


class Bartospace(tr.AbstractTransform):
    def process_string(self, s):
        return s.replace("|", " ")

    def process_list(self, inp):
        chars = []

        for sentence in inp:
            chars.extend(self.process_string(sentence))
        print(chars)
        return chars


class Metrics(object):
    def __init__(self):
        self.cer_transform = tr.Compose(
            [
                Bartospace(),
                tr.RemoveMultipleSpaces(),
                tr.Strip(),
                SentencesToListOfCharacters(),
            ]
        )

        self.wer_transform = tr.Compose(
            [
                SentencesToListOfCharacters(),
                tr.RemoveMultipleSpaces(),
                tr.Strip(),
            ]
        )


    def cer(self, predictions, references):
        return np.mean([jiwer.wer(r, p, truth_transform=self.cer_transform, hypothesis_transform=self.cer_transform) for p, r in zip(predictions, references)])

    def wer(self, predictions, references):
        return np.mean([jiwer.wer(r, p, truth_transform=self.wer_transform, hypothesis_transform=self.wer_transform) for p, r in zip(predictions, references)]