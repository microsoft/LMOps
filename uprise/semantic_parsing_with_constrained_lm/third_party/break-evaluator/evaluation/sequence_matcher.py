
import spacy

from edit_distance import SequenceMatcher
from tqdm import tqdm


class SequenceMatchScorer(object):
    def __init__(self, remove_stop_words):
        self.parser = spacy.load('en_core_web_sm', disable=['ner'])
        self.remove_stop_words = remove_stop_words
        # TODO: extend the default stop words list?

    def clean_base(self, text):
        parsed = self.parser(text)

        res = []
        for i in range(len(parsed)):
            if not self.remove_stop_words or not parsed[i].is_stop:
                res.append(parsed[i].lemma_)

        return res

    @staticmethod
    def clean_structural(text):
        return [token for token in text.split(' ') if token.startswith('@@')]

    def get_match_score(self, prediction, target, processing="base"):
        assert processing in ["base", "structural"]

        if processing == "structural":
            prediction_clean = self.clean_structural(prediction)
            target_clean = self.clean_structural(target)
            if prediction_clean == [] and target_clean == []:
                return 1.0
        else:
            prediction_clean = self.clean_base(prediction)
            target_clean = self.clean_base(target)

        sm = SequenceMatcher(a=prediction_clean, b=target_clean)

        # editdistance workaround on empty sequences
        if not prediction_clean and not target_clean:
            return 1
        return sm.ratio()

    def get_match_scores(self, predictions, targets, processing):
        scores = []

        num_examples = len(predictions)
        for i in tqdm(range(num_examples)):
            score = self.get_match_score(predictions[i], targets[i], processing)
            scores.append(score)

        return scores
