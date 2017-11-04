import warnings
from operator import itemgetter
import numpy as np

from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key is a word and value is Log Likelihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    for i in list(test_set.get_all_sequences().keys()):
        scores = list()
        X, lengths = test_set.get_item_Xlengths(i)
        for word, gaussian in models.items():
            log_l = -np.inf
            try:
                log_l = gaussian.score(X, lengths)
            except:
                pass

            scores.append((word, log_l))

        scores = sorted(scores, key=itemgetter(1), reverse=True)  # sort by log value; descending

        probabilities.append(dict(scores))  # add sorted probabilities as dict
        guesses.append(scores[0][0])  # select first word

    return probabilities, guesses
