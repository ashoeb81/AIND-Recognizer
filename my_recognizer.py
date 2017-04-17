import numpy as np
import pandas as pd
import warnings

from asl_data import SinglesData
from operator import itemgetter

def recognize(models: dict, test_set: SinglesData, unigram_lm=None):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :param unigram_lm: dict mapping from word to its log-likelihood (log prior probability).
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # Loop over each test word.
    for test_word_idx, test_word_features in test_set.get_all_Xlengths().items():
        # Dictionary to store scores of each model.
        scores_dict = {}
        for word, model in models.items():
            # If using a unigram language model (unigram_lm) then get the
            # prior probability of this word otherwise set to zero.
            prior_logL = 0
            if unigram_lm:
                # Words with additional integers (e.g. GO, GO1, GO2) all get the same log prior probability.
                word_no_digits = ''.join([c for c in word if not c.isdigit()])
                prior_logL = unigram_lm[word_no_digits]
            # If features can be scored then save results.
            try:
                logL = model.score(test_word_features[0], test_word_features[1])
                scores_dict[word] = logL + prior_logL
            except:
                scores_dict[word] = -np.inf + prior_logL
        # Append dictionary of scores.
        probabilities.append(scores_dict)
        # Find word best guess
        best_guess = sorted(scores_dict.items(), key=itemgetter(1), reverse=True)[0][0]
        guesses.append(best_guess)

    # Return probabilities and best guesses
    return probabilities, guesses
