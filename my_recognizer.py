import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
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
    for test_word_idx, test_word_features in test_set.get_all_Xlengths().items():
        test_word = test_set.wordlist[test_word_idx]
        scores = []
        for word, model in models.items():
            try:
                logL = model.score(test_word_features[0], test_word_features[1])
                scores.append((word, logL))
            except:
                continue
        print('Test word {}'.format(test_word))
        import pdb
        pdb.set_trace()

    # TODO implement the recognizer
    # return probabilities, guesses
    raise NotImplementedError
