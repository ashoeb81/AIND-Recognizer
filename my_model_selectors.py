import math
from operator import itemgetter
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """

        def compute_bic(logL, num_components):
            """Helper method for computing BIC
            
            Args:
                logL: Log-Likelihood of model being evaluated.
                num_components: Number of hidden states in model
            Returns:
                Float corresponding to the BIC value.
            """
            # To evaluate the BIC, we need the number of parameters in our model.  A Gaussian HMM parameters include
            # P: The initial state distribution which has "num_components" values to estimate.
            # A: The state transition matrix which has "num_components * (num_components-1)" values to estimate.
            # B: The emission probability pdf has "2*num_components" when the covariance is diagonal.
            num_parameters = (num_components-1) + num_components * (num_components - 1) + 2 * num_components
            return -2 * logL + num_parameters * np.log(len(self.lengths))

        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Dictionary that will store mapping between model size and tuple consisting of (model, BIC score)
        results_dict = {}

        # Loop over different model sizes and evaluate BIC.
        for num_components in range(self.min_n_components, self.max_n_components + 1):
            # Train model
            model  = self.base_model(num_components)
            # If training successful and we can score model then record BIC.
            if model:
                try:
                    # Recall that self.X and self.lengths were set by during SelectorBIC construction.
                    logL = model.score(self.X, self.lengths)
                    results_dict[num_components] = (model, compute_bic(logL, num_components))
                except Exception as e:
                    if self.verbose:
                        print("failure to score {} with {} states".format(self.this_word, num_components))

            # Find and return model with *lowest* BIC.
            if len(results_dict):
                model, _ = sorted(results_dict.items(), key=lambda x: x[1][1], reverse=False)[0][1]
                return model

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Dictionary that will store mapping between model size and tuple consisting of (model, DIC score)
        results_dict = {}

        # Loop over different model sizes and evaluate DIC.
        for num_components in range(self.min_n_components, self.max_n_components + 1):
            # Train model
            model  = self.base_model(num_components)
            # If training successful and we can score model then record model DIC.
            if model:
                logL_this_word, logL_other_words, num_other_words = [0., 0., 0.]
                # Loop over all words and update appropriate terms in DIC
                for word, XLength in self.hwords.items():
                    try:
                        logL = model.score(XLength[0], XLength[1])
                        if word == self.this_word:
                            # Update log(P(X(i))
                            logL_this_word += logL
                        else:
                            # Update SUM(log(P(X(all but i)
                            logL_other_words += logL
                            num_other_words += 1
                    except Exception as e:
                        if self.verbose:
                            print("failure to score {} with {} states".format(self.this_word, num_components))
                # DIC includes the term 1/M where M = num_other_words.  Make M > 0
                if num_other_words > 0:
                    results_dict[num_components] = (model, logL_this_word - logL_other_words/num_other_words)

        # Find and return model with largest DIC.
        if len(results_dict) > 0:
            model, _ = sorted(results_dict.items(), key=lambda x: x[1][1], reverse=True)[0][1]
            return model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # Get all word sequences.
        word_sequences = self.words[self.this_word]

        # Do 3-Fold cross validation if we have enough samples.  Otherwise perform 2-Fold CV.
        if len(word_sequences) > 3:
            n_splits = 3
        else:
            n_splits = 2
        split_method = KFold(n_splits=n_splits)

        # Dictionary that will store mapping between model size and CV loglikelihood.
        results_dict = {}

        # Loop over different model sizes and evaluate CV log-Likelihood.
        for num_components in range(self.min_n_components, self.max_n_components+1):

            # For each model size loop over the different CV train/test folds.
            log_likelihoods = []
            for cv_train_idx, cv_test_idx in split_method.split(word_sequences):

                # Get current training set
                self.X, self.lengths = combine_sequences(cv_train_idx, word_sequences)

                # Train model
                model = self.base_model(num_components)

                # If training succeeded try to score the test set and record the log-likelihood.
                if model:
                    try:
                        X_test, test_lengths = combine_sequences(cv_test_idx, word_sequences)
                        logL = model.score(X_test, test_lengths)
                        log_likelihoods.append(logL)
                    except:
                        if self.verbose:
                            print("failure to score {} with {} states".format(self.this_word, num_components))

            # For current model size, average the log-likelihood scores across folds with successful
            # training and test.
            if len(log_likelihoods):
                results_dict[num_components] = np.mean(log_likelihoods)

        # Find model size with largest CV log-likelihood and return model after retraining on all word sequences.
        if len(results_dict):
            best_model_size = sorted(results_dict.items(), key=itemgetter(1), reverse=True)[0][0]
            self.X, self.lengths = self.hwords[self.this_word]
            return self.base_model(best_model_size)