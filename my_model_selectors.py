import concurrent.futures
import math
import time
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

    def base_model(self, num_states: int):
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
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    L is the likelihood of the fitted model (model.score())
    p is the number of free parameters
    N is the size of the dataset

    BIC applies a larger penalty when N > e^2 = 7.4
    """

    def bic_score(self, states: int):
        model = self.base_model(states)
        p = model.startprob_.size + model.transmat_.size + model.means_.size + model.covars_.diagonal().size
        logL = model.score(self.X, self.lengths)
        logN = np.log(self.X.shape[0])
        score = -2 * logL + p * logN
        return model, score

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        min = self.min_n_components
        max = self.max_n_components
        lowest_log = math.inf
        best_model = None

        for n in range(min, max + 1):
            try:
                model, score = self.bic_score(n)
                if score < lowest_log:
                    lowest_log = score
                    best_model = model
            except:
                pass

        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))

    ^^^ here addition of second term is missing which means that it is 0(when all models are equal)
    ^^^ here alfa param is set to `1` as seen in `alfa/(M-1)`

    '''

    def dic_score(self, states):
        model = self.base_model(states)
        alpha = 1
        M = len(self.words)
        likelihood = model.score(self.X, self.lengths)
        antilikelihood = alpha/(M-1) * math.fsum([model.score(self.hwords[w][0], self.hwords[w][1]) for w in self.words if w != self.this_word])
        dic_score = likelihood - antilikelihood
        return model, dic_score

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        min = self.min_n_components
        max = self.max_n_components
        lowest_log = math.inf
        best_model = None

        for n in range(min, max + 1):
            try:
                model, score = self.dic_score(n)
                if score < lowest_log:
                    lowest_log = score
                    best_model = model
            except:
                pass

        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds
    # args: sequences, Xlengths, word, min_n_components=2, max_n_components=15, random_state = 14
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        start = time.time()  # let's see how long this takes

        min = self.min_n_components
        max = self.max_n_components
        lowest_log = math.inf
        best_n_components = None

        with concurrent.futures.ThreadPoolExecutor() as executor:
            for avgLog, n in executor.map(lambda params: avg_log(*params),
                                          [[self.sequences, i] for i in range(min, max + 1)]):
                if avgLog < lowest_log:
                    lowest_log = avgLog
                    best_n_components = n

        finish = time.time()

        print(f'time taken: {finish-start}')

        return self.base_model(best_n_components)


def avg_log(sequences, num_states: int):
    scores = list()
    n_splits = min(3, len(sequences))  # make sure that splits will have smaller test data
    kf = KFold(n_splits=n_splits)
    for train_index, test_index in kf.split(sequences):
        try:
            train_x, train_length = combine_sequences(train_index, sequences)
            test_x, test_length = combine_sequences(test_index, sequences)
            train_fitted_model = GaussianHMM(n_components=num_states, n_iter=1000).fit(train_x, train_length)
            train_score = train_fitted_model.score(test_x, test_length)
            scores.append(train_score)
        except:
            pass
    return np.mean(scores), num_states
