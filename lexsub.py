""" Different models for performing lexical substitution task 
    (find substitution for a word in context)"""
from nltk.corpus import lin_thesaurus as lin
from nltk.corpus import wordnet as wn
import numpy as np 
from numpy.linalg import norm
from gensim.models import KeyedVectors

def cos(v1, v2):
    return v1.dot(v2)/(norm(v1)*norm(v2))

WORD2VEC_PATH = "~/GoogleNews-vectors-negative300-SLIM.bin"

class NaiveLin(object):
    """Gets lexical substitutes ignoring context using the lin thesaurus"""
    def __init__(self, n_synonymns):
        self.n_synonyms = n_synonymns
        self.poses = ['n'] # supported POS values 
    
    def lex_sub(self, word):
        """
        word = string in form word.POS 
        context = XML node of parsed sentence  
        """
        w, _ , POS = word.partition('.')
        if POS in self.poses:
            fileid = 'sim%s.lsp' % POS.upper()
            thes_entry = lin.scored_synonyms(w, fileid = fileid)
            thes_entry = sorted(thes_entry, key = (lambda x : x[1]), reverse = True)
            thes_entry = thes_entry[:self.n_synonyms]
            return [word for (word,score) in thes_entry]
        else:
            raise ValueError("unsupported POS")

class NaiveWordNet(object):
    """ Gets lexical substitutes ignoring context using WordNet """
    def __init__(self, n_synonymns):
        self.n_synonymns = n_synonymns
        self.poses = ['n'] # supported POS values 

    def lex_sub(self, word):
        """
        """
        w,_,POS = word.partition('.')
        if POS in self.poses:
            synset = wn.synsets(w, wn.NOUN) # NOUN hard-coded 
            # synset = synset[:self.n_synonymns]
            result =  [lemma.name() for s in synset for lemma in s.lemmas()]
            return result[:self.n_synonymns]
        else:
            raise ValueError("unsupported POS")

class Word2Vec(object):
    "Find word substitutions for a word in context using word2vec skip-gram embedding"
    def __init__(self, n_synonymns):
        self.n_synonymns = n_synonymns
        self.poses = ['n'] # supported POS values 
        self.model = KeyedVectors.load_word2vec_format(WORD2VEC_PATH, binary=True)
        self.candidate_generator = NaiveLin(50)

    def get_substitutability(self, t, s, C):
        """ get substitutability of substitution s for target t in context C
        
        t = target word 
        s = candidate substitution 
        C = list of context words 
        """
        tvec = self.model.get_vector(t)
        svec = self.model.get_vector(s)
        # 1. target score: how similar is it to the target word? 
        tscore = cos(tvec, svec)
        # 2. context score: how similar is it to the context words?
        cscores = [cos(svec, self.model.get_vector(c)) for c in C ]
        cscore = sum(cscores)
        return (tscore + cscore)/(len(C)+1)  


    def lex_sub(self, word, context_words):
        w,_,POS = word.partition('.')
        # filter context words that exist in the word2vec model
        context_words = [word for word in context_words if word in self.model.vocab]
        # generate candidate substitutions
        candidates = self.candidate_generator.lex_sub(word)
        if POS in self.poses:
            cand_scores = [self.get_substitutability(w, s, context_words) if s in self.model.vocab else 0 for s in candidates ]
            assert(len(cand_scores) == len(candidates))
            sorted_candidates = sorted(zip(candidates, cand_scores), key = lambda x : x[1], reverse=True )
            return [sub for sub, score in sorted_candidates][:self.n_synonymns]
        else:
            raise ValueError("unsupported POS")


