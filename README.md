# lexsub : context-sensitive word substitutions using Word2Vec 

Disambiguating between the possible senses of a word in the context of a sentence is a fundamental problem in NLP. However, this assumes a universal set of "meanings" to disambiguate between. A more natural but also more practical task is finding a good substitution for a word in context. For example, in the sentence "She went to the bar last night", we know bar means pub, but the word bar has other meanings: a chocolate bar, or a ban/restriction on something. 

This repository uses a Word2Vec embedding based on the google news corpus, made available [here](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing) and through the [gensim library](https://radimrehurek.com/gensim/) to rank candidate word substitutions by their suitability to the context of the sentence.  

## Example Usage 

    from lexsub import LexSub
    from gensim.models import KeyedVectors
    
    word2vec_path = "~/GoogleNews-vectors-negative300-SLIM.bin"
    vectors = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
    ls = LexSub(vectors, candidate_generator='lin')

    sentence = "She had a drink at the bar"
    target = "bar.n"
    result = ls.lex_sub(target, sentence)
    print(result)
    # ['bars', 'pub', 'tavern', 'nightclub', 'restaurant']
