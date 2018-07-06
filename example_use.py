# download Google News word2vec binary file from https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing

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