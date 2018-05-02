""" useful tools for processing text """
import re 

def get_words(s):
    """ Extract a list of words from a sentence string with punctuation, spaces etc 
    s = sentence 
    """
    # strip punctuation 
    s = re.sub(r'[^\w\s]','',s)
    # replace newline 
    s = s.replace('\n', ' ')
    # get rid of spaces
    s = " ".join(s.split())
    return s.split(' ')

# test get_words
# print(get_words("heloo   and then i came down, with the crown \n and Up. again"))