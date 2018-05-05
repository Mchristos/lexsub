""" useful tools for processing text """
import re 

stopwords = ["it's", "she's", 'were', 'because', 'this', 'couldn', 'then', 'how'
, 'd', 'doesn', 'down', 's', 'they', 'she', "needn't", 'wasn', 'haven', 
'between', "wouldn't", 'the', 'ma', "wasn't", 'until', 'my', 'himself', 
"that'll", 'by', 'about', 'in', "aren't", "should've", 'why', 'nor', 
'before', 'when', 'we', 'here', 'only', "couldn't", 'ain', 'no', 'your', 
'will', 'own', 'his', "you'll", 'are', 'and', 'most', 'do', 'now', "isn't", 
'having', 'on', 'her', 'theirs', 'under', 'with', 'to', "mightn't", 'while', 
'its', 'be', 'll', 'don', 'over', 'again', 'their', 'won', 'too', 'during', 
'shan', 'herself', 'has', 'or', 'from', 'ours', 'into', 'our', 'above', 
'wouldn', 'you', 'of', 'so', 't', 'he', 'doing', 'as', 'i', 'can', 'shouldn', 
'have', 'at', 'other', 'hasn', 'more', 'yourselves', 'y', 'yours', 'very', 
'themselves', 'which', 'these', 'being', 'both', 'aren', 'did', 'than', 'needn',
 'for', 'itself', "haven't", 'through', 'weren', 'but', 'once', 'isn', 
 'ourselves', 'didn', 'not', 'yourself', 'mightn', 'after', 've', 'him', 
 'whom', "hasn't", 'a', 'hadn', "shouldn't", "mustn't", 'those', 'off', 
 'each', 'was', "didn't", "you'd", 'where', 'o', 'further', 'below', "shan't", 
 'myself', 'mustn', 'is', 'been', 'just', 'any', 'out', 'that', 'm', 'such', 
 'me', 'same', 'hers', 'some', 'had', 'does', 'against', 'should', "you've", 
 "doesn't", "you're", 'them', 'am', 'if', 'who', 'few', 'what', 'there', 
 "don't", "weren't", "won't", 'an', 'all', 're', 'it', 'up', "hadn't"]

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

def unique(iter):
    "removes duplicates from iterable preserving order"
    result = list()
    seen = set()
    for x in iter:
        if x not in seen:
            seen.add(x)
            result.append(x)
    return result

def process_candidates(candidates, target):
    """ words to lower case, replace underscores, remove duplicated words, 
        filter out target word and stop words """
    filterwords = stopwords + [target]
    return unique(filter(lambda x : x not in filterwords, 
                  map(lambda s : s.lower().replace('_', ' '), candidates)))



# test functions
# print(process_candidates("i am AM a lazy dog dog dog and DOG I am NOT lazy MUCH any_more but candy_coated".split(' '), "lazy"))
# print(get_words("heloo   and then i came down, with the crown \n and Up. again"))