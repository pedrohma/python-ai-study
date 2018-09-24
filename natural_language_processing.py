import nltk

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer

# DT means the determinant, VBP means the verb, JJ means the adjective, IN means the preposition and NN means the noun.
sentence=[("a","DT"),("clever","JJ"),("fox","NN"),("was","VBP"),
          ("jumping","VBP"),("over","IN"),("the","DT"),("wall","NN")]

# give the grammar in the form of regular expression
grammar = "NP:{<DT>?<JJ>*<NN>}"

# parsing the grammar
parser_chunking = nltk.RegexpParser(grammar)

# parsing the sentence
parser_chunking.parse(sentence)

# getting the output
Output_chunk = parser_chunking.parse(sentence)

Output_chunk.draw()
