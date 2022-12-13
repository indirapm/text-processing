'''
Text Processing Code Development
code by: Indira Puspita Margariza (indiramrgrz@gmail.com)

This code is being used to complete AI Engineer Technical Test at Zettabyte.
The goal of this code is to process given texts with clean words output which 
can be used for any text analytics model or othe related purposes. 
'''

#Importing necessary libraries. This code uses NLTK libarary for text processing
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')
from nltk.tokenize import word_tokenize, regexp_tokenize 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
import re
import pandas as pd

#Load text file. Please modify the path if necessary.
text = open(f'text.txt', 'r')
text = text.read()

'''
Tokenize the input text resulting list of words or character
word_tokenize function is used to tokenize the text but there
are still several words seperated with other characters which 
need to be tokenize.
'''
tokens = []
#
for token in word_tokenize(text):
  #Remove two words separated by "-" or "—"
  #Please add other character if necessary.
  if "-" in token:
    words = token.split("-")
    tokens.extend(words)
  elif "—" in token:
    words = token.split("—")
    tokens.extend(words)
  else:
    tokens.append(token)

def filter_token(tokens):
  '''
  Filter words with no vowel or consist non alphabetical character

  Parameter:
    tokens: List
      List of words or tokens
  Return:
    new_tokens: List
      List of filtered words
  '''
  new_tokens = []
  for token in tokens:
    token = re.sub(r'[^a-zA-Z]+', '', token.lower())
    count_vowel = len([vowel for vowel in token if vowel in 'aiueo'])
    if count_vowel != 0:
      new_tokens.append(token) 
  return new_tokens

#filter tokens
clean_tokens = filter_token(tokens)

#load and add stopwords
stop_words = stopwords.words("english")
stop_words.extend(['us','could','though','would','also','many','much', 'might', 
                  'must', 'whose', 'every', 'without','another', 'among'])

#Removing the stopwords
normalized_tokens = [token for token in clean_tokens if token not in stop_words]

#Categorized each word
tokens_tagset = nltk.pos_tag(normalized_tokens)
df_tokens = pd.DataFrame(tokens_tagset, columns=['Word', 'Tag'])

# Create Lemmatizer 
lemmatizer = WordNetLemmatizer()

#Lemmatize each word 
lemmatized_tokens = []
for word in normalized_tokens:
    output= [word, lemmatizer.lemmatize(word, pos='n'), 
            lemmatizer.lemmatize(word, pos='a'),
            lemmatizer.lemmatize(word, pos='v')]
    lemmatized_tokens.append(output)

#Create DataFrame using original words and their lemmatized words
df = pd.DataFrame(lemmatized_tokens, columns =['Word', 'Lemmatized Noun', 'Lemmatized Adjective', 'Lemmatized Verb'])
df['Tag'] = df_tokens['Tag']

'''
Simplifying words' tag into noun, adjective, and verb to choose which lemmatized
word to be used.
Lemmatizing words with their tags can increase the accuracy.
'''  
df = df.replace(['NN','NNP','NNS','NNPS'],'n')
df = df.replace(['JJ','JJR','JJS'],'a')
df = df.replace(['VB','VBD','VBG','VBN','VBP','VBZ'],'v')
words = []
for idx in range (0, len(lemmatized_tokens)):
  if df.loc[idx]['Tag']=='n':
    word = df['Lemmatized Noun'][idx]
  elif df.loc[idx]['Tag']=='a':
    word = df['Lemmatized Adjective'][idx]
  elif df.loc[idx]['Tag']=='v':
    word = df['Lemmatized Verb'][idx]
  else:
    word = df['Word'][idx]  
  words.append(word)

#This list of words `words` is the final result of this text processing.
df['Final Word'] = words