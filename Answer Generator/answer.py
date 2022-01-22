#Adam Albawab | ID#1001572887 | NetID:axa2887 | 04/30/21 | Assignment 3 - NLP 4392
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk
from nltk import tokenize
from nltk import sent_tokenize, word_tokenize
from scipy.sparse.csr import csr_matrix
from nltk.cluster.util import cosine_distance
import pandas as pd
import re
import spacy
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
lemmatizer = WordNetLemmatizer()
nlp = spacy.load("en_core_web_sm")
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('punkt')


fname=str(sys.argv[1])
ques=sys.argv[2]

print(fname)
file= open(fname, "r",  encoding="utf8")
file2= open(ques, "r",  encoding="utf8")
if fname:
  sampleText = file.read()
else:
  print("Error reading file! Please try again.")
  exit()

if q:
  questions = file2.read()
else:
  print("Error reading questions! Please try again.")
  exit()  
# Method that reads article line by line and adds it to a string.
def read_article(article):
    with open(article, "r") as f:
        return " ".join([clean_sent for clean_sent in [raw_sent.strip("\t ") for raw_sent in f.readlines()] if len(clean_sent) > 0])

# Method to tokenize the string into sentences (documents), returns the data set
def create_data_set(corpus_text):
    data_set = tokenize.sent_tokenize(corpus_text)
    for i, sent in enumerate(data_set):
       if len(data_set[i])>100:
         data_set.pop(i)
    return data_set

def get_root(sentence):
  z = ''
  doc= nlp(sentence)
  for i,tok in enumerate(doc):
    if tok.dep_.endswith("ROOT") == True:
      z = tok.text
  return z

# Method that creates the vectorizer and builds the document tfidf
def build_tfidf(dataset):
    vectorizer = TfidfVectorizer(input=dataset, analyzer='word', ngram_range=(1,1),
                                 min_df=0, stop_words=None)
    docs_tfidf = vectorizer.fit_transform(dataset)

    return vectorizer, docs_tfidf

#helper function for lemma conversion since the type of the word is required for lemmatization
def nltk2wn_tag(nltk_tag):
  if nltk_tag.startswith('J'):
    return wordnet.ADJ
  elif nltk_tag.startswith('V'):
    return wordnet.VERB
  elif nltk_tag.startswith('N'):
    return wordnet.NOUN
  elif nltk_tag.startswith('R'):
    return wordnet.ADV
  else:                    
    return None

#function to convert any sentence into a lemma
def lemmatize_sentence(sentence):
  nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))    
  wn_tagged = map(lambda x: (x[0], nltk2wn_tag(x[1])), nltk_tagged)
  res_words = []
  for word, tag in wn_tagged:
    if tag is None:                        
      res_words.append(word)
    else:
      res_words.append(lemmatizer.lemmatize(word, tag))
  return " ".join(res_words)


# Method to calculate the cosine similarity scores between the query and the documents.
def get_cosine_similarities(vectorizer, docs_tfidf, question):
    # Vectorizes the question
    query_tfidf = vectorizer.transform([question.lower()])

    # Computes the cosine similarities between the query and the docs.
    cosine_similarities = (query_tfidf * docs_tfidf.T).toarray()

    return cosine_similarities


# Method to test for if the matrix is zeroed or not
def is_zero(cosine_similarities):
    if not np.any(cosine_similarities):
        return True
    return False


# If the lemma of the root of the question is not in the sentence then discard that sentence and keep looking up to 20 times to find the most similar sentence
# that also contains root of the question. If no sentence containing the root is found then instead return "No match"
def get_return_value(data_set, cosine_similarities, question):
  check = True
  sentence='No match'
  percentage=0
  max=1
  top = 0;
  while check and data_set:
    if max:
      max=[np.argmax(cosine_similarities)]
      cosine_similarities = np.delete(cosine_similarities, max)
      sent=data_set[max[0]]
      root=get_root(question)
      rootLemma=lemmatize_sentence(root)
      sentenceLemma=lemmatize_sentence(sent)
      if rootLemma in sentenceLemma:
        sentence = sent
        percentage = np.max(cosine_similarities)
        check = False
      else:
        data_set.pop(max[0])
        top = top+1
        if top>20:
          check =False
  return sentence, percentage

#Removes titles since they will not be used in answering any questions
def remove_titles(text):
  sentenceEnds = [".","?","!"]
  lines = text.splitlines(keepends=True)
  article=""
  for i,line in enumerate(lines):
    if "\n" in line and not any(p in line for p in sentenceEnds):
      lines[i] = line.replace(line,"")
  while "" in lines:
    lines.remove("")
  for x in lines:
    article ="".join((article, x))
  return article

#function to extract the named entities from a sentecnce
def entity_extractor(snt):
  entities=[]
  for m, entt in enumerate(snt.ents):
      entities.append((entt.label_,entt.text))
  return entities

#Function used to find which part of the sentence or if the whole sentence should be returned
def get_answer(sentence, question, percent):
  doc= nlp(sentence)
  entityCheck=entity_extractor(doc)
  entityText = "none"
  if sentence == 'No match':
    return 'No match'
  if "what" in question:
    return sentence
  for i,p in enumerate(entityCheck):
    if entityCheck[i][0]=="PERSON" or entityCheck[i][0] =="NORP" or entityCheck[i][0] =="ORG":
      entityText=entityCheck[i][1]
      if "who" in question:
          return entityText
    if entityCheck[i][0]=="PERSON" or entityCheck[i][0] =="NORP":
      entityText=entityCheck[i][1]
      if "whom" in question:
          return entityText
    if entityCheck[i][0]=="PERSON" or entityCheck[i][0] =="NORP":
      entityText=entityCheck[i][1]
      if "whose" in question:
          return entityText
    elif entityCheck[i][0] =="LOC" or entityCheck[i][0] =="GEO" or entityCheck[i][0] =="GPE" or entityCheck[i][0] =="FAC" :
      entityText=entityCheck[i][1]
      if "where" in question:
          return entityText
    elif entityCheck[i][0]=="TIME":
      entityText=entityCheck[i][1]
      if "when" in question: 
        return (str("At" + entityText))
    elif entityCheck[i][0] =="DATE":
      entityText=entityCheck[i][1]
      if "when" in question: 
        return entityText
    elif entityCheck[i][0] =="MONEY" or entityCheck[i][0] =="CARDINAL" or entityCheck[i][0]=="QUANTITY" or entityCheck[i][0]=="PERCENT":
      entityText=entityCheck[i][1]  
      if "how much" or "how many" in question:
        return entityText
  if percent > 0.1:      
    return sentence
# Main called method to find the most similar document to the query.
def query(article, question):
    article = read_article(article)
    article=remove_titles(article)
    question = question.lower()
    # Check for if article has data in it. If article not found, skip to the end and return.
    if article:
        data_set = create_data_set(article)
        vectorizer, docs_tfidf = build_tfidf(data_set)
        cosine_similarities = get_cosine_similarities(vectorizer, docs_tfidf, question)
        sentence, percentage = get_return_value(data_set, cosine_similarities, question)
        answer = get_answer(sentence,question.lower(),percentage)
        return sentence, answer
    return "No match"

questions = questions.splitlines()
for question in questions:
  print("\nQuestion: "+question)
  result = query(article, question)
  print("Answer: "+result[1])