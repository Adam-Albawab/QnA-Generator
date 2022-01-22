# Adam Albawab | CSE4392 | May 7 2021
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
import operator
import sys
import re
import os
import inflect
import numpy as np
import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.cluster.util import cosine_distance
import spacy
from pattern.en import conjugate, lemma, lexeme, pluralize, singularize
import stanza
import os
corenlp_dir = './corenlp'
stanza.install_corenlp(dir=corenlp_dir)
os.environ["CORENLP_HOME"] = corenlp_dir
from stanza.server import CoreNLPClient
client = CoreNLPClient(
    annotators=['tokenize','ssplit', 'pos', 'lemma', 'ner'], 
    memory='4G', 
    endpoint='http://localhost:9001',
    be_quiet=True)
nltk.download('punkt')
lemmatizer = WordNetLemmatizer()
inflect = inflect.engine()

fname=str(sys.argv[1])
qnum=sys.argv[2]
qnum = int(qnum)
print(fname)
file= open(fname, "r",  encoding="utf8")
if fname:
  sampleText = file.read()
else:
  print("Error reading file! Please try again.")
  exit()

spaC = spacy.load("en_core_web_sm")

#Preprocess data to remove titles and \n
sentenceEnds = [".","?","!"]
lines = sampleText.splitlines(keepends=True)
for i,line in enumerate(lines):
  if "\n" in line and not any(p in line for p in sentenceEnds):
    lines[i] = line.replace(line,"")
lines = list(filter(lambda a: a != '', lines))
lines = [line.replace('\n', '') for line in lines]
lines=[line.replace("however ","") for line in lines]
lines=[line.replace("also ","") for line in lines]
lines=[line.replace("for instance ","") for line in lines]
lines=[re.sub("[\(\[].*?[\)\]]", "", line) for line in lines]
lines=[re.sub("[\(\[].*?[\)\]]", "", line) for line in lines]


sampleTextP = ' '.join(lines)
sentencesP = sent_tokenize(sampleTextP)


#Select top sentences according to relevance to article
class TopSentenceExtraction():
    def __init__(self):
        self.damping = 0.85
        self.conv = 1e-5
        self.iter = 100
        self.text = None
        self.rank_vec = None
    def subtree_matcher(self, doc):
      subjpass = 0

      for i,tok in enumerate(doc):
     # find dependency tag that contains the text "subjpass"    
        if tok.dep_.find("subjpass") == True:
          subjpass = 1

      x = ''
      y = ''
      z = ''
      # if subjpass == 1 then sentence is passive
      if subjpass == 1:
        for i,tok in enumerate(doc):
          if tok.dep_.find("subjpass") == True:
            y = tok.text
          if tok.dep_.endswith("obj") == True:
            x = tok.text
          if tok.dep_.endswith("ROOT") == True:
            z = tok.text
    # if subjpass == 0 then sentence is not passive
      else:
        for i,tok in enumerate(doc):
          if tok.dep_.endswith("subj") == True:
            x = tok.text
          if tok.dep_.endswith("obj") == True:
            y = tok.text
          if tok.dep_.endswith("ROOT") == True:
            z = tok.text
      return z
    def _sentenceSim(self, sentence1, sentence2, stopwords=None):
        if stopwords is None:
            stopwords = []

        sentence1 = [word.lower() for word in sentence1]
        sentence2 = [word.lower() for word in sentence2]

        all_words = list(set(sentence1 + sentence2))

        vector1 = [0] * len(all_words)
        vector2 = [0] * len(all_words)

        for word in sentence1:
            if word in stopwords:
                continue
            vector1[all_words.index(word)] += 1

        for word in sentence2:
            if word in stopwords:
                continue
            vector2[all_words.index(word)] += 1
        return (1 - cosine_distance(vector1, vector2))

    def _create_simMatrix(self, sentences, stopwords=None):
        simMatrix = np.zeros([len(sentences), len(sentences)])

        for idx1 in range(len(sentences)):
            for idx2 in range(len(sentences)):
                if idx1 == idx2:
                    continue

                simMatrix[idx1][idx2] = self._sentenceSim(sentences[idx1], sentences[idx2], stopwords=stopwords)

        simMatrix =  simMatrix + simMatrix.T - np.diag(simMatrix.diagonal())
        norm = np.sum(simMatrix, axis=0)
        simMatrix_norm = np.divide(simMatrix, norm, where=norm != 0)
        return simMatrix_norm

    def _rank_algorithm(self, simMatrix):

        rank_vec = np.array([1] * len(simMatrix))
        previous_vec = 0
        for x in range(self.iter):
            rank_vec = (1 - self.damping) + self.damping * np.matmul(simMatrix, rank_vec)
            if abs(previous_vec - sum(rank_vec)) < self.conv:
                break
            else:
                previous_vec = sum(rank_vec)
      
        return rank_vec

    def get_triples(self, reducedsent):
        relationSent=None
        output = client.annotate(reducedsent, properties={"annotators":"tokenize,ssplit,pos,depparse,natlog,openie",
                                "outputFormat": "json","openie.triple.strict":"true", "openie.max_entailments_per_clause" : "900"})
        result = [output["sentences"][0]["openie"] for item in output]
        doc1 = spaC(reducedsent)
        root = self.subtree_matcher(doc1)
        for i in result:
          for rel in i:
            largestSpan = 0
            totSpan=((rel['subjectSpan'][1])-(rel['subjectSpan'][0]))+((rel['relationSpan'][1])-(rel['relationSpan'][0]))+((rel['objectSpan'][1])-(rel['objectSpan'][0]))/3
            if totSpan > largestSpan:
              largestSpan = totSpan
              relationSent=rel['subject'],rel['relation'],rel['object'],root
          return relationSent
    def get_most_relevant_sentences(self, num):
        tempArray=[]
        triples=[]
        relevant_sentences = []
        relevant_triples = []
        sorted_list_reduced=[]
        if self.rank_vec is not None:
            sorted_list = list(np.argsort(self.rank_vec))
            sorted_list.reverse()
            for j,sent in enumerate(range(len(sorted_list))):
              if len(self.text[sorted_list[j]])<75:
                sorted_list_reduced.append(sorted_list[j])
            for k, p in enumerate(sorted_list_reduced):
              triple=self.get_triples(self.text[sorted_list_reduced[k]])
              if triple is not None:
                tempArray.append(self.text[sorted_list_reduced[k]])
                triples.append(triple)
              else:
                continue
            for i,sentence in enumerate(range(len(tempArray))):
                sentence = tempArray[i]
                trip=triples[i]
                relevant_sentences.append(sentence)
                relevant_triples.append(trip)
        return relevant_sentences, relevant_triples

    def execute(self, text, stop_words=None):
        self.text = text
        tokenized_sentences = [word_tokenize(sent) for sent in self.text]
        simMatrix = self._create_simMatrix(tokenized_sentences, stop_words)
        self.rank_vec = self._rank_algorithm(simMatrix)
client.stop()
client.start()
corpus = TopSentenceExtraction()
corpus.execute(sentencesP)
sentList, extractions = corpus.get_most_relevant_sentences(qnum)
client.stop()


def format_sentence(sentence):
  sent = sentence.capitalize()
  doc = spaC(sentence)
  for ent in doc.ents:
    strip_=ent.label_.strip(",")
    if strip_ == 'ORG' or strip_=="GPE" or strip_=="PERSON" or strip_=="LOC" or strip_=="LANGUAGE" or strip_=="EVENT" or strip_=="FAC":
      held=ent.text
      sent = sent.replace(held.lower(),held,1)
  sentResult= " ".join(sent.split())
  sentResult = sentResult.rstrip('.')
  sentResult = sentResult+str("?")
  return sentResult

def be_check(reldoc):
  rel= []
  rel = reldoc.split()
  for x in rel:
    relLem=lemmatizer.lemmatize(x, 'v')
    relLem=relLem.lower()
    if relLem == "be":
      return x
    else:
      return False

def tense_tagger_(root):
  rootDoc=spaC(root)
  for x in rootDoc:
    if x.tag_ == "VBD" or x.tag_=="VBN":
      return "past"
    else:
      return "present"

def entity_extractor(snt):
  entities=[]
  for m, entt in enumerate(snt.ents):
      entities.append((entt.label_,entt.text))
  return entities

def tagger_(doc):
  for x in subjDoc:
    if x.tag_ =="NNS" or x.tag_=="NNPS":
      return "plural"
    elif x.tag_ =="NN" or x.tag_ =="NNP":
      return "singular"

def aux_verb_finder(doc):
  aux=""
  for i,tok in enumerate(doc):
    if tok.dep_.endswith("aux") and tok.pos_.endswith("VERB"):
      aux = tok.text
  return aux


def question_type_selection(docc, subjDocc,objDocc, text, obj, relation, subj):
  tag=tagger_(objDocc)
  entityCheck=entity_extractor(docc)
  tagChecker=tagger_(subjDocc)
  tns = tense_tagger_(root)
  isPerson = False
  isLoc = False
  isTime = False
  g = False
  fixed = text
  rootD=lemmatizer.lemmatize(root, 'v')
  for n, p in enumerate(entityCheck):
    if entityCheck[n][0]=="PERSON" or entityCheck[n][0]=="GPE":
      isPerson=True
    elif entityCheck[n][0] =="LOC" or entityCheck[n][0] == "GEO" or entityCheck[n][0] == "FAC":
      isLoc=True
    elif entityCheck[n][0]=="TIME" or entityCheck[n][0] =="DATE":
      isTime=True
    elif entityCheck[n][0] =="MONEY" or entityCheck[n][0] =="CARDINAL" or entityCheck[n][0]=="QUANTITY" or entityCheck[n][0]=="PERCENT":
      isQuantity=True
    if isPerson:
      if not fixed.split()[0]  == root:
        rest = fixed.split(" "+root,1)[1]
      else:
        rest = relation +" " +obj
        g=True
      if rootD == "be" or g==True:
        fixed = "Who " + root +" "+rest
      else:
        if tns=="present":
          fixed = "Who is " + root +" "+rest
        elif tns=="past":
          fixed = "Who was " + root +" "+rest
          if tns=="past" and tagChecker=="plural":
            fixed = "Who were " + root +" "+rest
      if tagChecker=="plural" and not tns=="past":
        for r in rootDoc:
          rootj = inflect.plural(root)
          fixed = fixed.replace(root,rootj,1)

    elif isLoc:
      if not fixed.split()[0] == root:
        rest = fixed.split(" "+root,1)[1]
      else:
        rest = subj+" "+obj
        g=True
      if rootD == "be" or g==True:
        fixed = "Where " + root +" "+rest
      else:
        if tns=="present":
          fixed = "Where is " + root +" "+rest
        if tns=="past":
          fixed = "Where was " + root +" "+rest
      if tagChecker=="plural" and not tns =="past":
        for r in rootDoc:
          rootj = inflect.plural(root)
          fixed = fixed.replace(root,rootj,1)        
    elif isTime or "time" in text:
      if fixed.split()[0] == root:
        rest = fixed.split(" "+root,1)[1]
      else:
        rest = subj+" "+obj
        g=True
      if rootD == "be" or g==True:
        fixed = "When "+root+" "+rest
      else:
        if tns=="present":
          fixed = "When is " + root +" "+rest
        if tns=="past":
          fixed = "When was " + root +" "+rest

    elif not be_check(relation) and not be_check(root) and tagChecker and not isPerson and not "been" in text.lower():  
      if not fixed.split()[0]== root:
        rest = fixed.split(" "+root,1)[1]
      else:
        rest = relation +" " +obj
      fixed = "What " + " "+root +" "+ rest
      if tagChecker=="plural" and not tns =="past":
          rootj = inflect.plural(root)
          fixed = fixed.replace(root,rootj,1)
      elif tagChecker=="singular" and "and" in text:
          rootj = inflect.plural(root)
          fixed = fixed.replace(root,rootj,1) 
      if root.lower()=="have":
        fixed = fixed.replace(root, "has")    
    else:
      return None
    return fixed



for i, sent in enumerate(sentList):
  fixed = sentList[i]
  extracted=extractions[i]
  root=extracted[len(extracted)-1]
  relation=extracted[len(extracted)-3]
  subj=extracted[0]
  obj=extracted[2]
  objDoc=spaC(obj)
  subjDoc=spaC(subj)
  relDoc=spaC(relation)
  rootDoc=spaC(root)
  rootD=lemmatizer.lemmatize(root, 'v')
  doc = spaC(sentList[i])
  ax=aux_verb_finder(doc)
  ent= entity_extractor(doc)
  if question_type_selection(doc,subjDoc,objDoc,fixed, obj,relation,subj) ==None:

    if rootD=="be" and not root=="be" and not root.lower()=="been":
      fixed = fixed.replace(" "+root+" "," ",1)
      fixed = str(root) + " " + fixed

    elif ax:
      fixed = fixed.replace(" "+ax+" "," ",1)
      fixed = str(ax) + " " + fixed
      #if not root.lower() == "been":
        #fixed = fixed.replace(root,rootD,1)

    elif be_check(relation) and not root=="be":
      root = be_check(relation)
      fixed = fixed.replace(" "+root+" "," ",1)
      fixed = str(root) + " " + fixed

    elif not ax and not rootD=="be" and not be_check(relation):
      for x in rootDoc:
        if x.tag_== "VBD" or x.tag_ == "VBN":
          fixed = fixed.replace(" "+root+" "," "+rootD+" ",1)
          fixed = "Did "+fixed
        elif x.tag_ == "VBZ" or x.tag_ == "VBG":
          fixed = fixed.replace(" "+root+" "," "+rootD+" ",1)
          fixed = "Does " + fixed
        elif x.tag_ == "VB" or x.tag_ == "VBP":
          fixed = fixed.replace(" "+root+" "," "+rootD+" ",1)
          fixed = "Do " + fixed
  else:
    fixed = question_type_selection(doc,subjDoc,objDoc,fixed, obj,relation,subj)
  fixed = format_sentence(fixed)
  print(str(i+1)+" "+fixed)