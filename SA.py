from sklearn.naive_bayes import BernoulliNB
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import time
import sys
import pickle

start = time.time()
training_texts = []
vocabulary = []
test_texts = []

#extract text from the training file and create a vocabulary
def extract(training_file):
	text = []
	tokens = []
	with open(training_file,'r') as fid:
		for l in fid.readlines():
			text.append(l)
			tokens.append(word_tokenize(l))
	return text,tokens

# extract text from positive reviews training file
text,tokens = extract("train-pos.txt")
for x in text:
	training_texts.append(x)
for y in tokens:	
	vocabulary.append(y)

#extract text from negative reviews training file
text,tokens = extract("train-neg.txt")
for x in text:
	training_texts.append(x)
for y in tokens:	
	vocabulary.append(y)

#define training labels
pos_label = np.ones(12500)
neg_label = np.zeros(12500)
training_labels = np.concatenate((pos_label,neg_label)) 

#train the model
vocabulary = list(set([item for sublist in vocabulary for item in sublist]))
print len(vocabulary)
cv = CountVectorizer(vocabulary=vocabulary) # converts a colleciton of text documents to a matrix of token counts
training_mat = cv.fit_transform(training_texts)## returns document term matrix 
# model = BernoulliNB()
# model.fit(training_mat,training_labels)

# #test the model
# test_labels = np.concatenate((pos_label,neg_label))
# text, _ = extract("test-pos.txt")

# for x in text:
# 	test_texts.append(x)

# text, _ = extract("test-neg.txt")
# for x in text:
# 	test_texts.append(x)

# test_mat = cv.fit_transform(test_texts)
# accuracy = model.score(test_mat,test_labels)
# print accuracy

# #save the model
# with open("model.pickle",'wb') as fid:
# 	pickle.dump(model,fid)
# 	pickle.dump(cv,fid)
# end = time.time()
# print "Total time: %f" %(end-start)


