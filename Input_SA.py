import pickle
from sklearn.naive_bayes import BernoulliNB
#analyzing the user input
with open("model.pickle",'rb') as fid:
	model = pickle.load(fid)
	cv = pickle.load(fid)
user_input = raw_input().lower()
input_mat = cv.fit_transform([user_input])
prediction = model.predict(input_mat)
if prediction[0] == 0: 
	print "Negative"
else:
	print "Positive"
