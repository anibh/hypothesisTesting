import pyspark
import json
import re
import numpy as np
from datetime import datetime
import sys
from scipy import stats

start_time = datetime.now()
sc = pyspark.SparkContext('local[*]')
sc.setLogLevel('WARN')

rdd = sc.textFile(sys.argv[1])

def mapReviews(x):
	try:
		overall = x['overall']
	except KeyError:
		return ( 0.0, False, [])
		
	try:
		verified = x['verified']
	except KeyError:
		return ( 0.0, False, [])
		
	try:
		reviewText = x['reviewText']
	except KeyError:
		return ( 0.0, False, [])
		
	return (overall, verified, [word for word in re.findall(r'((?:[\.,!?;"])|(?:(?:\#|\@)?[A-Za-z0-9_\-]+(?:\'[a-z]{1,3})?))', reviewText)])
		

def mapWords(x):
	wordFreq = []
	for word in x[0][2]:
		wordFreq.append(((x[1], x[0][0], x[0][1], word.lower(), len(x[0][2])), 1))
	return wordFreq

def count(x):
	wordDict = commonWords
	for (word, value) in x[1]:
		if(wordDict.get(word) != None):
			wordDict[word] = value
		
	wordList = list(wordDict.items())
	return (x[0], wordList)
	
def wordMap(x):
	list = []
	for word, value in x[1]:
		if x[0][3] != 0:
			list.append((word, (value, x[0][1], x[0][2])))
	return list

def corrSingle(data):
	x = []
	y = []
	for (value, rating, verified) in data[1]:
		x.append(value)
		y.append(rating)
	N = len(x)
	x = (x - np.mean(x))/np.std(x)
	y = (y - np.mean(y))/np.std(y)
	x = x.reshape(-1, 1)
	coeff = np.linalg.inv(x.transpose().dot(x)).dot(x.transpose()).dot(y)
	df = N - 2
	yhat = coeff[0]*x
	t = coeff[0] / np.sqrt((np.sum((y - yhat)**2)/df)/np.sum((x - np.mean(x))**2))
	p_lt = stats.t.cdf(t, df)
	if p_lt > 0.5: p = (1-p_lt)*2
	else: p = p_lt*2
	return (data[0], coeff[0], p)
	
def corrMultiple(data):
	x = []
	y = []
	control = []
	for (value, rating, verified) in data[1]:
		if(verified): x.append([value, 1])
		else: x.append([value, 0])
		y.append(rating)
	N = len(x)
	x = (x - np.mean(x, axis=0))/np.std(x, axis=0)
	y = (y - np.mean(y))/np.std(y)
	coeff = np.linalg.inv(x.transpose().dot(x)).dot(x.transpose()).dot(y)
	df = N - 3
	yhat = np.matmul(coeff, x.T)
	t = coeff[0] / np.sqrt((np.sum((y - yhat)**2)/df)/np.sum((x[1] - np.mean(x[1]))**2))
	p_lt = stats.t.cdf(t, df)
	if p_lt > 0.5: p = (1-p_lt)*2
	else: p = p_lt*2
	return (data[0], coeff[0], p)

rdd = rdd.map(json.loads)\
			.map(mapReviews)\
			.zipWithIndex()\
			.flatMap(mapWords)\
			.reduceByKey(lambda a, b: a+b)
			
rdd.persist()

rdd1 = rdd.map(lambda x: (x[0][3], x[1]))\
			.reduceByKey(lambda a, b: a+b)\
			.map(lambda x: (x[1], x[0]))\
			.sortByKey(False).take(1000)
			
commonWords = dict()
wordSet = set()
for (f, w) in rdd1:
	commonWords[w] = 0
	wordSet.add(w)

rdd2 = rdd.map(lambda x: ((x[0][0], x[0][1], x[0][2], x[0][4]), (x[0][3], x[1])))\
			.groupByKey()\
			.map(count)\
			.flatMap(wordMap)\
			.groupByKey()

rdd.unpersist()
rdd2.persist()
rdd3 = rdd2.map(corrSingle)

rdd3.persist()
			
print(rdd3.map(lambda x: (x[1], (x[0], x[1], x[2]))).sortByKey(False).take(20))

print(rdd3.map(lambda x: (x[1], (x[0], x[1], x[2]))).sortByKey().take(20))

rdd3.unpersist()

rdd4 = rdd2.map(corrMultiple)
				
rdd2.unpersist()				
rdd4.persist()

print(rdd4.map(lambda x: (x[1], (x[0], x[1], x[2]))).sortByKey(False).take(20))

print(rdd4.map(lambda x: (x[1], (x[0], x[1], x[2]))).sortByKey().take(20))

rdd4.unpersist()
end_time = datetime.now()
print("Execution Time : ", end_time - start_time)