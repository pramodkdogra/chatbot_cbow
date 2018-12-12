# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 10:49:31 2018

@author: prdogra
"""
import nltk
from nltk.stem.lancaster import LancasterStemmer

# things we need for Tensorflow
import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import pickle

stemmer = LancasterStemmer()

file="intents.json"
pathfile="C:/Users/prdogra/Sync/MSC_Cognitive/Year2018_2019/COS524_Natural Language Processing/Project_code/CBOW/data/"+file

pckle_path="C:/Users/prdogra/Sync/MSC_Cognitive/Year2018_2019/COS524_Natural Language Processing/Project_code/CBOW/pickle/"

with open(pathfile) as json_data:
    intents = json.load(json_data)



#With intents JSON file loaded, Let's organize our documents, words and classification classes.
words = []
classes = []
documents = []
ignore_words = ['?']
# loop through each sentence in our intents patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = nltk.word_tokenize(pattern)
        # add to our words list
        words.extend(w)
        # add to documents in our corpus
        documents.append((w, intent['tag']))
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

print(words)

# remove duplicates
classes = sorted(list(set(classes)))

print (len(documents), "documents")
print (len(classes), "classes", classes)
print (len(words), "unique stemmed words", words)

#Unfortunately this data structure won’t work with Tensorflow, we need to transform it further: from documents of words into 
#tensors of numbers.

# create our training data
training = []
output = []
# create an empty array for our output
output_empty = [0] * len(classes)

# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # stem each word
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    # create our bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)

# create train and test lists
train_x = list(training[:,0])
train_y = list(training[:,1])


# Ready to build Model
"""
#######################################################################################
######## Please refer to http://tflearn.org/models/dnn/"


####################################################################################################################################3
"""
# reset underlying graph data
tf.reset_default_graph()
# Build neural network
#This first steos creates the input layer , we call it net
#len(train_x[0]) is the number of columns 81
net = tflearn.input_data(shape=[None, len(train_x[0])])
# We wwant fully connected layer where all neurons are onnected to all
#neurons from previous layers meaning 
#this line will create a fully connected layer with 8 hidden units/neurons
net = tflearn.fully_connected(net, 8)

net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir=pckle_path+'tflearn_logs')
# Start training (apply gradient descent algorithm)
model.fit(train_x, train_y, n_epoch=2000, batch_size=8, show_metric=True)
model.save(pckle_path+ 'model.tflearn')


# save all of our data structures

pickle.dump( {'words':words, 'classes':classes, 'train_x':train_x, 'train_y':train_y}, open( pckle_path+"training_data", "wb" ) )



