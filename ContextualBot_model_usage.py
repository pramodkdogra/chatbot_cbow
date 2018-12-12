# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 18:40:07 2018

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

# following steps are separate
# restore all of our data structures
import pickle

file="intents.json"
pathfile="C:/Users/prdogra/Sync/MSC_Cognitive/Year2018_2019/COS524_Natural Language Processing/Project_code/CBOW/data/"+file
pckle_path="C:/Users/prdogra/Sync/MSC_Cognitive/Year2018_2019/COS524_Natural Language Processing/Project_code/CBOW/pickle/"

#Greetings     
ALEX_GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey","hola", "whatsup", "Bonjour")
ALEX_GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "Glad to be talking to you", "I look forward to your questions on Cannabis"]
ALEX_EXIT_INPUTS = ("exit", "bye", "quit")
ALEX_THANKS=("See ya later, Bye." ,"Glad to be of Service, Bye.", "You take care now, Bye")

flag=True

data = pickle.load( open( pckle_path+"training_data", "rb" ) )
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

print(words)

stemmer = LancasterStemmer()
# import our chat-bot intents file

with open(pathfile) as json_data:
    intents = json.load(json_data)


#We will have load our saved Tensorflow (tflearn framework) model. First ww need to define the Tensorflow model structure 
# as we did in the prior section.
    
# load our saved model

# reset underlying graph data

tf.reset_default_graph()

# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)



# Define model and setup tensorboard

model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
model.load(pckle_path+'model.tflearn')

#Before we can begin processing intents, we need a way to produce a bag-of-words from user input. 
#This is the same technique as we used earlier to create our training documents.

def sentence_clean_up(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bagofword(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = sentence_clean_up(sentence)
    # bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))
    
p = bagofword("is your shop open today?", words)
print (p)

p = bagofword("What is Cannabis", words)
print (p)

#We are now ready to build our response processor.

ERROR_THRESHOLD = 0.25
def sentence_classify(sentence):
    # generate probabilities from the model
    results = model.predict([bagofword(sentence, words)])[0]
    # filter out predictions below a threshold
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    # return tuple of intent and probability
    #print(return_list)
    return return_list

def find_response(sentence, userID='123', show_details=False):
    results = sentence_classify(sentence)
    print("setence classify results")
    print(results)
    # if we have a classification then find the matching intent tag
    if results:
        # loop as long as there are matches to process
        while results:
            for i in intents['intents']:
                # find a tag matching the first result
                if i['tag'] == results[0][0]:
                    # a random response from the intent
                    return i['responses']
                    #return print(random.choice(i['responses']))

            print(results.pop(0))
            results.pop(0)


def alexGreeting(sentence):
 
    for word in sentence.split():
        if word.lower() in ALEX_GREETING_INPUTS:
            return random.choice(ALEX_GREETING_RESPONSES)

def process():
    global word_tokens, flag, speech_flag
    #translate = YandexTranslate('trnsl.1.1.20181104T001234Z.a1af2170d545df9c.7d91bb6753fe9159bf35278d97f857e6fb86de92')
    
    while(flag==True): 
        input_from_user=input()
        input_from_user=input_from_user.lower()
        print("user input " + input_from_user)
        #if(input_from_user!='exit'):
        if (input_from_user not in ALEX_EXIT_INPUTS):
            if(input_from_user=='thanks' or input_from_user=='thank you' ):
                flag=False
                speech_flag=False
                print("ALEX: You are welcome." + random.choice(ALEX_THANKS))
            else:
                if(alexGreeting(input_from_user)!=None):
                    print("ALEX: "+alexGreeting(input_from_user))
                else:
                    bot_response=find_response(input_from_user)
                                        
                    #final_words=list(set(word_tokens))
                    if bot_response != None:
                        print("ALEX: " + bot_response)
                    
        else:
            flag=False
            speech_flag=False
            print("ALEX: Good Bye! You take care now !")
                
#Each sentence passed to response() is classified. Our classifier uses model.predict() and is lighting fast. The probabilities returned by the model are lined-up with our intents definitions to produce a list of potential responses.
#If one or more classifications are above a threshold, we see if a tag matches an intent and then process that. We’ll treat our classification list as a stack and pop off the stack looking for a suitable match until we find one, or it’s empty.
#Let’s look at a classification example, the most likely tag and its probability are returned.  
            
sentence_classify('is your shop open today?')
sentence_classify('what is thc?')

#Notice that ‘is your shop open today?’ is not one of the patterns for this intent: “patterns”: [“Are you open today?”, “When do you open today?”, “What are your hours today?”] however the terms ‘open’ and ‘today’ proved irresistible to our model (they are prominent in the chosen intent).
#We can now generate a chatbot response from user-input:

find_response('is your shop open today?')
find_response('what is CBD?')
find_response('define CBD?')
find_response('what can you tell me about CBD?')
find_response('What is Cannabis made up of?')
find_response('Tell me about CBD?')
find_response('do you take cash?')
find_response('Goodbye, see you later')


    
def main():
    global flag
    print("\n\n ALEX: Hi! My name is Alex (CBOW). I will answer your queries about Cannabis Legalization in Canada. \n\n I am designed to hear your Speech. Do you want to ask your question Verbally? \n If you yes then please enter Yes and and say your question or else enter No followed by typing your question. \n\n If you want to exit this chat, type Thanks, Exit or Bye or Quit at any time!")
    
    process()
    flag=False


if __name__ == "__main__":
    main()               