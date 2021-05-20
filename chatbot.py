## Main file to run the actual bot
# Code cleans up the responses based on predections for the generated model in training.py

# Importing the necessities
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from tensorflow.keras.models import load_model
import random

lemmatizer = WordNetLemmatizer()

# Loading the needed content 
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb')) # rb means 'read binary mode'
classes = pickle.load(open('classes.pkl','rb'))
model = load_model('chatbot_model.h5')

# Clean up any sentence from the user
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Takes the sentences that are cleaned up and creates a bag of words that are used for predicting classes 
# (which are based off the results from the training model earlier).
def bow(sentence, words, show_details=True):
    # Tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # Bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                # Assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

# This function will output a list of intents and the probabilities,
# their likelihood of matching the correct intent
def predict_class(sentence, model):
    # Filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25 # Anything with error higher than 25% will not be considered
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

# Takes the list outputted and checks the JSON file 
# to output the most correct response with the highest probability.
def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

# Infinite loop to ask questions and receive answers based off the trained model
while True:
    message = input("") # Insert into here a user's command in Discord 
    ints = predict_class(message, model) 
    res = getResponse(ints, intents)
    print(res) # Replace this with displaying text in Discord


