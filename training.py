## Most code from https://towardsdatascience.com/how-to-create-a-chatbot-with-python-deep-learning-in-less-than-an-hour-56a063bdfc44
## Used to train a model to the contents within intents.json
# The file for reading the natural language data into a training set 
# and using a Keras sequential neural network to create a model

# Importing necesssities
# Natural Language Toolkit, tools for cleaning and preparing text for the DL algo
import nltk 
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle # To load pkl files
import numpy as np # To perform the linear operations

# Keras, deep learning framework to train the model
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

# Inititalise ChatBot training for where to store the language data
words = [] # List of words for chatbot training
classes = [] # A list of different types of classes of responses (i.e. greetings or help)
documents = [] 
ignore_words = ['?', '!', '.', ',', "'", '"']

# Opening and loading the JSON information
data_file = open('intents.json').read()
intents = json.loads(data_file)

# Loading the information from JSON to this file in the form of three aforementioned lists
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Take each word and tokenize it
        w = nltk.word_tokenize(pattern)
        words.extend(w)

        # Adding documents
        documents.append((w, intent['tag'])) # Each pair of patterns with corresponding tags

        # Adding classes to our class list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatizing the content (converting words to its basic form, such as worked to work)
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]

# Sort of the lists
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Testing what the results look like
#print (len(documents), "documents")
#print (len(classes), "classes", classes)
#print (len(words), "unique lemmatized words", words)
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

# Initialising the training data
training = []
output_empty = [0] * len(classes)
for doc in documents:
    # Initializing bag of words
    bag = []
    # List of tokenized words for the pattern
    pattern_words = doc[0]
    # Lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # Create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # Output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row]) 
# Shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)
# Create train and test lists. X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")

## The Deep Learning Model from Keras called Sequential (a neural network)
# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Fitting the model to a Histogram and saving the model to be used in chatbot.py
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist) # Model created here and used in chatbot.py

# Confirm the model has successfully been created
print("Model Created!")
