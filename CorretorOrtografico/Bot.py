# things we need for NLP
# import our chat-bot intents file
import json
import random

import nltk
# things we need for Tensorflow
import numpy as np
import tensorflow as tf
import tflearn
from nltk.stem.lancaster import LancasterStemmer

from Classes.Voz import Voz


class Bot:
    def __init__(self,treinar=False):

        self.stemmer = LancasterStemmer()

        with open('Arquivos/intents.json') as json_data:
            self.intents = json.load(json_data)

        self.words = []
        self.classes = []
        self.documents = []
        self.ignore_words = ['?']
        # loop through each sentence in our intents patterns
        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                # tokenize each word in the sentence
                w = nltk.word_tokenize(pattern)
                # add to our words list
                self.words.extend(w)
                # add to documents in our corpus
                self.documents.append((w, intent['tag']))
                # add to our classes list
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])

        # stem and lower each word and remove duplicates
        self.words = [self.stemmer.stem(w.lower()) for w in self.words if w not in self.ignore_words]
        self.words = sorted(list(set(self.words)))

        # remove duplicates
        self.classes = sorted(list(set(self.classes)))

        print (len(self.documents), " Documentos")
        print (len(self.classes), " Classes: --\n", self.classes)
        print (len(self.words), " Palavras originais: \n", self.words)

        # create a data structure to hold user context
        self.context = {}
        self.ERROR_THRESHOLD = 0.25
        self.voz = Voz()
        self.treinar(treinar)

    def treinar(self,treinar):

        # create our training data
        training = []
        output = []
        # create an empty array for our output
        output_empty = [0] * len(self.classes)

        # training set, bag of words for each sentence
        for doc in self.documents:
            # initialize our bag of words
            bag = []
            # list of tokenized words for the pattern
            pattern_words = doc[0]
            # stem each word
            pattern_words = [self.stemmer.stem(word.lower()) for word in pattern_words]
            # create our bag of words array
            for w in self.words:
                bag.append(1) if w in pattern_words else bag.append(0)

            # output is a '0' for each tag and '1' for current tag
            output_row = list(output_empty)
            output_row[self.classes.index(doc[1])] = 1

            training.append([bag, output_row])

        # shuffle our features and turn into np.array
        random.shuffle(training)
        training = np.array(training)

        # create train and test lists
        train_x = list(training[:,0])
        train_y = list(training[:,1])

        # reset underlying graph data
        tf.reset_default_graph()
        # Build neural network
        net = tflearn.input_data(shape=[None, len(train_x[0])])
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
        net = tflearn.regression(net)

        # Define model and setup tensorboard
        self.model = tflearn.DNN(net, tensorboard_dir='Modelo/tflearn_logs')
        # Start training (apply gradient descent algorithm)
        if(treinar):
            self.model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
            self.model.save('Modelo/model.tflearn')

            # save all of our data structures
            import pickle
            pickle.dump({'words': self.words, 'classes': self.classes, 'train_x': train_x, 'train_y': train_y},
                        open("Modelo/training_data", "wb"))
        else:
            # load our saved model
            self.model.load('./Modelo/model.tflearn')



    def clean_up_sentence(self,sentence):
        # tokenize the pattern
        sentence_words = nltk.word_tokenize(sentence)
        # stem each word
        sentence_words = [self.stemmer.stem(word.lower()) for word in sentence_words]
        return sentence_words

    # return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
    def bow(self,sentence, words, show_details=False):
        # tokenize the pattern
        sentence_words = self.clean_up_sentence(sentence)
        # bag of words
        bag = [0]*len(words)
        for s in sentence_words:
            for i,w in enumerate(words):
                if w == s:
                    bag[i] = 1
                    if show_details:
                        print ("found in bag: %s" % w)

        return(np.array(bag))


#p = bow("is your shop open today?", self.words)
#print (p)
#print (self.classes)
#print("\n")

#print(model.predict([p]))

    def classify(self, sentence):
        # generate probabilities from the model
        results = self.model.predict([self.bow(sentence, self.words)])[0]
        # filter out predictions below a threshold
        results = [[i,r] for i,r in enumerate(results) if r>self.ERROR_THRESHOLD]
        # sort by strength of probability
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append((self.classes[r[0]], r[1]))
        # return tuple of intent and probability
        return return_list

    def response(self, sentence, userID='123', show_details=False):
        results = self.classify(sentence)
        # if we have a classification then find the matching intent tag
        if results:
            # loop as long as there are matches to process
            while results:
                for i in self.intents['intents']:
                    # find a tag matching the first result
                    if i['tag'] == results[0][0]:
                        # set context for this intent if necessary
                        if 'context_set' in i:
                            if show_details: print ('context:', i['context_set'])
                            self.context[userID] = i['context_set']

                        # check if this intent is contextual and applies to this user's conversation
                        if not 'context_filter' in i or \
                            (userID in self.context and 'context_filter' in i and i['context_filter'] == self.context[userID]):
                            if show_details: print ('tag:', i['tag'])
                            # a random response from the intent
                            retorno = random.choice(i['responses'])
                            return retorno

                results.pop(0)

    def teste(self):



        while(True):
            nome = input("Digite: ")
            retorno = self.response(nome)
            print(retorno)
            self.voz.entrada(retorno)
#a.teste()


#a.response('is your shop open tod"Olá " + frase + ", tudo bem?"ay?')
#a.response('Can we rent a moped?')
#a.response('today')
#a.response('Thank rou')
