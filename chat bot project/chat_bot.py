import nltk
from nltk.stem.lancaster import LancasterStemmer
#nltk.download('punkt')
stemmer = LancasterStemmer()

import numpy
# import tflearn
import tensorflow as tf
import random
import json
import pickle

import time
from selenium import webdriver
import selenium

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation

class Bot:
    def __init__(self):
        with open("spanish.json") as file:
            self.data = json.load(file)

        try:
            with open("data.pickle", "rb") as f:
                self.words, self.labels, self.training, self.output = pickle.load(f)
        except:
            words = []
            labels = []
            docs_x = []
            docs_y = []

            for intent in self.data["intents"]:
                for pattern in intent["patterns"]:
                    wrds = nltk.word_tokenize(pattern)
                    words.extend(wrds)
                    docs_x.append(wrds)
                    docs_y.append(intent["tag"])

                if intent["tag"] not in labels:
                    labels.append(intent["tag"])

            words = [stemmer.stem(w.lower()) for w in words if w != "?"]
            self.words = sorted(list(set(words)))

            self.labels = sorted(labels)

            training = []
            output = []

            out_empty = [0 for _ in range(len(self.labels))]

            for x, doc in enumerate(docs_x):
                bag = []

                wrds = [stemmer.stem(w.lower()) for w in doc]

                for w in self.words:
                    if w in wrds:
                        bag.append(1)
                    else:
                        bag.append(0)

                output_row = out_empty[:]
                output_row[self.labels.index(docs_y[x])] = 1

                training.append(bag)
                output.append(output_row)


            self.training = numpy.array(training)
            self.output = numpy.array(output)

            with open("data.pickle", "wb") as f:
                pickle.dump((self.words, self.labels, self.training, self.output), f)

        # print(words)
        # print(labels)
        # print(training)
        # print(output)

        try:
            self.model = tf.keras.models.load_model('model.h5')
        except:
            #Definición de la arquitectura
            self.model = Sequential()
            self.model.add(Dense(128, input_dim=self.training.shape[1], kernel_initializer='normal',activation='relu'))
            self.model.add(Dense(16, kernel_initializer='normal',activation='relu'))
            self.model.add(Dense(16, kernel_initializer='normal',activation='relu'))
            self.model.add(Dense(len(self.output[0]), kernel_initializer='normal', activation='softmax'))

            # Compilación del modelo
            self.model.compile(loss="categorical_crossentropy", optimizer='adam',metrics=['accuracy'])

            self.model.fit(self.training, self.output, epochs=1000, batch_size=8)

            self.model.save('model.h5')

        # print(model.summary())

    def bag_of_words(self,s, words):
        bag = [0 for _ in range(len(words))]

        s_words = nltk.word_tokenize(s)
        s_words = [stemmer.stem(word.lower()) for word in s_words]

        for se in s_words:
            for i, w in enumerate(words):
                if w == se:
                    bag[i] = 1
                
        return bag


    def chat(self, wrs):

        x_data = np.array([self.bag_of_words(wrs, self.words)])
        results = self.model.predict(x_data)

        if np.max(results)*100 < float(90):
            return random.choice(["no te entiendo","podrias gestionar mejor tus palabras :)","what?"])
        else:
            results_index = numpy.argmax(results)
            tag = self.labels[results_index]

            for tg in self.data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']

            return random.choice(responses)

        
MR = Bot()

url = 'https://web.whatsapp.com/'
browser = webdriver.Chrome('chromedriver.exe')
browser.get(url)


time.sleep(10) # 10 secons


user = 'name of contact or number directory'


def search(ur, bwr):
    bwr.find_element_by_css_selector("#side > div._1Ra05 > div > label > div > div._1awRl.copyable-text.selectable-text").send_keys(ur)
    bwr.find_element_by_css_selector("#side > div._1Ra05 > div > label > div > div._1awRl.copyable-text.selectable-text").send_keys(webdriver.common.keys.Keys.ENTER)
def send(ms, bwr):
    bwr.find_element_by_css_selector("#main > footer > div._3SvgF._1mHgA.copyable-area > div.DuUXI._1WoQE > div > div._1awRl.copyable-text.selectable-text").send_keys(ms)
    bwr.find_element_by_css_selector("#main > footer > div._3SvgF._1mHgA.copyable-area > div.DuUXI._1WoQE > div > div._1awRl.copyable-text.selectable-text").send_keys(webdriver.common.keys.Keys.ENTER)   

search(user, browser)
send('-activate-',browser)

word = '-activate-'
ago_word = '-activate-'

while True:            
    chars = browser.find_element_by_class_name('_26MUt').text
    chars = list(chars.split('\n'))[:-1]
    ago_word = chars[-1]
    
    if word != ago_word:
        word = MR.chat(ago_word)
        send(word, browser)

    if ago_word.lower() == 'quit':
        send('-desactivate-', browser)
        break

time.sleep(60)

browser.close()
