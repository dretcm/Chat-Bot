from gtts import gTTS
from playsound import playsound
import speech_recognition as sr
import os

import nltk
from nltk.stem.lancaster import LancasterStemmer
#nltk.download('punkt')
stemmer = LancasterStemmer()

import tensorflow as tf
import random
import json
import pickle
import time

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


            self.training = np.array(training)
            self.output = np.array(output)

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
            results_index = np.argmax(results)
            tag = self.labels[results_index]

            for tg in self.data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']

            return random.choice(responses)

MR = Bot()
r = sr.Recognizer() 

NOMBRE_ARCHIVO = "voz.mp3"

while True:
    with sr.Microphone() as source:
        print('Speak Anything : ')
        audio = r.listen(source)
    try:
        text = r.recognize_google(audio)
        print('You said: {}'.format(text))
    except:
        print('Sorry could not hear')

    message = MR.chat(text)

    tts = gTTS(message, lang='es') # lang='es-us') 'es-us': 'Spanish (United States)'

    tts.save(NOMBRE_ARCHIVO)

    playsound(NOMBRE_ARCHIVO)
    os.system(f'DEL /F /A {NOMBRE_ARCHIVO}')

    if text.lower() == 'adios':
        break

