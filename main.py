from flask import Flask, render_template, request
from flask import jsonify
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy
import tflearn
import tensorflow
import random
import json
import os

app = Flask(__name__)


with open("intents.json",encoding="utf8") as file:
    data = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []
suggestion=[] 
   

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])    

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)


training = numpy.array(training)
output = numpy.array(output)

tensorflow.reset_default_graph()
net = tflearn.input_data(shape=[None,len(training[0])])
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,len(output[0]),activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

model.fit(training,output,n_epoch=25000,batch_size=50,show_metric=True)
model.save("chatbot.tflearn")

def bag_of_words(s, words): 
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat',methods=["POST"])
def chat():
    #while True:
     ##  if inp.lower() == "quit":
       #     break
    user_input = request.form['user_input']
    results = model.predict([bag_of_words(user_input, words)])[0]
    results_index = numpy.argmax(results)
    tag = labels[results_index]
        
    if results[results_index] >0.6:
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
        resval=random.choice(responses)
    else:
        resval="Sorry I don't understand,try asking something else."
    
         

    for tg in data["intents"]:
        if tg['tag'] == tag:
            suggestion=(tg['patterns'])
            if resval =="Sorry I don't understand,try asking something else.":
                suggestion = "You can try:"+', '.join(tg['patterns'])
        return render_template('index.html',user_input=user_input,bot_response=resval)


if __name__ == '__main__':
    app.run(debug=True)




