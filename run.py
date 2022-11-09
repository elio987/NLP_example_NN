"""Exercise 1"""
#Leonardo Gracida Munoz A01379812
#Import the libraries to do sentimental analysis
from transformers import pipeline
#library that contains de puntiation characters
import string
#We import the sentimental analysis model
sentiment_pipeline = pipeline("sentiment-analysis")
#Open de model and extract all the reviews
with open('exercise_1_folder/tiny_movie_reviews_dataset.txt','r') as file:
    lines = file.readlines()
    for i in lines:
        #For each line que extract the punctiation characters, for example: ""
        """The translate function replace checracters with other characters, and the function maketrans,
        creates a dictionary to can replace characters with other characters and delete characters of the string."""
        i = i.translate(str.maketrans('', '', string.punctuation))
        #Print the review and the prediction of the review
        print("Review:")
        print(i)
        print("Prediction:")
        print(sentiment_pipeline(i))

"""Exercise 2"""
#Leonardo Gracida Munoz A01379812
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.data import Sentence
from flair.models import SequenceTagger
from flair.visual.training_curves import Plotter
#Tag of the data we will use
columnas = {0 : 'text', 1 : 'ner'}
#Folder where is stored the dataset, in this case the dataset is in the same folder
data_folder = 'exercise_2_folder'
#We create the corpues with the complete dataset
corpus: Corpus = ColumnCorpus(data_folder, columnas,train_file = 'train',test_file = 'test',dev_file = 'dev')
print("Insert number of samples to train, max samples: ",str(len(corpus.train)))
#Number of examples of train
N_EXAMPLES_TO_TRAIN = input("Insert: ")
#Number of epochs
epochs = input("Number of epochs: ")
if int(N_EXAMPLES_TO_TRAIN) > len(corpus.train):
    N_EXAMPLES_TO_TRAIN = len(corpus.train)
#We do downsampling to train with the number of examples we want.
corpus= corpus.downsample(int(N_EXAMPLES_TO_TRAIN)/len(corpus.train))
#We check one of the test tweets
print(corpus.train[2].to_tagged_string('ner'))
#we check the structure of the corpus
print(corpus)
tag_dictionary = corpus.make_label_dictionary(label_type = "ner")
#print(tag_dictionary)
#We select the model we want to train
tagger = SequenceTagger.load("flair/ner-english-ontonotes-fast")
#We create the trainer
trainer = ModelTrainer(tagger, corpus)
#We train the model
trainer.train('exercise_2_folder/resources/taggers/ner-english',train_with_dev=True,max_epochs=int(epochs),
              monitor_train = True, monitor_test = True)
#Load the model
model = SequenceTagger.load('exercise_2_folder/resources/taggers/ner-english/final-model.pt')

#We create a png image with the loss or error of the train and test
plotter = Plotter()
plotter.plot_training_curves('exercise_2_folder/resources/taggers/ner-english/loss.tsv')

#We will do 10 predictions
for i in range(10):
    exp = corpus.test[i].to_tagged_string('text')
    #Print the text
    print(exp)
    sentence = Sentence(exp)
    #Lets do the prediction
    model.predict(sentence)
    #We print the entities
    print("Entities: ")
    for entity in sentence.get_spans('ner'):
        print(entity)

"""Exercise 3"""

#Import the libraries
from googletrans import Translator
from nltk.translate.bleu_score import sentence_bleu
from transformers import pipeline
import numpy as np
#Open the files with the sentences we want to translate
with open("exercise_3_folder/eng_100.txt","r") as file:
    lines_eng = file.readlines()
with open("exercise_3_folder/esp_100.txt","r") as file:
    lines_esp = file.readlines()
#We start the translator
translator = Translator(service_urls=['translate.googleapis.com'])
mean = []
prueb_esp = []
#We do the translations
for i in range(100):
    trans = translator.translate(lines_eng[i], dest='es').extra_data["possible-translations"][0][2]
    for j in trans:
        prueb_esp.append(j[0].split())
    #We get the blue score with a corpus of 2
    mean.append(sentence_bleu(prueb_esp,lines_esp[i].split(),weights=(0, 1, 0, 0)))
print("Google translator blue score: ",np.mean(mean))
#We start the translator of HuggingFace
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")
mean = []
prueb_esp = []
for i in range(100):
    trans = translator(lines_eng[i])
    prueb_esp.append(trans[0]['translation_text'].split())
    #We do the blue score with a corpus of two
    mean.append(sentence_bleu(prueb_esp,lines_esp[i].split(),weights=(0, 1, 0, 0)))
    #print(prueb_esp,lines_esp[i].split())
print("Huggingface translator blue score: ",np.mean(mean))