#Leonardo Gracida Munoz A01379812
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.data import Sentence
from flair.models import SequenceTagger
from flair.visual.training_curves import Plotter
#Number of examples of train
N_EXAMPLES_TO_TRAIN = 600
#Tag of the data we will use
columnas = {0 : 'text', 1 : 'ner'}
#Folder where is stored the dataset, in this case the dataset is in the same folder
data_folder = ''
#We create the corpues with the complete dataset
corpus: Corpus = ColumnCorpus(data_folder, columnas,
                              train_file = 'train',
                              test_file = 'test',
                              dev_file = 'dev')
#We do downsampling to train with the number of examples we want.
corpus= corpus.downsample(N_EXAMPLES_TO_TRAIN/len(corpus.train))
#We check one of the test tweets
print(corpus.train[2].to_tagged_string('ner'))
#we check the structure of the corpus
print(corpus)
#tag_dictionary = corpus.make_label_dictionary(label_type = "ner")
#print(tag_dictionary)
#We select the model we want to train
tagger = SequenceTagger.load("flair/ner-english-ontonotes-fast")
#We create the trainer
trainer = ModelTrainer(tagger, corpus)
#We train the model
trainer.train('resources/taggers/ner-english',max_epochs=7,
              monitor_train = True, monitor_test = True)
#Load the model
model = SequenceTagger.load('resources/taggers/ner-english/final-model.pt')

#We create a png image with the loss or error of the train and test
plotter = Plotter()
plotter.plot_training_curves('resources/taggers/ner-english/loss.tsv')

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