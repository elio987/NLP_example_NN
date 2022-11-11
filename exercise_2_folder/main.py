Leonardo Gracida Munoz A01379812
from flair.data import Corpus,Dictionary
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
#We get the new dictionary of tags
new_tag_dictionary = corpus.make_label_dictionary(label_type = "ner")
print(new_tag_dictionary)
#We get the previous tagger
previous_tagger: SequenceTagger = SequenceTagger.load("flair/ner-english-ontonotes-fast")
#We extract the previous embeddings of the pretrained model and insert the new tag dictionary
tagger: SequenceTagger = SequenceTagger(
        hidden_size=previous_tagger.hidden_size,
        embeddings=previous_tagger.embeddings,
        tag_dictionary=new_tag_dictionary,
        tag_type='ner',
    )

#Reuse que internal layers
tagger.embedding2nn = previous_tagger.embedding2nn
tagger.rnn = previous_tagger.rnn
#We create the trainer
trainer = ModelTrainer(tagger, corpus)
#We train the model
trainer.train('exercise_2_folder/resources/taggers/ner-english',train_with_dev=True,max_epochs=int(epochs),
              monitor_train = True, monitor_test = True,learning_rate = 0.1,embeddings_storage_mode='gpu')
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
    print(sentence.get_spans('ner'))