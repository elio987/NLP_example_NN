#Leonardo Gracida Munoz A01379812
#Import the task functions
from exercise1_func import SentimentalModel
from exercise2_func import trainer_model,test,train_model
from exercise3_func import Translate
from flair.models import SequenceTagger

"""Exercise1"""
model = "sentiment-analysis"
#The path of the dataset of the reviews
file_path = 'exercise_1_folder/tiny_movie_reviews_dataset.txt'
#We start the sentimental analysis
Sentimental = SentimentalModel(model=model,file_path=file_path)
Sentimental.sentimental_analysis()
"""Exercise2"""
#The name of the flair pretrained model we want to retrain
model_name = "flair/ner-english-ontonotes-fast"
#We get the trainer and the corpus of te dataset and model
trainer,corpus,epochs = trainer_model(model_name)
train_model(trainer,epochs)
#Load the model
model = SequenceTagger.load('exercise_2_folder/resources/taggers/ner-english/final-model.pt')
#We do some predictiones with the the test corpus
test(model,corpus)

"""Exercise3"""
#The paths the english and spanish text
path_txt_eng = "exercise_3_folder/eng_100.txt"
path_txt_esp = "exercise_3_folder/esp_100.txt"
#We call the apis and get the bleu score
Traductor = Translate(path_txt_eng=path_txt_eng,path_txt_esp=path_txt_esp)
Traductor.translate_blue_score()
