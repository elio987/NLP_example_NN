#Leonardo Gracida Munoz A01379812
#Import the libraries to do sentimental analysis
from transformers import pipeline
#library that contains de puntiation characters
import string
import unittest

class SentimentalModel():
    def __init__(self,model,file_path):
        #We import the sentimental analysis model
        self.sentiment_pipeline = pipeline(model)
        self.file_path = file_path

    def sentimental_analysis(self):
        """Function that do sentimental analysis of text reviews"""
        #Open de model and extract all the reviews
        predictions = []
        with open(self.file_path,'r') as file:
            lines = file.readlines()
            for i in lines:
                #For each line que extract the punctiation characters, for example: ""
                """The translate function replace checracters with other characters, and the function maketrans,
                creates a dictionary to can replace characters with other characters and delete characters of the string."""
                i = i.translate(str.maketrans('', '', string.punctuation))
                #Print the prediction of the review
                pred = self.sentiment_pipeline(i)[0]['label']
                predictions.append(pred)
                print(pred)
        return predictions

class SentimentalTest(unittest.TestCase):
    def test_1(self):
        model = "sentiment-analysis"
        file_path = 'exercise_1_folder/tiny_movie_reviews_dataset.txt'
        sentimentalModel = SentimentalModel(model = model,file_path=file_path)
        predictions = sentimentalModel.sentimental_analysis()
        true_preds = ["NEGATIVE","POSITIVE","POSITIVE","NEGATIVE","NEGATIVE","POSITIVE","NEGATIVE",
            "POSITIVE","NEGATIVE","POSITIVE","POSITIVE","POSITIVE","NEGATIVE","NEGATIVE","POSITIVE",
            "POSITIVE","POSITIVE","POSITIVE","POSITIVE","NEGATIVE"]
        self.assertEqual(predictions,true_preds)
