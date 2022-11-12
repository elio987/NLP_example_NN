#Leonardo Gracida Munoz A01379812
#Import the libraries to do sentimental analysis
from transformers import pipeline
#library that contains de puntiation characters
import string
def sentiment_analysis(file_path):
    """Function that do sentimental analysis of text reviews"""
    #We import the sentimental analysis model
    sentiment_pipeline = pipeline("sentiment-analysis")
    #Open de model and extract all the reviews
    with open(file_path,'r') as file:
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