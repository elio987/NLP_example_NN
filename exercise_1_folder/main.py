#Leonardo Gracida Munoz A01379812
#Import the libraries to do sentiment analysis
from transformers import pipeline
#library that contains de punctation characters
import string


# It’s best practice to have your code in a descriptive method or small class, if possible, rather than running at the top-level.  makes it easier for other modules to import the functionality later if needed! 
#We import the sentimental analysis model
sentiment_pipeline = pipeline("sentiment-analysis")
#Open de model and extract all the reviews
with open('tiny_movie_reviews_dataset.txt','r') as file:
    lines = file.readlines()
    for line in lines: # only use i for number variables, and in general try to give descriptive names to vars! 
        #For each line que extract the punctuation characters, for example: ""
        """The translate function replace characters with other characters, and the function maketrans,
        creates a dictionary to can replace characters with other characters and delete characters of the string."""
        line = line.translate(str.maketrans('', '', string.punctuation))
        #Print the review and the prediction of the review
        print("Review:")
        print(line)
        print("Prediction:")
        print(sentiment_pipeline(line)) # make sure this is formatted like the homework writeup describes! 
