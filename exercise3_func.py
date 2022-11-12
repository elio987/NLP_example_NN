from googletrans import Translator
from nltk.translate.bleu_score import sentence_bleu
from transformers import pipeline
import numpy as np
import unittest

class Translate():
    def __init__(self,path_txt_eng,path_txt_esp):
        #Open the files with the sentences we want to translate
        with open(path_txt_eng,"r") as file:
            self.lines_eng = file.readlines()
        with open(path_txt_esp,"r") as file:
            self.lines_esp = file.readlines()

    def translate_blue_score(self):
        #We start the translator
        translator = Translator(service_urls=['translate.googleapis.com'])
        mean = []
        prueb_esp = []
        #We do the translations
        for i in range(100):
            trans = translator.translate(self.lines_eng[i], dest='es').extra_data["possible-translations"][0][2]
            for j in trans:
                prueb_esp.append(j[0].split())
            #We get the blue score with a corpus of 2
            mean.append(sentence_bleu(prueb_esp,self.lines_esp[i].split(),weights=(0, 1, 0, 0)))
        mean_google = np.round(np.mean(mean),2)
        print("Google translator blue score: ",mean_google)
        #We start the translator of HuggingFace
        translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")
        mean = []
        prueb_esp = []
        for i in range(100):
            trans = translator(self.lines_eng[i])
            prueb_esp.append(trans[0]['translation_text'].split())
            #We do the blue score with a corpus of two
            mean.append(sentence_bleu(prueb_esp,self.lines_esp[i].split(),weights=(0, 1, 0, 0)))
        mean_hug = np.round(np.mean(mean),2)
        print("Huggingface translator blue score: ",mean_hug)
        return [mean_google,mean_hug]

class TraductorTest(unittest.TestCase):
    def test_3(self):
        path_txt_eng = "exercise_3_folder/eng_100.txt"
        path_txt_esp = "exercise_3_folder/esp_100.txt"
        translateBlue = Translate(path_txt_eng=path_txt_eng,path_txt_esp=path_txt_esp)
        translateBlueScore = translateBlue.translate_blue_score()
        print(translateBlueScore)
        blue_score_true = [0.49,0.48]
        self.assertEqual(blue_score_true,translateBlueScore)
