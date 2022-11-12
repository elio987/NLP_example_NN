from googletrans import Translator
from nltk.translate.bleu_score import sentence_bleu
from transformers import pipeline
import numpy as np
def translate_blue_score(path_txt_eng,path_txt_esp):
    #Open the files with the sentences we want to translate
    with open(path_txt_eng,"r") as file:
        lines_eng = file.readlines()
    with open(path_txt_esp,"r") as file:
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