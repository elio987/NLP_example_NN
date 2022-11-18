# This doesn't seem to translate! maybe unfinished?

with open("europarl-v7.es-en.en","r") as file:
    lines_eng = file.readlines()
with open("europarl-v7.es-en.es","r") as file:
    lines_esp = file.readlines()
print(len(lines_eng[:100]))

with open("eng_100.txt","w") as file:
    for i in lines_eng[:100]:
        file.write(i)
with open("esp_100.txt","w") as file:
    for i in lines_esp[:100]:
        file.write(i)
