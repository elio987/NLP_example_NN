# NLP_example_NN
Here we use three different NN, NER, sentimen analysis and translator
- The run.py file call the functions and classes of the three tasks, and each folder has each exercise separated and the dataset of each exercise.
- We call the functions or class of the task using three differents py files "exercise1_func.py,exercise2_func.py,exercise3_func.py".
- The model of the exercise 2 is inside que folder called "exercise_2_folder/resources/taggers/ner-english/final-model.pt", but you need to train the model first to create the folder.<br>
- To run the class test for exercise 1 run: <br>
```python3 -m unittest exercise1_func.py ```
- To run the class test for exercise 3 run: <br>
```python3 -m unittest exercise3_func.py ```
- To install and run "run.py" use the next command if your default python version is Python3.x:<br>
```pip install -r requirements.txt && python run.py```
- If not:<br>
```pip3 install -r requirements.txt && python3 run.py```<br>
- The learning line or plot of the loss while training and testing is inside "exercise_2_folder/resources/training.png", but here there is an example of the final plot (using the complete dataset):<br>
![Alt text](training.png?raw=true "Learning line")
