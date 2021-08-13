import numpy as np 
import pandas as pd
#pour l'accès au système des fichier afin de récuprer les images du dataset
import os
from sklearn.utils import shuffle 
import matplotlib.pyplot as plt 
#pour le chargement des images 
import cv2
#pour effectuer les differentes étapes du CNN
from keras import models,layers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
#pour faire de la matrice de confusion
from sklearn.metrics import confusion_matrix 
import matplotlib.image as mpimg
import tensorflow as tf 

import itertools

#Définition et l'initialisation des paramètres


train_folder='dataset\training_set'
test_folder='dataset\test_set'
Class_label=['beaches','bus','dinosaurs','elephants','flowers','foods','horses','monuments','mountains_and_snow','people_and_villages_in_Africa']
IMG_HEIGHT=200
IMG_WIDTH=200
input_size=(200,200)

#Chargement des données
 def create_dataset_CV(data_folder):

    dataset=[]
    img_data_array=[]
    class_num=[]
    image_train= []
    i=0
    #parourir tous les dossiers du training dataset
    for d in os.listdir(data_folder):
        #parcourir toutes les images dans le dossier 
        for file in os.listdir(os.path.join(data_folder, d)):
            #Concaténer le chemin avec le data folder 
            image_path= os.path.join(data_folder, d,  file)
            #lire l'image et récuperer le RGB et la redimmensionner
            img= cv2.imread(image_path)  
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img,input_size) 
            #changer le type de l'image et la normaliser
            img=np.array(img)
            img = img.astype('float32')
            img/= 255 
            #ajouter l'image et son étiquette à la liste 
            img_data_array.append(img)
            class_num.append(i)
        i=i+1
         
    img_data_array=np.array(img_data_array)
    class_num=np.array(class_num)   
    return img_data_array,class_num


#Récuperer les données en appleant la fonction create_dataset_CV
imgtrain1,labeltrain1=create_dataset_CV(train_folder)
imgtest1,labeltest1=create_dataset_CV(test_folder)
#Rendre les images en type array 
imgtrain1=np.array(imgtrain1)
labeltrain1=np.array(labeltrain1)
imgtest1=np.array(imgtest1)
labeltest1=np.array(labeltest1)


#Définition de l'architecture NET-5 et les différentes couches du modèle CNN
#Net-5
model = tf.keras.Sequential([
#1 couche conventionelle avec taille de fentres 3,padding insensible et une la fonction d'activiation Relu 
tf.keras.layers.Conv2D(6, kernel_size=3, strides=1,  activation='relu', padding='same',input_shape=(200,200,3)),
#une couche average pooling
tf.keras.layers.AveragePooling2D(pool_size=(2,2),strides=(2,2),padding='valid'),
# couche conventionnelle 
tf.keras.layers.Conv2D(16, kernel_size=3, strides=1, activation='relu', padding='valid'),
#une couche average pooling
tf.keras.layers.AveragePooling2D(pool_size=(2,2),strides=(2,2),padding='valid'), 
tf.keras.layers.Flatten(),
###########################################################################
tf.keras.layers.Dense(120, activation='relu'),
#dans notre modèle elle existe 10 classes 
tf.keras.layers.Dense(units=10, activation="softmax")])

#l'optimiseur adam qui prend en compte le gradient afin d'accélerer la vitesse d'apprentissage
#metrics accuracy c'est pour mesurer le nombre de prédiction correcte sur le nombre total de prédiction 
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


#Entrainement du modèle 
hist = model.fit(imgtrain1,labeltrain1, batch_size=128, epochs=4, validation_split = 0.3)

#Evaluation du modèle 
plt.plot(hist.history['accuracy'],label='accuracy')
plt.plot(hist.history['val_accuracy'],label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
#plt.ylim([0.005, 1])
plt.legend(loc='lower right')

test_loss,test_acc = model.evaluate(imgtrain1,labeltrain1, verbose=2)

#prédire le label pour une image  
pred=model.predict(imgtest1)
predlabel=np.argmax(pred,axis=1)
i=np.random.randint(pred.shape[0])
#afficher l'image
plt.figure()
plt.imshow(imgtest1[i])
k=predlabel[i]
plt.title('Image #{}'.format(i)+Class_label[k])