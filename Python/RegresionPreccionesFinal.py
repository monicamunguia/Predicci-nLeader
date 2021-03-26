# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 14:11:37 2021

@author: Edwin
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt

#Extraccion de dataset
#Leer archivo del dataset
datos = pd.read_excel('Respuesta.xlsx')
dataframe = pd.DataFrame(datos)
datos.head()
encoder = LabelEncoder()
#datos["62.- Sexo"]=encoder.fit_transform(datos["62.- Sexo"])
datos['62.- Sexo']=encoder.fit_transform(datos['62.- Sexo'])
datos.head()
#print(datos)

print("Introduce tu edad")
edad = input()
print("¿Cuantas personas tienes a tu cargo?")
persC = input()
print("Introduce tu sexo", "\n", "0 = Femenino \n", "1 = Masculino")
sexo = input()

#Pregunta 1

X1 = (dataframe[["63.- Edad", "66.- Personas a mi cargo (en número)", "62.- Sexo"]])
y1 = (dataframe[["1.- Mis estrategias son para gestionar proyectos de innovación."]])
#Entrenamiento
X1_train,X1_test,y1_train,y1_test = train_test_split(X1, y1, test_size=0.25, random_state=0)
model= LogisticRegression()
model.fit(X1_train,y1_train)
datanew = {
           '63.- Edad': [edad],
           '66.- Personas a mi cargo (en número)': [persC],
           '62.- Sexo': [sexo],
           }
df1 = pd.DataFrame(datanew,columns=['63.- Edad', '66.- Personas a mi cargo (en número)', '62.- Sexo']) 
prediccion1=model.predict(df1)



#Pregunta 2

X2 = (dataframe[["63.- Edad", "66.- Personas a mi cargo (en número)", "62.- Sexo"]])
y2 = (dataframe[["2.- Como parte de la cultura organizacional para mí la creatividad es muy importante."]])
X2_train,X2_test,y2_train,y2_test = train_test_split(X2, y2, test_size=0.25, random_state=0)
model= LogisticRegression()
model.fit(X2_train,y2_train)
datanew = {
           '63.- Edad': [edad],
           '66.- Personas a mi cargo (en número)': [persC],
           '62.- Sexo': [sexo],
           }
df2 = pd.DataFrame(datanew,columns=['63.- Edad', '66.- Personas a mi cargo (en número)', '62.- Sexo']) 
prediccion2=model.predict(df2)



#Pregunta 3

X3 = (dataframe[["63.- Edad", "66.- Personas a mi cargo (en número)", "62.- Sexo"]])
y3 = (dataframe[["3.- Propongo innovaciones en la organización."]])
X3_train,X3_test,y3_train,y3_test = train_test_split(X3, y3, test_size=0.25, random_state=0)
model= LogisticRegression()
model.fit(X3_train,y3_train)
datanew = {
           '63.- Edad': [edad],
           '66.- Personas a mi cargo (en número)': [persC],
           '62.- Sexo': [sexo],
           }
df3 = pd.DataFrame(datanew,columns=['63.- Edad', '66.- Personas a mi cargo (en número)', '62.- Sexo']) 
prediccion3=model.predict(df3)




#Pregunta 4

X4 = (dataframe[["63.- Edad", "66.- Personas a mi cargo (en número)", "62.- Sexo"]])
y4 = (dataframe[["4.- Me doy tiempo para trabajar en mis propias ideas o proyectos."]])
X4_train,X4_test,y4_train,y4_test = train_test_split(X4, y4, test_size=0.25, random_state=0)
model= LogisticRegression()
model.fit(X4_train,y4_train)
datanew = {
           '63.- Edad': [edad],
           '66.- Personas a mi cargo (en número)': [persC],
           '62.- Sexo': [sexo],
           }
df4 = pd.DataFrame(datanew,columns=['63.- Edad', '66.- Personas a mi cargo (en número)', '62.- Sexo']) 
prediccion4=model.predict(df4)




#Pregunta 5

X5 = (dataframe[["63.- Edad", "66.- Personas a mi cargo (en número)", "62.- Sexo"]])
y5 = (dataframe[["5.- He propuesto y/o desarrollado productos nuevos en las áreas donde me he desempeñado."]])
X5_train,X5_test,y5_train,y5_test = train_test_split(X5, y5, test_size=0.25, random_state=0)
model= LogisticRegression()
model.fit(X5_train,y5_train)
datanew = {
           '63.- Edad': [edad],
           '66.- Personas a mi cargo (en número)': [persC],
           '62.- Sexo': [sexo],
           }
df5 = pd.DataFrame(datanew,columns=['63.- Edad', '66.- Personas a mi cargo (en número)', '62.- Sexo']) 
prediccion5=model.predict(df5)




#Pregunta 6

X6 = (dataframe[["63.- Edad", "66.- Personas a mi cargo (en número)", "62.- Sexo"]])
y6 = (dataframe[["6.- Discuto abiertamente con mis compañeros acerca de cumplir con los objetivos y metas de la empresa."]])
X6_train,X6_test,y6_train,y6_test = train_test_split(X6, y6, test_size=0.25, random_state=0)
model= LogisticRegression()
model.fit(X6_train,y6_train)
datanew = {
           '63.- Edad': [edad],
           '66.- Personas a mi cargo (en número)': [persC],
           '62.- Sexo': [sexo],
           }
df6 = pd.DataFrame(datanew,columns=['63.- Edad', '66.- Personas a mi cargo (en número)', '62.- Sexo']) 
prediccion6=model.predict(df6)




#Pregunta 7

X7 = (dataframe[["63.- Edad", "66.- Personas a mi cargo (en número)", "62.- Sexo"]])
y7 = (dataframe[["7.- Mantengo un diálogo abierto con aquellos que no están de acuerdo conmigo."]])
X7_train,X7_test,y7_train,y7_test = train_test_split(X7, y7, test_size=0.25, random_state=0)
model= LogisticRegression()
model.fit(X7_train,y7_train)
datanew = {
           '63.- Edad': [edad],
           '66.- Personas a mi cargo (en número)': [persC],
           '62.- Sexo': [sexo],
           }
df7 = pd.DataFrame(datanew,columns=['63.- Edad', '66.- Personas a mi cargo (en número)', '62.- Sexo']) 
prediccion7=model.predict(df7)



#Pregunta 8

X8 = (dataframe[["63.- Edad", "66.- Personas a mi cargo (en número)", "62.- Sexo"]])
y8 = (dataframe[["8.- Me expreso abiertamente en las reuniones."]])
X8_train,X8_test,y8_train,y8_test = train_test_split(X8, y8, test_size=0.25, random_state=0)
model= LogisticRegression()
model.fit(X8_train,y8_train)
datanew = {
           '63.- Edad': [edad],
           '66.- Personas a mi cargo (en número)': [persC],
           '62.- Sexo': [sexo],
           }
df8 = pd.DataFrame(datanew,columns=['63.- Edad', '66.- Personas a mi cargo (en número)', '62.- Sexo']) 
prediccion8=model.predict(df8)



#Pregunta 9

X9 = (dataframe[["63.- Edad", "66.- Personas a mi cargo (en número)", "62.- Sexo"]])
y9 = (dataframe[["9.- Doy información importante respecto a nuevas soluciones al trabajar en innovación."]])
X9_train,X9_test,y9_train,y9_test = train_test_split(X9, y9, test_size=0.25, random_state=0)
model= LogisticRegression()
model.fit(X9_train,y9_train)
datanew = {
           '63.- Edad': [edad],
           '66.- Personas a mi cargo (en número)': [persC],
           '62.- Sexo': [sexo],
           }
df9 = pd.DataFrame(datanew,columns=['63.- Edad', '66.- Personas a mi cargo (en número)', '62.- Sexo']) 
prediccion9=model.predict(df9)



#Pregunta 10

X10 = (dataframe[["63.- Edad", "66.- Personas a mi cargo (en número)", "62.- Sexo"]])
y10 = (dataframe[["10.- Preparo a la organización para los cambios."]])
X10_train,X10_test,y10_train,y10_test = train_test_split(X10, y10, test_size=0.25, random_state=0)
model= LogisticRegression()
model.fit(X10_train,y10_train)
datanew = {
           '63.- Edad': [edad],
           '66.- Personas a mi cargo (en número)': [persC],
           '62.- Sexo': [sexo],
           }
df10 = pd.DataFrame(datanew,columns=['63.- Edad', '66.- Personas a mi cargo (en número)', '62.- Sexo']) 
prediccion10=model.predict(df10)



#Pregunta 11

X11 = (dataframe[["63.- Edad", "66.- Personas a mi cargo (en número)", "62.- Sexo"]])
y11 = (dataframe[["11.- Demuestro originalidad en mi trabajo."]])
#Entrenamiento
X11_train,X11_test,y11_train,y11_test = train_test_split(X11, y11, test_size=0.25, random_state=0)
model= LogisticRegression()
model.fit(X11_train,y11_train)
datanew = {
           '63.- Edad': [edad],
           '66.- Personas a mi cargo (en número)': [persC],
           '62.- Sexo': [sexo],
           }
df11 = pd.DataFrame(datanew,columns=['63.- Edad', '66.- Personas a mi cargo (en número)', '62.- Sexo']) 
prediccion11=model.predict(df11)



#Pregunta 12

X12 = (dataframe[["63.- Edad", "66.- Personas a mi cargo (en número)", "62.- Sexo"]])
y12 = (dataframe[["12.- Conozco la magnitud y la estructura de las actividades de innovación que generan cambio."]])
X12_train,X12_test,y12_train,y12_test = train_test_split(X12, y12, test_size=0.25, random_state=0)
model= LogisticRegression()
model.fit(X12_train,y12_train)
datanew = {
           '63.- Edad': [edad],
           '66.- Personas a mi cargo (en número)': [persC],
           '62.- Sexo': [sexo],
           }
df12 = pd.DataFrame(datanew,columns=['63.- Edad', '66.- Personas a mi cargo (en número)', '62.- Sexo']) 
prediccion12=model.predict(df12)



#Pregunta 13

X13 = (dataframe[["63.- Edad", "66.- Personas a mi cargo (en número)", "62.- Sexo"]])
y13 = (dataframe[["13.- Ayudo a las personas a establecer metas innovadoras que ayuden a un cambio."]])
X13_train,X13_test,y13_train,y13_test = train_test_split(X13, y13, test_size=0.25, random_state=0)
model= LogisticRegression()
model.fit(X13_train,y13_train)
datanew = {
           '63.- Edad': [edad],
           '66.- Personas a mi cargo (en número)': [persC],
           '62.- Sexo': [sexo],
           }
df13 = pd.DataFrame(datanew,columns=['63.- Edad', '66.- Personas a mi cargo (en número)', '62.- Sexo']) 
prediccion13=model.predict(df13)




#Pregunta 14

X14 = (dataframe[["63.- Edad", "66.- Personas a mi cargo (en número)", "62.- Sexo"]])
y14 = (dataframe[["14.- Soy hábil para trabajar en equipo y resolver problemas complejos."]])
X14_train,X14_test,y14_train,y14_test = train_test_split(X14, y14, test_size=0.25, random_state=0)
model= LogisticRegression()
model.fit(X14_train,y14_train)
datanew = {
           '63.- Edad': [edad],
           '66.- Personas a mi cargo (en número)': [persC],
           '62.- Sexo': [sexo],
           }
df14 = pd.DataFrame(datanew,columns=['63.- Edad', '66.- Personas a mi cargo (en número)', '62.- Sexo']) 
prediccion14=model.predict(df14)




#Pregunta 15

X15 = (dataframe[["63.- Edad", "66.- Personas a mi cargo (en número)", "62.- Sexo"]])
y15 = (dataframe[["15.- Estoy seguro que la gente se siente tratada en forma justa y equitativa."]])
X15_train,X15_test,y15_train,y15_test = train_test_split(X15, y15, test_size=0.25, random_state=0)
model= LogisticRegression()
model.fit(X15_train,y15_train)
datanew = {
           '63.- Edad': [edad],
           '66.- Personas a mi cargo (en número)': [persC],
           '62.- Sexo': [sexo],
           }
df15 = pd.DataFrame(datanew,columns=['63.- Edad', '66.- Personas a mi cargo (en número)', '62.- Sexo']) 
prediccion15=model.predict(df15)




#Pregunta 16

X16 = (dataframe[["63.- Edad", "66.- Personas a mi cargo (en número)", "62.- Sexo"]])
y16 = (dataframe[["16.- Me gusta el trabajo en el que pueda influir en los demás."]])
X16_train,X16_test,y16_train,y16_test = train_test_split(X16, y16, test_size=0.25, random_state=0)
model= LogisticRegression()
model.fit(X16_train,y16_train)
datanew = {
           '63.- Edad': [edad],
           '66.- Personas a mi cargo (en número)': [persC],
           '62.- Sexo': [sexo],
           }
df16 = pd.DataFrame(datanew,columns=['63.- Edad', '66.- Personas a mi cargo (en número)', '62.- Sexo']) 
prediccion16=model.predict(df16)




#Pregunta 17

X17 = (dataframe[["63.- Edad", "66.- Personas a mi cargo (en número)", "62.- Sexo"]])
y17 = (dataframe[["17.- Propongo que se considere una parte del  presupuesto para actividades de innovación....."]])
X17_train,X17_test,y17_train,y17_test = train_test_split(X17, y17, test_size=0.25, random_state=0)
model= LogisticRegression()
model.fit(X17_train,y17_train)
datanew = {
           '63.- Edad': [edad],
           '66.- Personas a mi cargo (en número)': [persC],
           '62.- Sexo': [sexo],
           }
df17 = pd.DataFrame(datanew,columns=['63.- Edad', '66.- Personas a mi cargo (en número)', '62.- Sexo']) 
prediccion17=model.predict(df17)



#Pregunta 18

X18 = (dataframe[["63.- Edad", "66.- Personas a mi cargo (en número)", "62.- Sexo"]])
y18 = (dataframe[["18.- Procuro resolver problemas de motivación que influyen en la innovación."]])
X18_train,X18_test,y18_train,y18_test = train_test_split(X18, y18, test_size=0.25, random_state=0)
model= LogisticRegression()
model.fit(X18_train,y18_train)
datanew = {
           '63.- Edad': [edad],
           '66.- Personas a mi cargo (en número)': [persC],
           '62.- Sexo': [sexo],
           }
df18 = pd.DataFrame(datanew,columns=['63.- Edad', '66.- Personas a mi cargo (en número)', '62.- Sexo']) 
prediccion18=model.predict(df18)



#Pregunta 19

X19 = (dataframe[["63.- Edad", "66.- Personas a mi cargo (en número)", "62.- Sexo"]])
y19 = (dataframe[["19.- Ofrezco gestionar y/o proporcionar entrenamiento a quienes desean hacer mejoras."]])
X19_train,X19_test,y19_train,y19_test = train_test_split(X19, y19, test_size=0.25, random_state=0)
model= LogisticRegression()
model.fit(X19_train,y19_train)
datanew = {
           '63.- Edad': [edad],
           '66.- Personas a mi cargo (en número)': [persC],
           '62.- Sexo': [sexo],
           }
df19 = pd.DataFrame(datanew,columns=['63.- Edad', '66.- Personas a mi cargo (en número)', '62.- Sexo']) 
prediccion19=model.predict(df19)



#Pregunta 20

X20 = (dataframe[["63.- Edad", "66.- Personas a mi cargo (en número)", "62.- Sexo"]])
y20 = (dataframe[["20.- Soy honesto y directo al retroalimentar el desempeño innovador."]])
X20_train,X20_test,y20_train,y20_test = train_test_split(X20, y20, test_size=0.25, random_state=0)
model= LogisticRegression()
model.fit(X20_train,y20_train)
datanew = {
           '63.- Edad': [edad],
           '66.- Personas a mi cargo (en número)': [persC],
           '62.- Sexo': [sexo],
           }
df20 = pd.DataFrame(datanew,columns=['63.- Edad', '66.- Personas a mi cargo (en número)', '62.- Sexo']) 
prediccion20=model.predict(df20)


#Pregunta 21

X21 = (dataframe[["63.- Edad", "66.- Personas a mi cargo (en número)", "62.- Sexo"]])
y21 = (dataframe[["21.- Utilizo recompensas para reforzar el desempeñó en actividades innovadoras."]])
#Entrenamiento
X21_train,X21_test,y21_train,y21_test = train_test_split(X21, y21, test_size=0.25, random_state=0)
model= LogisticRegression()
model.fit(X21_train,y21_train)
datanew = {
           '63.- Edad': [edad],
           '66.- Personas a mi cargo (en número)': [persC],
           '62.- Sexo': [sexo],
           }
df21 = pd.DataFrame(datanew,columns=['63.- Edad', '66.- Personas a mi cargo (en número)', '62.- Sexo']) 
prediccion21=model.predict(df21)



#Pregunta 22

X22 = (dataframe[["63.- Edad", "66.- Personas a mi cargo (en número)", "62.- Sexo"]])
y22 = (dataframe[["22.- Combino actividades de manera que el equipo ejerza sus capacidades."]])
X22_train,X22_test,y22_train,y22_test = train_test_split(X22, y22, test_size=0.25, random_state=0)
model= LogisticRegression()
model.fit(X22_train,y22_train)
datanew = {
           '63.- Edad': [edad],
           '66.- Personas a mi cargo (en número)': [persC],
           '62.- Sexo': [sexo],
           }
df22 = pd.DataFrame(datanew,columns=['63.- Edad', '66.- Personas a mi cargo (en número)', '62.- Sexo']) 
prediccion22=model.predict(df22)



#Pregunta 23

X23 = (dataframe[["63.- Edad", "66.- Personas a mi cargo (en número)", "62.- Sexo"]])
y23 = (dataframe[["23.- Divido las cosas o situaciones para averiguar cómo funcionan."]])
X23_train,X23_test,y23_train,y23_test = train_test_split(X23, y23, test_size=0.25, random_state=0)
model= LogisticRegression()
model.fit(X23_train,y23_train)
datanew = {
           '63.- Edad': [edad],
           '66.- Personas a mi cargo (en número)': [persC],
           '62.- Sexo': [sexo],
           }
df23 = pd.DataFrame(datanew,columns=['63.- Edad', '66.- Personas a mi cargo (en número)', '62.- Sexo']) 
prediccion23=model.predict(df23)




#Pregunta 24

X24 = (dataframe[["63.- Edad", "66.- Personas a mi cargo (en número)", "62.- Sexo"]])
y24 = (dataframe[["24.- Evito sacar conclusiones acerca de las ideas propuestas por otras personas."]])
X24_train,X24_test,y24_train,y24_test = train_test_split(X24, y24, test_size=0.25, random_state=0)
model= LogisticRegression()
model.fit(X24_train,y24_train)
datanew = {
           '63.- Edad': [edad],
           '66.- Personas a mi cargo (en número)': [persC],
           '62.- Sexo': [sexo],
           }
df24 = pd.DataFrame(datanew,columns=['63.- Edad', '66.- Personas a mi cargo (en número)', '62.- Sexo']) 
prediccion24=model.predict(df24)




#Pregunta 25

X25 = (dataframe[["63.- Edad", "66.- Personas a mi cargo (en número)", "62.- Sexo"]])
y25 = (dataframe[["25.- Al resolver problemas, trabajo mejor al  analizar la situación."]])
X25_train,X25_test,y25_train,y25_test = train_test_split(X25, y25, test_size=0.25, random_state=0)
model= LogisticRegression()
model.fit(X25_train,y25_train)
datanew = {
           '63.- Edad': [edad],
           '66.- Personas a mi cargo (en número)': [persC],
           '62.- Sexo': [sexo],
           }
df25 = pd.DataFrame(datanew,columns=['63.- Edad', '66.- Personas a mi cargo (en número)', '62.- Sexo']) 
prediccion25=model.predict(df25)




#Pregunta 26

X26 = (dataframe[["63.- Edad", "66.- Personas a mi cargo (en número)", "62.- Sexo"]])
y26 = (dataframe[["26.- Me gustan las personas que se muestran seguras al realizar propuestas innovadoras ."]])
X26_train,X26_test,y26_train,y26_test = train_test_split(X26, y26, test_size=0.25, random_state=0)
model= LogisticRegression()
model.fit(X26_train,y26_train)
datanew = {
           '63.- Edad': [edad],
           '66.- Personas a mi cargo (en número)': [persC],
           '62.- Sexo': [sexo],
           }
df26 = pd.DataFrame(datanew,columns=['63.- Edad', '66.- Personas a mi cargo (en número)', '62.- Sexo']) 
prediccion26=model.predict(df26)




#Pregunta 27

X27 = (dataframe[["63.- Edad", "66.- Personas a mi cargo (en número)", "62.- Sexo"]])
y27 = (dataframe[["27.- Soy perseverante en la solución de problemas difíciles."]])
X27_train,X27_test,y27_train,y27_test = train_test_split(X27, y27, test_size=0.25, random_state=0)
model= LogisticRegression()
model.fit(X27_train,y27_train)
datanew = {
           '63.- Edad': [edad],
           '66.- Personas a mi cargo (en número)': [persC],
           '62.- Sexo': [sexo],
           }
df27 = pd.DataFrame(datanew,columns=['63.- Edad', '66.- Personas a mi cargo (en número)', '62.- Sexo']) 
prediccion27=model.predict(df27)



#Pregunta 28

X28 = (dataframe[["63.- Edad", "66.- Personas a mi cargo (en número)", "62.- Sexo"]])
y28 = (dataframe[["28.- Estoy dispuesto para averiguar un nuevo uso de métodos o equipo existentes."]])
X28_train,X28_test,y28_train,y28_test = train_test_split(X28, y28, test_size=0.25, random_state=0)
model= LogisticRegression()
model.fit(X28_train,y28_train)
datanew = {
           '63.- Edad': [edad],
           '66.- Personas a mi cargo (en número)': [persC],
           '62.- Sexo': [sexo],
           }
df28 = pd.DataFrame(datanew,columns=['63.- Edad', '66.- Personas a mi cargo (en número)', '62.- Sexo']) 
prediccion28=model.predict(df28)



#Pregunta 29

X29 = (dataframe[["63.- Edad", "66.- Personas a mi cargo (en número)", "62.- Sexo"]])
y29 = (dataframe[["29.- Me gusta ser el primero en probar una idea o un método nuevo."]])
X29_train,X29_test,y29_train,y29_test = train_test_split(X29, y29, test_size=0.25, random_state=0)
model= LogisticRegression()
model.fit(X29_train,y29_train)
datanew = {
           '63.- Edad': [edad],
           '66.- Personas a mi cargo (en número)': [persC],
           '62.- Sexo': [sexo],
           }
df29 = pd.DataFrame(datanew,columns=['63.- Edad', '66.- Personas a mi cargo (en número)', '62.- Sexo']) 
prediccion29=model.predict(df29)



#Pregunta 10

X30 = (dataframe[["63.- Edad", "66.- Personas a mi cargo (en número)", "62.- Sexo"]])
y30 = (dataframe[["30.- Trabajo con empeño en  solucionar un problema que ha causado a otros gran dificultad."]])
X30_train,X30_test,y30_train,y30_test = train_test_split(X30, y30, test_size=0.25, random_state=0)
model= LogisticRegression()
model.fit(X30_train,y30_train)
datanew = {
           '63.- Edad': [edad],
           '66.- Personas a mi cargo (en número)': [persC],
           '62.- Sexo': [sexo],
           }
df30 = pd.DataFrame(datanew,columns=['63.- Edad', '66.- Personas a mi cargo (en número)', '62.- Sexo']) 
prediccion30=model.predict(df30)


#31.- Pienso que no siempre los métodos lógicos, paso a paso, son los mejores para solucionar problemas.",	"32.- Realizo propuestas para el desarrollado productos mejorado.",	"33.- Me apoyo de otras áreas para desarrollar una idea innovadora.",	"34.- Observo las habilidades en otras áreas como punto de apoyo para innovar.", "36.- Aprovecho la oportunidad de incorporar ideas de otras disciplinas a mi trabajo.",	"37.- Si se pregunta a mis compañeros o colegas, dirán que soy ingenioso.",	"38.- Aplico las técnicas  y herramientas de Innovación.",	"39.- A veces me vuelvo demasiado entusiasta.",	"40.- Confío en mis “corazonadas” cuando busco la solución de un problema.",	"41.- A veces me divierto al romper las reglas con tal de innovar.",	"42.- Realizo actividades de Investigación y Desarrollo (I+D).",	"43.- 2. Mido el impacto de las innovaciones propuestas innovadoras",	"44.- Considero a las áreas de la empresa como fuentes de información para la I+D.",	"45.- Propongo cambios en los productos.",	"46.- Se sienten orgullosos de mi trabajo.",	"47.- Cuento con su respeto.",	"48.- Tienen plena confianza en mi.",	"49.- Confían en mi capacidad para superar cualquier obstáculo.",	"50.- Pongo especial énfasis en la resolución cuidadosa de los problemas antes de actuar.",	"51.- Hago que se basen en el razonamiento y en la evidencia para resolver problemas.",	"52.- Trato de que vean los problemas como una oportunidad para aprender.",	"53.- Les hago pensar sobre viejos problemas de forma nueva.",	"54.- Les pido que fundamente sus opiniones con argumentos sólidos.",	"55.- Les doy nuevas formas de enfocar los problemas que antes les resultaban desconcertantes.",	"56.- Les doy charlas para motivarlos.",	"57.- Potencio su motivación con éxito.",	"58.- Trato de desarrollar nuevas formas para motivarlos.",	"59.- Me preocupo de formar a aquellos que lo necesitan.",	"60.- Centro mi atención en los casos en lo que no se consigue alcanzar las metas esperadas."]).axis()



#Print final
print("Banco de preguntas", '\n', "1.- Mis estrategias son para gestionar proyectos de innovación.",  '\n',"2.- Como parte de la cultura organizacional para mí la creatividad es muy importante.",  '\n',"3.- Propongo innovaciones en la organización.", '\n',	"4.- Me doy tiempo para trabajar en mis propias ideas o proyectos.",  '\n',"5.- He propuesto y/o desarrollado productos nuevos en las áreas donde me he desempeñado.", '\n',	"6.- Discuto abiertamente con mis compañeros acerca de cumplir con los objetivos y metas de la empresa.", '\n',	"7.- Mantengo un diálogo abierto con aquellos que no están de acuerdo conmigo.", '\n',	"8.- Me expreso abiertamente en las reuniones.", '\n',	"9.- Doy información importante respecto a nuevas soluciones al trabajar en innovación.", '\n',	"10.- Preparo a la organización para los cambios.", '\n',	"11.- Demuestro originalidad en mi trabajo.", '\n',	"12.- Conozco la magnitud y la estructura de las actividades de innovación que generan cambio.",	 '\n',"13.- Ayudo a las personas a establecer metas innovadoras que ayuden a un cambio.", '\n',	"14.- Soy hábil para trabajar en equipo y resolver problemas complejos.", '\n',	"15.- Estoy seguro que la gente se siente tratada en forma justa y equitativa.",	 '\n',"16.- Me gusta el trabajo en el que pueda influir en los demás.", '\n',	"17.- Propongo que se considere una parte del  presupuesto para actividades de innovación.....",	 '\n',"18.- Procuro resolver problemas de motivación que influyen en la innovación.", '\n',	"19.- Ofrezco gestionar y/o proporcionar entrenamiento a quienes desean hacer mejoras.", '\n',	"20.- Soy honesto y directo al retroalimentar el desempeño innovador.", '\n',	"21.- Utilizo recompensas para reforzar el desempeñó en actividades innovadoras.",	 '\n',"22.- Combino actividades de manera que el equipo ejerza sus capacidades.", '\n',	"23.- Divido las cosas o situaciones para averiguar cómo funcionan.", '\n',	"24.- Evito sacar conclusiones acerca de las ideas propuestas por otras personas.", '\n',	"25.- Al resolver problemas, trabajo mejor al  analizar la situación.", '\n',"26.- Me gustan las personas que se muestran seguras al realizar propuestas innovadoras .",	 '\n',"27.- Soy perseverante en la solución de problemas difíciles.", '\n',	"28.- Estoy dispuesto para averiguar un nuevo uso de métodos o equipo existentes.",	 '\n',"29.- Me gusta ser el primero en probar una idea o un método nuevo.",'\n',	"30.- Trabajo con empeño en  solucionar un problema que ha causado a otros gran dificultad.", '\n')
print (df1, '\n'*2)
print('La prediccion en orden de preguntas es:', '\n') 
print('Recuerda que la letra A es la mas fuerte y la letra E es la mas debil', '\n')
print (prediccion1, prediccion2, prediccion3, prediccion4, prediccion5, prediccion6, prediccion7, prediccion8, prediccion9, prediccion10,"\n", prediccion11, prediccion12, prediccion13, prediccion14, prediccion15, prediccion16, prediccion17, prediccion18, prediccion19, prediccion20, "\n", prediccion21, prediccion22, prediccion23, prediccion24, prediccion25, prediccion26, prediccion27, prediccion28, prediccion29, prediccion30,'\n'*2)
graf = plt.bar(dataframe['29.- Me gusta ser el primero en probar una idea o un método nuevo.'], dataframe['63.- Edad'])
print(graf)