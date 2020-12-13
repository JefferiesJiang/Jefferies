import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib
# I loaded the csv file.
SAT_data = pd.read_csv('Program_Files_(x86)/Sublime_Text_3/SATMODEL/SAT__College_Board__2010_School_Level_Results.csv')

SAT_data.shape

SAT_data.describe()

dataset.plot(x='Number of test takers', y='Mathematics Mean', style='o')  
plt.title('MinTemp vs MaxTemp')  
plt.xlabel('Number of test takers')  
plt.ylabel("Score")  
plt.show()
# I reshaped the model.
x  = dataset['Number of test takers'].values.reshape(-1,1)
y = dataset['Mathematics Mean'].values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# I trained created the model.
SAT_model = LinearRegression()  
# I trained the model.
SAT_model.fit(X_train, y_train





