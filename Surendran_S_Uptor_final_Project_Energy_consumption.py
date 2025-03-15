"""Import all required Libraries"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""1. Data collection"""
data = pd.read_csv("Surendran_S_Uptor_final_Project_energy_consumption_data.csv")      #read the csv file
pd.set_option("display.max_columns",None)              #set all columns to show
df = pd.DataFrame(data)                                #Conversion into DataFrame

"""2. Exploratory Data Analysis"""
"""
print(df.head())                      #Top five datas in dataset 
print(df.columns)                   #read columns names
print(df.info())                      #information of dataset to find missing values 
print(df.describe().T)                #shows the numerical columns for analysis 
print(df.isnull().sum())              #find null values count for preprocessing
print(df.duplicated().sum())          #To find duplicated values count for preprocessing 
print(df.nunique())                   #To find unique values in columns for preprocessing 
"""
"""3. Visualization of data"""

# corr = df.corr()                                           #correlation_analysis
# plt.figure(figsize=(10, 8))                                #define plot size
# sns.heatmap(corr,annot=True,cmap="magma",linewidths=0.5)   #correlation heatmap
# plt.title("Correlation Matrix")                            #title of the heatmap
# plt.show()                                                 # shows of the heatmap

""" 4. Supervised_Learning (Import all required Libraries)"""

from sklearn.preprocessing import LabelEncoder,StandardScaler,OneHotEncoder,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score

df.columns = [col.lower() for col in df.columns]          #all columns changed into lower case

x = df.drop("energy_consumption",axis=1)           #assign the all columns except energy consumption in x axis
y = df["energy_consumption"]                             #assign the energy consumption in Y axis

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)   #model_selection

model = LinearRegression()                      #Linear Regression model
model.fit(x_train,y_train)                      #fit the model into Linear Regression model

y_pred = model.predict(x_test)                  #Predict the model
# print(y_pred)                                   #print predicted value

model_accuracy = r2_score(y_test,y_pred)        #Accuracy measurement
print(model_accuracy)                           #print accuracy

tree = DecisionTreeRegressor(max_depth=5)       #Decision Tree Regression model
tree.fit(x_train, y_train)                      #fit the model into Decision Tree Regression model

y_pred_1 = tree.predict(x_test)                 #Predict the model
# print(y_pred_1)                                 #Print the model

model_accuracy = r2_score(y_test,y_pred)        #Accuracy measurement
print(model_accuracy)                           #print accuracy

""" 5. Unsupervised_Learning (Import all required Libraries)"""

from sklearn.cluster import KMeans

df.drop(df.head(49000).index,inplace = True)            #select 500 rows only by drop rows
print(df.info())

df.drop(["day","building_type","employee_count","holiday"],axis='columns',inplace=True) #select required columns
print(df.info())
print(df.describe())

X = df.iloc[:,[1,3]].values                            #select columns for clustering
print(X)

Kmeans_value = KMeans(n_clusters=5,init='k-means++',random_state=42)               #Clustering_Model
y_Kmeans = Kmeans_value.fit_predict(X)                                             #data fit into the Model

plt.scatter(X[:,0],X[:,1],c=y_Kmeans, cmap="RdBu", marker="*",label = "Kmeans", alpha=1.0)      #scatter plotting
plt.title("Clustering of Dataset")
plt.xlabel("Humidity")
plt.ylabel("Surendran_S_Uptor_final_Project_Energy_Consumption")
plt.grid()
plt.legend()
plt.show()





