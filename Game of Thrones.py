# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 23:07:16 2019

@author: faisa
"""
#Importing Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf # regression modeling
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split # train/test split
from sklearn.neighbors import KNeighborsRegressor # KNN for Regression
import statsmodels.formula.api as smf # regression modeling
import sklearn.metrics # more metrics for model performance evaluation
from sklearn.model_selection import cross_val_score # k-folds cross validation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler # standard scaler
from sklearn.ensemble import RandomForestClassifier

#Uploading FIle
file = 'GOT_character_predictions.xlsx'
df= pd.read_excel(file)

#Fixing Columns
df=df.rename(index=str, columns={"S.No": "S_No"})
df['house'] = df.house.str.title()
df['spouse'] = df.spouse.str.title()
#Flagging Missing Values
for col in df:
    if df[col].isnull().any():
        df['m_'+col] = df[col].isnull().astype(int)

##Fixing repeating houses and culteres in the dataset

house1 = {
    'House Baratheon': ["House Baratheon of King's Landing", 'House Baratheon Of Dragonstone'],
    'House Lannister': ['House Lannister of Casterly Rock'],
    'House Martell': ['House Nymeros Martell'],
    }

cult = {
    'Summer Islands': ['summer islands', 'summer islander', 'summer isles'],
    'Ghiscari': ['ghiscari', 'ghiscaricari',  'ghis'],
    'Asshai': ["asshai'i", 'asshai'],
    'Lysene': ['lysene', 'lyseni'],
    'Andal': ['andal', 'andals'],
    'Braavosi': ['braavosi', 'braavos'],
    'Dornish': ['dornishmen', 'dorne', 'dornish'],
    'Myrish': ['myr', 'myrish', 'myrmen'],
    'Westermen': ['westermen', 'westerman', 'westerlands'],
    'Westerosi': ['westeros', 'westerosi'],
    'Stormlander': ['stormlands', 'stormlander'],
    'Norvoshi': ['norvos', 'norvoshi'],
    'Northmen': ['the north', 'northmen'],
    'Free Folk': ['wildling', 'first men', 'free folk'],
    'Qartheen': ['qartheen', 'qarth'],
    'Reach': ['the reach', 'reach', 'reachmen'],
}
##Title fixes
def get_cult(value):
    value = value.lower()
    v = [k for (k, v) in cult.items() if value in v]
    return v[0] if len(v) > 0 else value.title()

def get_house(value):
    value = value.lower()
    v = [k for (k, v) in house1.items() if value in v]
    return v[0] if len(v) > 0 else value.title()


df.loc[:, "culture"] = [get_cult(x) for x in df.culture.fillna("NaN")]
df.loc[:, "house"] = [get_house(x) for x in df.house.fillna("NaN")]

#Dealing with missing values
fill_categorical = 'NaN'
df['title'] = df['title'].fillna(fill_categorical)
df['culture'] = df['culture'].fillna(fill_categorical)
df['house'] = df['house'].fillna(fill_categorical)
df['father'] = df['father'].fillna(fill_categorical)
df['mother'] = df['mother'].fillna(fill_categorical)
df['heir'] = df['heir'].fillna(fill_categorical)
df['spouse'] = df['spouse'].fillna(fill_categorical)
df['title'] = df['title'].fillna(fill_categorical)
df['age'] = df['age'].fillna(df.age.mean())
df['dateOfBirth'] = df['dateOfBirth'].fillna(-1)
df['male'] = df['male'].fillna(-1)
df['isAliveMother'] = df['isAliveMother'].fillna(-1)
df['isAliveFather'] = df['isAliveFather'].fillna(-1)
df['isAliveHeir'] = df['isAliveHeir'].fillna(-1)
df['isAliveSpouse'] = df['isAliveSpouse'].fillna(-1)



#Creating new Features
"""Creating a new feature that assigns status based on whether
the character had a title"""
def Status(c):
  if c['title'] == 'NaN' :
    return 'low_born'
  else:
    return 'High_born'
df['Status'] = df.apply(Status, axis=1)

"""Creating a new feathure that counts the number fo books
a character appears in"""

df.loc[:, "no_of_books"] = df[[x for x in df.columns if x.startswith("book")]].sum(axis = 1)


#Looking at the relationship of deadrelations with house and culture
violent_culture=df.groupby('culture')['numDeadRelations'].sum().sort_values()

violent_house=df.groupby('house')['numDeadRelations'].sum().sort_values()


"""Creating a feature that distingushes the cultures with the 
highest number of dead relatives"""

df['culture_violence'] = pd.np.where(df.culture.str.contains("Valarian"), "high",
                   pd.np.where(df.culture.str.contains("Northmen"), "high",
                   pd.np.where(df.culture.str.contains("Riverman"), "high",
                   pd.np.where(df.culture.str.contains("Dothraki"), "high",
                   pd.np.where(df.culture.str.contains("Dornish"), "high",
                   pd.np.where(df.culture.str.contains("Westermen"), "high",
                   pd.np.where(df.culture.str.contains("Valemen"), "high",                               
                   pd.np.where(df.culture.str.contains("Ironborn"), "high", "not_violent"))))))))

"""Creating a feature that distingushes the Great houses(Internet research) from
the others"""

def Tar_house(c):
        
  if c['house'] == 'House Targaryen':
    return 'Great_house'
  else:
    return 'Not_Great'

def Arr_house(c):
        
  if c['house'] == 'House Arryn':
    return 'Great_house'
  else:
    return 'Not_Great'


def grey_house(c):
        
  if c['house'] == 'House Greyjoy':
    return 'Great_house'
  else:
    return 'Not_Great'

def Bar_house(c):
        
  if c['house'] == 'House Baratheon':
    return 'Great_house'
  else:
    return 'Not_Great'

def Tyl_house(c):
        
  if c['house'] == 'House Tyrell':
    return 'Great_house'
  else:
    return 'Not_Great'

def Fry_house(c):
        
  if c['house'] == 'House Frey':
    return 'Great_house'
  else:
    return 'Not_Great'

def Bol_house(c):
        
  if c['house'] == 'House Bolton':
    return 'Great_house'
  else:
    return 'Not_Great'

def Mar_house(c):
    
  if c['house'] == 'House Martell':
    return 'Great_house'
  else:
    return 'Not_Great'


def Lan_house(c):
        
  if c['house'] == 'House Lannister':
    return 'Great_house'
  else:
    return 'Not_Great'

df['Great_house'] = df.apply(Tar_house, axis=1)
df['Great_house'] = df.apply(Lan_house, axis=1)
df['Great_house'] = df.apply(Mar_house, axis=1)
df['Great_house'] = df.apply(Bol_house, axis=1)
df['Great_house'] = df.apply(Fry_house, axis=1)
df['Great_house'] = df.apply(Tyl_house, axis=1)
df['Great_house'] = df.apply(grey_house, axis=1)
df['Great_house'] = df.apply(Arr_house, axis=1)
df['Great_house'] = df.apply(Bar_house, axis=1)

#looking at the popularity range
df.popularity.quantile([0.05,0.1,0.25,0.5,0.65,0.8,1])

"""Creating a new feature that divides the charcter from high-low-meduim
populariy"""

def popularity_level(c):
  if c['popularity'] < 0.014:
    return 'un_pop'
  elif c['popularity'] < 0.6:
    return 'med_pop'
  else:
    return 'pop'
df['popularity_level'] = df.apply(popularity_level, axis=1)


#Factorizing the non numeric columns to faciliate ML alogorthms
df.loc[:, "title"] = pd.factorize(df.title)[0]
df.loc[:, "culture"] = pd.factorize(df.culture)[0]
df.loc[:, "mother"] = pd.factorize(df.mother)[0]
df.loc[:, "father"] = pd.factorize(df.father)[0]
df.loc[:, "heir"] = pd.factorize(df.heir)[0]
df.loc[:, "house"] = pd.factorize(df.house)[0]
df.loc[:, "spouse"] = pd.factorize(df.spouse)[0]
df.loc[:, "Great_house"] = pd.factorize(df.Great_house)[0]
df.loc[:, "culture_violence"] = pd.factorize(df.culture_violence)[0]
df.loc[:, "popularity_level"] = pd.factorize(df.culture_violence)[0]
df.loc[:, "Status"] = pd.factorize(df.Status)[0]
df.loc[:, "age"] = pd.factorize(df.age)[0]
df.loc[:, "name"] = pd.factorize(df.name)[0]


#Building a simple logistice regression as a basemodel
mod = smf.logit(formula="""isAlive ~ 
 S_No  +  name  +  title  +  male  +  culture  +  dateOfBirth  +  mother  +
        father  +  heir  +  house  +  spouse  +  book1_A_Game_Of_Thrones  +
        book2_A_Clash_Of_Kings  +  book3_A_Storm_Of_Swords  +
        book4_A_Feast_For_Crows  +  book5_A_Dance_with_Dragons  +
        isAliveMother  +  isAliveFather  +  isAliveHeir  +  isAliveSpouse  +
        isMarried  +  isNoble  +  age  +  numDeadRelations  +  popularity  +  m_title  +  m_culture  +  m_dateOfBirth  +  m_mother  +
        m_father  +  m_heir  +  m_house  +  m_spouse  +  m_isAliveMother  +
        m_isAliveFather  +  m_isAliveHeir  +  m_isAliveSpouse  +  m_age  +
        Status  +  no_of_books  +  culture_violence  +  Great_house +popularity_level
    """, data=df)
res = mod.fit()
print(res.summary())

#Selecting relevant features and creating target and deature data set
df3=df.copy()
ml_data1=df3.drop([ "name", "S_No","title","dateOfBirth","isAlive","isAliveFather",
                   "isAliveMother","m_father","m_heir",
                   "Great_house","m_isAliveHeir",
                   "m_isAliveHeir","m_isAliveFather","m_isAliveMother",
                   "m_isAliveSpouse","m_father",
                   "mother", "father","heir",'isAliveHeir'], 1, inplace = True)
ml_target = df.loc[:, 'isAlive']
    
ml_data1=df3

#Standardizing the feature scales
scaler = StandardScaler()


# Fitting the scaler with our data
scaler.fit(ml_data1)


# Transforming our data after fit
X_scaled = scaler.transform(ml_data1)


# Putting our scaled data into a DataFrame
X_scaled_df = pd.DataFrame(ml_data1)




#Splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
            X_scaled_df,
            ml_target,
            test_size = 0.1,
            random_state = 508,
            stratify = ml_target)



#Applying the best RF model after tuning the hyperparemeters
RF = RandomForestClassifier(bootstrap = False,
                                    criterion = 'entropy',
                                    min_samples_leaf = 5,
                                    n_estimators = 100,
                                    warm_start = True)



RF.fit(X_train, y_train)


RF_pred =  RF.predict(X_test)


print('Training Score',  RF.score(X_train, y_train).round(4))
print('Testing Score:',  RF.score(X_test, y_test).round(4))


RF_train =  RF.score(X_train, y_train)
RF_test  =  RF.score(X_test, y_test)

cv_rf_3 = cross_val_score( RF, X_scaled_df, ml_target, cv = 3)

print(pd.np.mean(cv_rf_3))

###################


########################
# Saving Results
########################



# Saving model predictions

model_predictions_df = pd.DataFrame({'Actual' : y_test,
                                     'RF_Predicted': RF_pred,})


model_predictions_df.to_excel("RF_Model_Predictions.xlsx")



