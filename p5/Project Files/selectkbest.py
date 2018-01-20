#!/usr/bin/python
# Code to plot scoring metrics for SelectkBest for each value of k.
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
########################
from collections import defaultdict
import json
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from matplotlib import lines
########################
#features_all is used to calculate missing values
features_all = ['poi',
                'salary', 
                'deferral_payments', 
                'total_payments', 
                'loan_advances', 
                'bonus', 
                'restricted_stock_deferred', 
                'deferred_income', 
                'total_stock_value', 
                'expenses', 
                'exercised_stock_options', 
                'other', 
                'long_term_incentive', 
                'restricted_stock',
                'director_fees',
                'to_messages',
                #'email_address', 
                'from_poi_to_this_person', 
                'from_messages',
                'from_this_person_to_poi', 
                'shared_receipt_with_poi'
                ]

features_list = ['poi',
                'salary', 
                'deferral_payments', 
                'total_payments', 
                'loan_advances', 
                'bonus', 
                'restricted_stock_deferred', 
                'deferred_income', 
                'total_stock_value', 
                'expenses', 
                'exercised_stock_options', 
                'other', 
                'long_term_incentive', 
                'restricted_stock',
                'director_fees',
                'to_messages',
                #'email_address', 
                'from_poi_to_this_person', 
                'from_messages',
                'from_this_person_to_poi', 
                'shared_receipt_with_poi',
                'fraction_from_poi',
                'fraction_to_poi']
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Remove outliers
data_dict.pop("TOTAL", 0 )
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0 )
#Find missing values
nan_features = defaultdict(int)
for p in data_dict:
    for feature in features_all:
        if data_dict[p][feature] == "NaN":
            nan_features[feature] = nan_features[feature] + 1
print json.dumps(nan_features,indent=1)

### Task 3: Create new feature(s)
def computeFraction( poi_messages, all_messages ):
    fraction = 0.
    if poi_messages == "NaN":
        return 0;
    if all_messages == "NaN":
        return 0
    fraction = poi_messages*1.0/all_messages
    return fraction

submit_dict = {}
for name in data_dict:

    data_point = data_dict[name]
   
    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
    
    data_point["fraction_from_poi"] = fraction_from_poi

    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )

    submit_dict[name]={"from_poi_to_this_person":fraction_from_poi,
                       "from_this_person_to_poi":fraction_to_poi}
    data_point["fraction_to_poi"] = fraction_to_poi

 
### Store to my_dataset for easy export below.
my_dataset = data_dict
#print my_dataset
print "Number of entries",len(my_dataset)
count_poi = 0
for p in my_dataset:
    if my_dataset[p]["poi"] == True:
        count_poi=count_poi + 1
print "Number of Persons of Interest",count_poi

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#1. GaussianNB
gnb = None;
svc = None;
rfc = None;
dtc = None;

scaler = MinMaxScaler()
skb = SelectKBest()


#******************************
#STEP 1: Choose the algorithm you want to the plot for
#******************************
#Pipe_for_gaussiannb
#gnb = GaussianNB()
#pipe =  Pipeline(steps=[('scaling',scaler),("SKB", skb), ("NaiveBayes", GaussianNB())])
#parameters = {'SKB__k': range(2,21)}

#Pipe for Random Forest
#rfc = RandomForestClassifier(random_state=23)
#pipe =  Pipeline(steps=[('scaling',scaler),("SKB", skb), ("RFC", rfc)])
#parameters = {'SKB__k': range(2,10),'RFC__n_estimators': [2,3,5,10], "RFC__criterion": ('gini', 'entropy'),\
#      'RFC__min_samples_leaf':[1,2,3,4,5],'RFC__min_samples_split':[2,3,4,5]}

#Pipe for SVM
#svc  = svm.SVC(kernel="rbf")
#pipe =  Pipeline(steps=[('scaling',scaler),("SKB", skb), ("SVM",svc)])
#parameters = {'SKB__k': range(2,21),'SVM__C':[0.1,1,10,100,1000,10000],'SVM__gamma':[0.001,0.0001]}

#Pipe for Decision Tree
dtc = DecisionTreeClassifier()
pipe =  Pipeline(steps=[("SKB", skb), ("DTC",dtc)])
parameters = {'SKB__k': range(2,21),'DTC__criterion':['gini', 'entropy'],'DTC__max_depth':[None, 1, 2, 3, 4],'DTC__class_weight':[None, 'balanced']}

#******************************
#STEP 2: Run the scoring methods (precision,recall,f1score)
#******************************
sss = StratifiedShuffleSplit(labels, 100, test_size=0.3, random_state=40)
# Precision
gs_p = GridSearchCV(pipe, param_grid = parameters, cv = sss, scoring = 'precision')
#Recall
gs_r = GridSearchCV(pipe, param_grid = parameters, cv = sss, scoring = 'recall')
#F1 Score
gs_f1 = GridSearchCV(pipe, param_grid = parameters, cv = sss, scoring = 'f1')

# Compute list to plot precision scores for each value of k
gs_p.fit(features, labels)
grid_scores = gs_p.grid_scores_
precision_values = []
k_values = []

#Create a list of lists
precision_list = defaultdict(list)
for score in grid_scores: #Algorithm selected
    if gnb != None:
        precision_list[score[0]['SKB__k']].append(score[1])
    if svc != None:
        if (score[0]['SVM__gamma'] == 0.001) and (score[0]['SVM__C'] == 10000):
            precision_list[score[0]['SKB__k']].append(score[1])
    if dtc != None:
         if (score[0]['DTC__criterion'] == 'gini') and (score[0]['DTC__max_depth'] == 2) and (score[0]['DTC__class_weight'] == 'balanced'):
            precision_list[score[0]['SKB__k']].append(score[1])
    if rfc != None:
         if (score[0]['RFC__criterion'] == 'gini') and (score[0]['RFC__n_estimators'] == 3) and (score[0]['RFC__min_samples_leaf'] == 1) and (score[0]['RFC__min_samples_split'] == 2):
            precision_list[score[0]['SKB__k']].append(score[1])
        
#Convert dict to list and append values
for val in precision_list:
    k_values.append(val)
    precision_values.append(precision_list[val])
print "Precision:"    
print gs_p.grid_scores_
print k_values,precision_values
print gs_p.best_score_
#Recall
gs_r.fit(features, labels)
grid_scores = gs_r.grid_scores_

recall_values = []
recall_list = defaultdict(list)
for score in grid_scores:#Algorithm selected
    if gnb != None:
        recall_list[score[0]['SKB__k']].append(score[1])
    if svc != None:
        if (score[0]['SVM__gamma'] == 0.001) and (score[0]['SVM__C'] == 10000):
            recall_list[score[0]['SKB__k']].append(score[1])
    if dtc != None:
         if (score[0]['DTC__criterion'] == 'gini') and (score[0]['DTC__max_depth'] == 2) and (score[0]['DTC__class_weight'] == 'balanced'):
            recall_list[score[0]['SKB__k']].append(score[1])
    if rfc != None:
         if (score[0]['RFC__criterion'] == 'gini') and (score[0]['RFC__n_estimators'] == 3) and (score[0]['RFC__min_samples_leaf'] == 1) and (score[0]['RFC__min_samples_split'] == 2):
            recall_list[score[0]['SKB__k']].append(score[1])
#Convert dict to list and append values    
for val in recall_list: 
    recall_values.append(recall_list[val])
print "Recall:"   
print gs_r.grid_scores_
print k_values,recall_values
print gs_r.best_score_
##F1 Score
gs_f1.fit(features, labels)
grid_scores = gs_f1.grid_scores_
f1_values = []
f1_list = defaultdict(list)
for score in grid_scores:#Algorithm selected
    if gnb != None:
        f1_list[score[0]['SKB__k']].append(score[1])
    if svc != None:
        if (score[0]['SVM__gamma'] == 0.001) and (score[0]['SVM__C'] == 10000):
            f1_list[score[0]['SKB__k']].append(score[1])
    if dtc != None:
         if (score[0]['DTC__criterion'] == 'gini') and (score[0]['DTC__max_depth'] == 2) and (score[0]['DTC__class_weight'] == 'balanced'):
            f1_list[score[0]['SKB__k']].append(score[1])
    if rfc != None:
         if (score[0]['RFC__criterion'] == 'gini') and (score[0]['RFC__n_estimators'] == 3) and (score[0]['RFC__min_samples_leaf'] == 1) and (score[0]['RFC__min_samples_split'] == 2):
            f1_list[score[0]['SKB__k']].append(score[1])
#Convert dict to list and append values    
for val in f1_list:    
    f1_values.append(f1_list[val])
print "F1 Score:"      
print gs_f1.grid_scores_
print k_values,f1_values
print gs_f1.best_score_
#
#Plot for k value vs score
plt.plot(k_values, precision_values, 'b', label='precision')
plt.plot(k_values, recall_values, 'g', label='recall')
plt.plot(k_values, f1_values, 'r', linestyle = '--', label='f1 score')
legend = plt.legend(loc='upper left')
plt.xlabel('Number of Features (k)')
plt.ylabel('Score')
plt.title('Scoring metrics for Random Forest \n (criterion=gini, n_estimators = 3,min_samples_leaf = 1\n min_samples_split = 2)')
plt.show()