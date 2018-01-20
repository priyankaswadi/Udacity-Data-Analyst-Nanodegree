#!/usr/bin/python
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
########################
from collections import defaultdict
import json
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier
########################
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
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

### Task 2: Remove outliers
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

### Task 4 and 5: Try a varity of classifiers and tune your classifier to achieve better than .3 precision and recall 


scaler = MinMaxScaler()
skb = SelectKBest()
gnb = GaussianNB()
rfc = RandomForestClassifier(random_state=23)
svm  = svm.SVC(kernel="rbf")
dtc = DecisionTreeClassifier()

#Pipe for gaussiannb
#pipe =  Pipeline(steps=[('scaling',scaler),("SKB", skb), ("NaiveBayes", GaussianNB())])
#parameters = {'SKB__k': range(2,21)}

#Pipe for Random Forest
#pipe =  Pipeline(steps=[('scaling',scaler),("SKB", skb), ("RFC", rfc)])
#parameters = {'SKB__k': range(2,10),'RFC__n_estimators': [2,3,5,10], "RFC__criterion": ('gini', 'entropy'),\
#             'RFC__min_samples_leaf':[1,2,3,4,5],'RFC__min_samples_split':[2,3,4,5]}

#Pipe for SVM
#pipe =  Pipeline(steps=[('scaling',scaler),("SKB", skb), ("SVM",svm)])
#parameters = {'SKB__k': range(2,21),'SVM__C':[0.1,1,10,100,1000,10000],'SVM__gamma':[0.001,0.0001]}

#Pipe for Decision Tree
pipe =  Pipeline(steps=[("SKB", skb), ("DTC",dtc)])
parameters = {'SKB__k': range(2,10),'DTC__criterion':['gini', 'entropy'],'DTC__max_depth':[None, 1, 2, 3, 4],'DTC__class_weight':[None, 'balanced']}
#For cross validation in gridsearchcv
sss = StratifiedShuffleSplit(labels, 100, test_size=0.3, random_state=40)

gs = GridSearchCV(pipe, param_grid = parameters, cv = sss, scoring = 'f1', verbose=1000)
gs.fit(features, labels)
clf = gs.best_estimator_
pred = clf.predict(features)
print "*******************************"
print clf
print gs.best_params_
features_selected=[features_list[i+1] for i in clf.named_steps['SKB'].get_support(indices=True)]
scores = clf.named_steps['SKB'].scores_
#Due to specific nature of statement , comment line below for other algorithms except DTC
importances = clf.named_steps['DTC'].feature_importances_
print features_selected
print scores
print gs.best_score_
#Comment line below for other algorithms except DTC
print importances
print 'precision = ', precision_score(labels,pred)
print 'recall = ', recall_score(labels,pred)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)