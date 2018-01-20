#!/usr/bin/python
# Code to plot outliers plots
import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat
### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/my_dataset.pkl", "r") )
features = ["salary", "bonus","poi"]
#Remove outliers before plotting
data_dict.pop("TOTAL", 0 )
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0 )
data = featureFormat(data_dict, features)
salary = []
bonus = []
poi = []
data=data.tolist()
data.sort()
highest_salary = data[-2][0]

for k,v in data_dict.items():
    if v['salary']==highest_salary:
        print k
#Divide the data points between POI and non POI to plot in different colors
salary_poi = [data[ii][0] for ii in range(0, len(data)) if data[ii][2]==1]
bonus_poi =  [data[ii][1] for ii in range(0, len(data)) if data[ii][2]==1]
salary_non_poi = [data[ii][0] for ii in range(0, len(data)) if data[ii][2]==0]
bonus_non_poi =  [data[ii][1] for ii in range(0, len(data)) if data[ii][2]==0]
    
matplotlib.pyplot.scatter(salary_poi, bonus_poi, color = "r", label="poi")
matplotlib.pyplot.scatter(salary_non_poi,bonus_non_poi, color = "b", label="non_poi")
matplotlib.pyplot.legend()
matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.title("After removing outlier")
matplotlib.pyplot.show()
#

