#!/usr/bin/python
#Plotting the new features fraction_from_poi and fraction_to_poi
import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/my_dataset.pkl", "r") )
features = ["fraction_to_poi","fraction_from_poi","poi"]
data = featureFormat(data_dict, features)
salary = []
bonus = []
poi = []
data=data.tolist()
data.sort()

#Divide the data points between POI and non POI to plot in different colors

fraction_to_poi_poi = [data[ii][0] for ii in range(0, len(data)) if data[ii][2]==1]
fraction_from_poi_poi =  [data[ii][1] for ii in range(0, len(data)) if data[ii][2]==1]
fraction_to_poi_non_poi = [data[ii][0] for ii in range(0, len(data)) if data[ii][2]==0]
fraction_from_poi_non_poi =  [data[ii][1] for ii in range(0, len(data)) if data[ii][2]==0]
    
matplotlib.pyplot.scatter(fraction_to_poi_poi, fraction_from_poi_poi, color = "r", label="poi")
matplotlib.pyplot.scatter(fraction_to_poi_non_poi,fraction_from_poi_non_poi , color = "b", label="non_poi")
matplotlib.pyplot.legend()

matplotlib.pyplot.xlabel("Fraction of emails from this person to POI")
matplotlib.pyplot.ylabel("Fraction of emails to this person from POI")
matplotlib.pyplot.title("Proportion of emails to/from POI")
matplotlib.pyplot.show()
#

