#Audit Cities data
import xml.etree.cElementTree as ET
from collections import defaultdict
import re
import pprint

osmfile = "albany.osm"

mapping_city = {"Schenectary":"Schenectady",
           "Rexrford":"Rexford",
           "Troy, NY":"Troy",
           "Slingerlands, NY":"Slingerlands",
           "clifton Park":"Clifton Park",
           "clifton Park":"Clifton Park"
          }
def is_city(elem):    
    return (elem.attrib['k'] == "addr:city")

def audit_city(cities, city):
    #create a list of cities
    cities[city].add(city)
    return cities

def audit_cities(osmfile):    
    osm_file=open(osmfile, 'r')
    cities = defaultdict(set)
    for event, elem in ET.iterparse(osm_file, events=("start",)):
        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.iter("tag"):
                if is_city(tag):
                    cities = audit_city(cities, tag.attrib["v"])
    osm_file.close()
    return dict(cities)
   
def update_city(name, mapping_city):   
    for m in mapping_city:        #replace incorrect name with correct one from mapping dict
        if name == m:            
            name = mapping_city[m]
    return name

if __name__ == '__main__':
    cities = audit_cities(osmfile)
    for city in cities:        
            c = update_city(city,mapping_city)
            print city + "-->" + c