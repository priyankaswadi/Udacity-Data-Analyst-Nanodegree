#Map incorrect and abbreviated street names with correct/better ones
import xml.etree.cElementTree as ET
from collections import defaultdict
import re
import pprint

OSMFILE = "albany.osm"
street_type_re = re.compile(r'\b\S+\.?$', re.IGNORECASE)

# UPDATE THIS VARIABLE
mapping = {"rd": "Road",
           "Rd": "Road",
           "road": "Road",
           "Ave": "Avenue",           
           "Ave.": "Avenue",    
           "AVE": "Avenue",    
           "way" : "Way",
           "street": "Street",
           "way":"Way",
           "Dr.":"Drive",
           "Blvd":"Boulevard",
           "rt":"Route",
           "Ext": "Extension",
           "Jay":"Jay Street",
           "Nott St E":"Nott Street East",
           "Troy-Schenetady-Road":"Troy Schenectady Road",
           "Troy-Schenetady Rd" :"Troy Schenectady Road",           
           "Delatour":"Delatour Road",
           "Deltour": "Delatour Road",
           "Sparrowbush": "Sparrowbush Road"
           
          }


def audit_street_type(street_types, street_name):
    m = street_type_re.search(street_name)
    if m:
        street_type = m.group()
        if street_type not in expected:
            street_types[street_type].add(street_name)


def is_street_name(elem):
    return (elem.attrib['k'] == "addr:street")


def audit(osmfile):
    osm_file = open(osmfile, "r")
    street_types = defaultdict(set)
    for event, elem in ET.iterparse(osm_file, events=("start",)):

        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.iter("tag"):
                if is_street_name(tag):
                    audit_street_type(street_types, tag.attrib['v'])
    osm_file.close()
    return street_types


def update_name(name, mapping):    
    n = street_type_re.search(name)
    if n:
        n = n.group()
    for m in mapping:
        if n == m:
            name = name[:-len(n)] + mapping[m]
    return name


def test():
    st_types = audit(OSMFILE)   
    pprint.pprint(dict(st_types))
    for st_type, ways in st_types.iteritems():
        for name in ways:
            better_name = update_name(name, mapping)
            if (name == better_name):
                continue
            print name + " --> " + better_name
            
                

if __name__ == '__main__':
    test()