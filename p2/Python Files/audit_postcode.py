#Audit Postal Codes
import xml.etree.cElementTree as ET
from collections import defaultdict
import re

osmfile = "albany.osm"
def is_postcode(elem):   
    return (elem.attrib['k'] == "addr:postcode")

def audit_postcode(postcodes, postcode):    #create a list of postcodes
    postcodes[postcode].add(postcode)
    return postcodes

def audit_postcodes(osmfile):    
    osm_file=open(osmfile, 'r')
    postcodes = defaultdict(set)
    for event, elem in ET.iterparse(osm_file, events=("start",)):
        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.iter("tag"):
                if is_postcode(tag):
                    postcodes = audit_postcode(postcodes, tag.attrib["v"])
    osm_file.close()
    return dict(postcodes)   
    
def update_postcode(postcode):   #update postal code to remove trailing digits
    if len(postcode)==10 and postcode[5] == '-':
        return postcode[:5]
    elif postcode == "1220y":# update the one instance of incorrectly entered postal code for this extract
        return "12207"
    else:
        return postcode
		
if __name__ == '__main__':
    postcodes = audit_postcodes(osmfile)
    for postcode in postcodes:      
            pcode = update_postcode(postcode)
            print postcode + "-->" + pcode