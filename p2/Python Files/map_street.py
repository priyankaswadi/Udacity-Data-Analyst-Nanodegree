	#Print unusual street names from the dataset
expected = ["Street", "Avenue", "Boulevard", "Drive", "Court", "Place", "Square", "Lane", "Road", 
            "Trail", "Parkway", "Commons", "Circle", "Way", "Extension", "Turnpike", "Broadway", "Terrace", "Plaza", "View"]

osm_file = open("albany.osm", "r")

for event, elem in ET.iterparse(osm_file):
    if is_street_name(elem):
        street_name = elem.attrib['v']
        m = street_type_re.search(street_name)
        if m:
            street_type = m.group()
            if street_type not in expected:
                print street_name
