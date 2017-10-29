from collections import OrderedDict
from collections import Counter

def get_year_from_timestamp(timestamp): #return year part from the timestamp
    return int(timestamp[:4])
    

timestamps = []
osm_file=open(osmfile, 'r')
for event, elem in ET.iterparse(osm_file, events=("start",)):
    if elem.tag == "node" or elem.tag == "way":
            year = get_year_from_timestamp(elem.attrib["timestamp"])
            timestamps.append(year)            
osm_file.close()
data = Counter(timestamps) #create a dict of he form {year:'no. of occurences'}
D = OrderedDict(sorted(D.items(), key=lambda v: v))
plt.bar(range(len(D)), D.values(), align='center')
plt.xticks(range(len(D)), D.keys(),fontsize=10, rotation=90)
plt.show()
