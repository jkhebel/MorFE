import xml.etree.ElementTree as ET

xml_file = '/Users/jai/Downloads/full database.xml'

tree = ET.parse(xml_file)

root = tree.getroot()

[elem.tag for elem in root.iter('*name')]

for elem in root:
    for i in range(len(elem)):
        print(elem[i])
    break
    # if "name" in elem[i].tag:
    #     name = (elem[i].text).lower()
    #     print(name)


tag = 'classification'
for elem in root:
    for i in range(len(elem)):
        # if tag in elem[i].tag:
        #     print(elem[i])

        print(elem[i].tag, elem[i].text)
