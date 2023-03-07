import sys
from os.path import dirname
sys.path.append(dirname(__file__))

import json
from loguru import logger
from src.atminer.data_converter import DataConverter
import bconv

coll = bconv.load('../data/tmp/input/awqipQxjojnzAwukvMo3ZZv9r3KW-zookeys.687.13164.bioc.json', fmt='bioc_json', byte_offsets=False)

for doc in coll:
    dc = DataConverter(logger, 2)
    luke_data = dc.to_luke(doc)


for e in luke_data:
    if e["head_text"] != e["text"][e["head"][0]:e["head"][1]]:
        print("MALE FORMATTED!")
        break
    else:
        print(e["head_text"], e["text"][e["head"][0]:e["head"][1]])
        print(e["head_type"])

    if e["tail_text"] != e["text"][e["tail"][0]:e["tail"][1]]:
        print("MALE FORMATTED!")
        break
    else:
         print(e["tail_text"], e["text"][e["tail"][0]:e["tail"][1]])
         print(e["tail_type"])
    print()
print(luke_data)