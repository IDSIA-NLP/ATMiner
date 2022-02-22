import json
from loguru import logger
from data_converter import DataConverter


with open("/Users/joseph/Code/ATMiner/data/tmp/oger_output/PMC3082960.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for doc in data["documents"]:
    id_counter = 0
    for psg in doc["passages"]:
        for anno in psg["annotations"]: 
            anno["id"] = id_counter
            id_counter += 1

dc = DataConverter(logger, "spacy", "en_core_web_sm", 2)
luke_data = dc.bioc_to_luke(data)

print(luke_data)