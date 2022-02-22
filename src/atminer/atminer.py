#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Description: ...
Author: ...
Month Year
"""

import subprocess
import json 
import os
from data_converter import DataConverter

# --------------------------------------------------------------------------------------------

# Use OGEr as the basemodel for the EntityRecognizer
class EntityRecognizer(object):
    def __init__(self, model_name="oger", logger=None):
        self.model_name = model_name
        self.logger = logger
        
    
    def predict(self):
        # predict the entities based given a text
        # return the entity offsets, lables and ids
        if self.model_name == "oger":
            working_dir = os.getcwd()
            os.chdir("./oger_service")

            cmd = ['./run_oger.sh']
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            p.wait()
            os.chdir(working_dir)
            self.logger.debug(f"NER subprocess return code: {p.returncode}")
        else:
            self.logger.error(f"NER model {self.model_name} not supported.")
            raise ValueError("NER model not supported.")



# --------------------------------------------------------------------------------------------

# Use LUKE as the basemodel for the EntityRecognizer
class RelationExtractor(object):
    def __init__(self, model_name="luke", logger=None, context_mode=None, context_size=None ):
        self.model_name = model_name
        self.logger = logger

        self.context = context_mode
        self.context_size =  context_size

        self.data_converter = DataConverter()

        self.data = None # luke data format


    def _predict_with_luke(luke_data):

        for doc in luke_data:
            for relation_dict in doc['relations']:

                pred_rel = ... # .... Code that makes the prediction

                #? HOW A PREDICTED RELATION SHOULD LOOK LIKE
                #? {
                #?     "id": "0",
                #?     "node": [
                #?         {
                #?             "role": "Arthropod",
                #?             "refid": "0"
                #?         },
                #?         {
                #?             "role": "Trait",
                #?             "refid": "3"
                #?         }
                #?     ],
                #?     "infons": {
                #?         "type": "hasTrait",
                #?         "context_start_char": 0,
                #?         "context_end_char": 533,
                #?     }
                #? },


    def predict(self, data, input_format='bioc_json'):
        # predict the relation based given a text, the entity offsets and the entity labels
        # return the relation type
        if self.model_name == "luke":
            if input_format == 'bioc_json':
                luke_data = self.data_converter.bioc_to_luke(self.data)

                # pred_relations = self._predict_with_luke(luke_data)

            else:
                self.logger.error(f"Input format {input_format} not supported.")
                raise ValueError("Input format not supported.")

        else:
            self.logger.error(f"Relation extraction {self.model_name} model not supported.")
            raise ValueError("Relation extraction not supported.")


# --------------------------------------------------------------------------------------------

# ArthroTraitMiner to run the whole pipline with one command
class ATMiner(object):
    def __init__(self, config, logger):

        self.config = config
        self.logger = logger

        self.rel_extractor = RelationExtractor(
            model_name = self.config['rel_ext']['model'], 
            logger=logger,
            context_mode=self.config['context']['mode'],
            context_size=self.config['context']['size'])

        self.ent_recognizer = EntityRecognizer(
            model_name = self.config['ner']['model'], 
            logger=logger)


        self.input = None
        self.data = None    #BioC format
    

    def _load_txt_file(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        return lines


    def _write_txt_file(self, file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            for line in self.input:
                f.write(line)


    def _load_bioc_json_file(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data


    def _create_ner_input(self):
        if self.config['ner']['model'] == 'oger':
            tmp_file = self.config['tmp']['path'] + self.config['tmp']['oger_input'] + self.config['input']['file'] + '.txt'
            self._write_txt_file(tmp_file)
        elif self.config['ner']['model'] == 'APossibleSecondModel':
            pass
        else:
            self.logger.error("NER model not supported.")
            raise ValueError("NER model not supported.")


    def _load_ner_output(self):
        if self.config['ner']['model'] == 'oger':
            tmp_file = self.config['tmp']['path'] + self.config['tmp']['oger_output'] + self.config['input']['file'] + '.json'
            self.data = self._load_bioc_json_file(tmp_file)
            self._make_unique_bioc_entity_ids()

        elif self.config['ner']['model'] == 'APossibleSecondModel':
            pass
        else:
            self.logger.error("NER model not supported.")
            raise ValueError("NER model not supported.")


    def _make_unique_bioc_entity_ids(self): 
        """Replace the annotations entities in the BioC JSON with unique ids."""
        for doc in self.data["documents"]:
            id_counter = 0
            for psg in doc["passages"]:
                for anno in psg["annotations"]:
                    anno["id"] = id_counter
                    id_counter += 1


    def _create_rel_ext_input(self):
        pass


    def load(self):
        # Load a plain text or XML file
        if self.config['input']['format'] == 'txt':
            in_file = self.config['input']['path'] + self.config['input']['file'] + '.txt'
            self.input = self._load_txt_file(in_file)
                
        elif self.config['input_format'] == 'bioc_json':
            pass

        else:
            self.logger.error("Input format not supported.")
            raise ValueError("Input format not supported.")


    def ner(self):
        # Produce the Named Entity Recognition 
        self._create_ner_input()
        self.ent_recognizer.predict()
        self._load_ner_output()
        pass


    def split(self, strategy=None):
        # Use a strategy/model to split a text into sentences of context windows 
        pass


    def relation_extraction(self):
        # Extract the relation 
        pass


    def write(self):
        # Write the results to an annoted file or produce the database outputs
        pass 


    def run(self):
        # Load the input
        self.load()

        # Run the NER
        self.ner()

        # Run the relation extraction
        # self.relation_extraction()      
        
        # Write the output 
        # self.write()     
