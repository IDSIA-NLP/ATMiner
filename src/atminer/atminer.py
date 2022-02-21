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
# --------------------------------------------------------------------------------------------

# Should be able to keep track of the different resources need and to produce automatic updates
class ResourceManager(object):
    def __init__(self,):
        self.model_name = model_name
        


# --------------------------------------------------------------------------------------------

# Use OGEr as the basemodel for the EntityRecognizer
class EntityRecognizer(object):
    def __init__(self, model_name="OGER", logger=None):
        self.model_name = model_name
        self.logger = logger
        
    
    def predict(self):
        # predict the entities based given a text
        # return the entity offsets, lables and ids
        working_dir = os.getcwd()
        os.chdir("./oger_service")

        cmd = ['./run_oger.sh']
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        p.wait()
        os.chdir(working_dir)
        self.logger.debug(f"NER subprocess return code: {p.returncode}")
        pass



# --------------------------------------------------------------------------------------------

# Use LUKE as the basemodel for the EntityRecognizer
class RelationExtractor(object):
    def __init__(self, model_name="LUKE", logger=None):
        self.model_name = model_name
        self.logger = logger
        
    
    def predict(self):
        # predict the relation based given a text, the entity offsets and the entity labels
        # return the relation type
        pass


# --------------------------------------------------------------------------------------------

# ArthroTraitMiner to run the whole pipline with one command
class ATMiner(object):
    def __init__(self, config, logger):

        self.config = config
        self.logger = logger

        self.rel_extractor = RelationExtractor(
            model_name = self.config['rel_ext']['model'], 
            logger=logger)

        self.ent_recognizer = EntityRecognizer(
            model_name = self.config['ner']['model'], 
            logger=logger)


        self.input = None
        self.document = None    #BioC format
    

    def _load_txt_file(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        return lines


    def _write_txt_file(self, file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            for line in self.input:
                f.write(line)


    def _create_ner_input(self):
        if self.config['ner']['model'] == 'oger':
            tmp_file = self.config['tmp']['path'] + self.config['tmp']['oger_input'] + self.config['input']['file'] + '.txt'
            self._write_txt_file(tmp_file)
        elif self.config['ner']['model'] == 'APossibleSecondModel':
            pass
        else:
            self.logger.error("NER model not supported.")
            raise ValueError("NER model not supported.")


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
        #self._load_ner_output()
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
