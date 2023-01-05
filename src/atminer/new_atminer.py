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
from io import StringIO
import uuid

import bconv
from transformers import LukeTokenizer, LukeModel, LukeForEntityPairClassification
from oger.ctrl.router import Router, PipelineServer


from atminer.data_converter import DataConverter

# --------------------------------------------------------------------------------------------

class ATEntity(object):
    def __init__(self, text, spans, ent_type, preferred_form="", resource="", native_id="", cui="", extra_info=None):
        self.id_ = uuid.uuid4().hex
        self.text = text
        self.spans = sorted((start, end) for start, end in spans)
        
        self.metadata = {}
        self.metadata["ent_type"] = ent_type
        self.metadata["preferred_form"]  = preferred_form
        self.metadata["resource"]  = resource
        self.metadata["native_id"]  = native_id
        self.metadata["cui"]  = cui

        if type(extra_info) == dict or extra_info == None:
            if not extra_info == None:
                if not set(extra_info.keys()) & set(self.metadata.keys()):
                    self.metadata.update(extra_info)
                else:
                    raise ValueError("Extra-info cannot have overlapping keys with metadata.")
        else:
            raise ValueError("Extra-info must be type of dict or None.")

    def shift_offset(self, shift_by):
        self.start += shift_by
        self.end += shift_by

    def update_metadata(self, extra_info):
        if type(extra_info) == dict:
            if not set(extra_info.keys()) & set(self.metadata.keys()):
                self.metadata.update(extra_info)
            else:
                raise ValueError("Extra-info cannot have overlapping keys with metadata.")
        else:
            raise ValueError("Extra-info must be type of dict")

    

# Use OGEr as the basemodel for the EntityRecognizer
class EntityRecognizer(object):
    def __init__(self, model_name="oger", model_config=None, logger=None):
        
        self.model_name = model_name
        self.model_config = model_config

        self.logger = logger

        if self.model_name == "oger":
            self._init_oger_pipeline()


    def _init_oger_pipeline(self):
        conf = Router(settings=self.model_config["settings_path"])
        # Initiziate oger pipline
        self.oger_pipeline = PipelineServer(conf)


    def _predict_with_oger(self, text):
        doc = self.oger_pipeline.load_one(StringIO(text), 'txt')

        self.oger_pipeline.process(doc)

        entities = []
        for ent in doc.iter_entities():
            entities.append(ATEntity(
                ent.text,
                [(ent.start,ent.end)],
                ent.type,
                preferred_form = ent.pref, 
                resource = ent.db, 
                native_id = ent.cid, 
                cui = ent.cui,
                extra_info={"annotator": "OGER model"}
            ))
        return entities


    def predict(self, text):
        """ Create the NER prediction.

        Raises:
            ValueError: if NER model is not supported.

        Returns:
            None: no return NER output is stored in the ./data/tmp/oger_output, as of now
        """

        # predict the entities based given a text
        # return the entity offsets, lables and ids
        if self.model_name == "oger":
                return self._predict_with_oger(text)
        else:
            self.logger.error(f"NER model {self.model_name} not supported.")
            raise ValueError("NER model not supported.")



# --------------------------------------------------------------------------------------------

# Use LUKE as the basemodel for the EntityRecognizer
class RelationExtractor(object):
    def __init__(self, model_name="luke",model_path=None, logger=None, context_mode=None, context_size=None, local_files_only=None ):
        self.model_name = model_name
        self.model_path = model_path
        self.logger = logger
        self.local_files_only = local_files_only
        self.context = context_mode
        if self.context == "single":
            self.context_size = 1
        else:
            self.context_size =  context_size

        self.data_converter = DataConverter(logger, "spacy", "en_core_web_sm", context_size = self.context_size)

        self.output_format = None


    def _create_bioc_relation(self, rel_id, head_role, head_id, tail_role, tail_id, rel_type, context_start_char, context_end_char):
        relation = dict()
        relation["id"] = rel_id
        relation["node"] = [{
                "role": head_role, 
                "refid": head_id, 
            },
            {
                "role": tail_role, 
                "refid": tail_id, 
        }]
        relation["infons"] = {
            "type": rel_type,
            "context_start_char":context_start_char,
            "context_end_char":context_end_char,
        }
        return relation

    def _predict_with_luke(self, luke_data):

        if self.local_files_only:
            self.logger.info("[Rel. Ext.] Load model from local files.")
            model = LukeForEntityPairClassification.from_pretrained(self.model_path, local_files_only=True)
            tokenizer = LukeTokenizer.from_pretrained(self.model_path)
        else:
            self.logger.info("[Rel. Ext.] Load model from HuggingFace model hub.")
            model = LukeForEntityPairClassification.from_pretrained("studio-ousia/luke-large-finetuned-tacred")
            tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-large-finetuned-tacred")

        relations = dict()
        for doc in luke_data:
            for rel_idx, relation_dict in enumerate(doc['relations']):

                entity_spans = [ 
                    tuple(relation_dict["head"]),
                    tuple(relation_dict["tail"])
                ]
                inputs = tokenizer(relation_dict["text"], entity_spans=entity_spans, return_tensors="pt")
                outputs = model(**inputs)
                logits = outputs.logits
                predicted_class_idx = int(logits[0].argmax())
                pred_rel = model.config.id2label[predicted_class_idx]
                
                if self.output_format == 'bioc_json':
                    formatted_rel = self._create_bioc_relation(
                        rel_idx, 
                        relation_dict["head_type"], 
                        relation_dict["head_id"], 
                        relation_dict["tail_type"], 
                        relation_dict["tail_id"], 
                        pred_rel, 
                        relation_dict["context_start_char"], 
                        relation_dict["context_end_char"])
                else:
                    self.logger.error(f"Output format {self.output_format} not supported.")
                    raise ValueError("Output format not supported.")

                relations[doc["id"]] = formatted_rel

        return relations

    def predict(self, data, input_format='bioc_json', output_format='bioc_json'):
        """Create the relation predictions.

        Args:
            data (dict): the input data for the rel extraction model
            input_format (str, optional): input format. Defaults to 'bioc_json'.
            output_format (str, optional): output format. Defaults to 'bioc_json'.

        Raises:
            ValueError: if relation extraction model is not supported
            ValueError: if input format is not supported

        Returns:
            dict: dictionary formatted according to the output format
        """

        # predict the relation based given a text, the entity offsets and the entity labels
        # return the relation type
        self.output_format = output_format

        if self.model_name == "luke":
            if input_format == 'bioc_json':
                luke_data = self.data_converter.bioc_to_luke(data)

                pred_relations = self._predict_with_luke(luke_data)

            else:
                self.logger.error(f"Input format {input_format} not supported.")
                raise ValueError("Input format not supported.")

        else:
            self.logger.error(f"Relation extraction {self.model_name} model not supported.")
            raise ValueError("Relation extraction not supported.")

        return pred_relations

# --------------------------------------------------------------------------------------------

# ArthroTraitMiner to run the whole pipline with one command
class ATMiner(object):
    def __init__(self, config, logger):

        self.config = config
        self.logger = logger

        self.rel_extractor = RelationExtractor(
            model_name = self.config['rel_ext']['model'], 
            model_path = self.config['models']['path'] + self.config['rel_ext']['model_path'],
            local_files_only = self.config['rel_ext']['from_local'],
            logger=logger,
            context_mode=self.config['context']['mode'],
            context_size=self.config['context']['size'])

        self.ent_recognizer = EntityRecognizer(
            model_name = self.config['ner']['model'],
            model_config= self.config['ner'],
            logger=logger)


        self.input = None
        #! Remove self.data = None after the update of the relation extraction
        self.data = None    #BioC format
        self.doc = None

        self.input_formats = [
            'bioc_xml', 
            'bioc_json', 
            'conll', 
            'pubtator', 
            'pubtator_fbk', 
            'pubmed', 'pxml', 
            'pmc', 
            'nxml', 
            'pubanno_json', 
            'pubanno_json.tgz', 
            'txt', 
            'txt.json'
        ]
        
        self.output_formats = [
            'bioc_xml', 
            'bioc_json', 
            'conll', 
            'pubtator', 
            'pubtator_fbk', 
            'pubanno_json', 
            'pubanno_json.tgz', 
            'text_csv', 
            'text_tsv'
        ]
    

    
    # def _load_txt_file(self, file_path):
    #     with open(file_path, "r", encoding="utf-8") as f:
    #             lines = f.readlines()
    #     return lines


    # def _write_txt_file(self, file_path):
    #     with open(file_path, "w", encoding="utf-8") as f:
    #         for line in self.input:
    #             f.write(line)


    # def _load_bioc_json_file(self, file_path):
    #     with open(file_path, "r", encoding="utf-8") as f:
    #         data = json.load(f)
    #     return data

    # def _write_bioc_json_file(self, file_path, json_data):
    #     with open(file_path, "w", encoding="utf-8") as f:
    #         json.dump(json_data, f, indent=4, sort_keys=True)

    
    def _load_ner_input(self):
        pass
    
    def _load_rel_ext_input(self):
        pass
    
    def _ner_predict_from_document(self, document):
        for section in document:
            for sentence in section:
                entities = self.ent_recognizer.predict(sentence.text)
                #TODO: Improvment make it optional to add predicted entities later to the document
                # Add entities to the sentence.
                new_entities = []
                for entity in entities:
                    # Note: bconv checks if entity text match with offset selected sentence text
                    self.logger.debug(f"Entity: {vars(entity)}")
                    self.logger.debug(f"Spans: {entity.spans}")
                    new_entities.append(bconv.Entity(entity.id_, entity.text, entity.spans, entity.metadata))
                    # Append entites to document sentence
                    if new_entities:
                        sentence.add_entities(new_entities)

    def _ner_predict(self):

        if self.doc_type == "collection":
            for document in self.doc:
                self._ner_predict_from_document(document)
                
        elif self.doc_type == "document":
            self._ner_predict_from_document(self.doc)

        else:
            self.logger.error("Document type is not supported.")
            raise ValueError("Document type is not supported.")
            

    def _check_ner_predictions(self):
        if self.doc_type == "collection":
            for document in self.doc:
                self.logger.debug(f"Number of document entities: {len(list(document.iter_entities()))}")
                self.logger.debug(f"Document entities:{[ [e.id, e.start, e.end, e.text, e.metadata ] for e in list(document.iter_entities())]}")
                
        elif self.doc_type == "document":
            self.logger.debug(f"Number of document entities: {len(list(self.doc.iter_entities()))}")
            self.logger.debug(f"Document entities:{[ [e.id, e.start, e.end, e.text, e.metadata ] for e in list(self.doc.iter_entities())]}")

        else:
            self.logger.error("Document type is not supported.")
            raise ValueError("Document type is not supported.")

        

    def _rel_ext_predict(self):

        # TODO rewrite everything to that it works with the new doc element         
        pred_relations = self.rel_extractor.predict(self.data, input_format=self.config["output"]["format"], output_format=self.config["output"]["format"])
        if self.config['output']['format'] == 'bioc_json':
            for doc in self.data["documents"]:
                doc["relations"] = pred_relations[doc["id"]]
        else:
            self.logger.error("Output format not supported.")
            raise ValueError("Output format not supported.")

    def _sanitize_rel_ext_output(self):
        #TODO run some checks to make sure that the doc model has the write format 
        pass


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

    def load_input(self):
        if self.config["input"]["format"] in self.input_formats:
            file_path = f'{self.config["input"]["path"]}{self.config["input"]["file"]}.{self.config["input"]["extension"]}'
            self.logger.info(f"Loading input document: {file_path}")

            self.doc = bconv.load(file_path, fmt=self.config["input"]["format"])

            if type(self.doc) == bconv.doc.document.Collection:
                self.doc_type = "collection"
            if type(self.doc) == bconv.doc.document.Document:
                self.doc_type = "document"
            else:
                self.logger.error("Document type is not supported.")
                raise ValueError("Document type is not supported.")
        else:
            self.logger.error("Input format not supported.")
            raise ValueError("Input format not supported.")

    # def load(self):
    #     """Load the input files for the ATMiner predition pipeline

    #     Raises:
    #         ValueError: if the input format is not supported.
    #     """

    #     # Load a plain text or XML file
    #     if self.config['input']['format'] == 'txt':
    #         in_file = self.config['input']['path'] + self.config['input']['file'] + '.txt'
    #         self.input = self._load_txt_file(in_file)
                
    #     elif self.config['input_format'] == 'bioc_json':
    #         pass

    #     else:
    #         self.logger.error("Input format not supported.")
    #         raise ValueError("Input format not supported.")


    def ner(self):
        """Run the NER prediction pipeline.
        """
        # Produce the Named Entity Recognition  
        # self._load_ner_input()
        self._ner_predict()
        self._check_ner_predictions()
        

    def relation_extraction(self):
        """Run the relation extraction pipeline.

        Raises:
            ValueError: if the document output format is not supported.
        """

        # Extract the relation 
        # Both the input_format and output_format is the main output_format
        self._load_rel_ext_input()
        self._rel_ext_predict()
        self._sanitize_rel_ext_output()


    # def write(self):
    #     """Write the preditions to a file.

    #     Raises:
    #         ValueError: if the output format is not supported.
    #     """

    #     # Write the results to an annoted file or produce the database outputs
    #     if self.config['output']['format'] == 'bioc_json':
    #         out_file = self.config['output']['path'] + self.config['output']['file'] + ".bioc.json"
    #         self._write_bioc_json_file(out_file, self.data)
    #     else:
    #         self.logger.error("Output format not supported.")
    #         raise ValueError("Output format not supported.")

    def write_output(self): 
        if self.config["output"]["format"] in self.input_formats:
            file_path = f'{self.config["output"]["path"]}{self.config["output"]["file"]}.{self.config["output"]["extension"]}'
            with open(file_path, 'w', encoding='utf8') as f:
                #TODO: Might need more specification of the different output formats options
                bconv.dump(self.doc, f, fmt=self.config["output"]["format"])
        else:
            self.logger.error("Input format not supported.")
            raise ValueError("Input format not supported.")

        self.logger.info(f"Wrote output document to file: {file_path}")

    def run(self):
        # Load the input
        self.load_input()

        # Run the NER
        self.ner()

        # # Run the relation extraction
        # self.relation_extraction()      
        
        # # Write the output 
        self.write_output()     



    def eval(self):
        pass