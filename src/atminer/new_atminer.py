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
import pandas as pd

import bconv
from transformers import LukeTokenizer, LukeModel, LukeForEntityPairClassification
from oger.ctrl.router import Router, PipelineServer


from atminer.new_dataconv import DataConverter

# --------------------------------------------------------------------------------------------

class ATEntity(object):
    def __init__(self, text, spans, ent_type, preferred_form="", resource="", native_id="", cui="", extra_info=None):
        self.id_ = uuid.uuid4().hex
        self.text = text
        self.spans = sorted((start, end) for start, end in spans)
        
        self.metadata = {}
        self.metadata["type"] = ent_type
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
    def __init__(self, model_name="oger", model_version=None, model_config=None, logger=None):
        
        self.model_name = model_name
        self.model_version = model_version
        self.model_config = model_config

        self.logger = logger

        if self.model_name == "oger":
            self._init_oger_pipeline()


    def _init_oger_pipeline(self):
        conf = Router(settings=self.model_config["settings_path"])
        self.logger.debug(f"OGER conf: {vars(conf)}")
        
        self.logger.debug(f"OGER conf: {vars(conf)}")
        # Initiziate oger pipline
        self.oger_pipeline = PipelineServer(conf, lazy=True)
        self.logger.debug(f"OGER PipelineServer conf: {vars(self.oger_pipeline._conf)}")

    def _predict_with_oger(self, text):
        doc = self.oger_pipeline.load_one(StringIO(text), 'txt')

        self.oger_pipeline.process(doc)
        self.oger_pipeline.postfilter(doc)

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
                extra_info={"annotator": f"{self.model_name}-{self.model_version}"}
            ))
        return entities


    def predict(self, text):
        """ Create the NER prediction.

        Raises:
            ValueError: if NER model is not supported.

        Returns:
            None: no return NER output is stored in the ./data/tmp/oger_output, as of now
        """
        if self.model_name == "oger":
                return self._predict_with_oger(text)
        else:
            self.logger.error(f"NER model {self.model_name} not supported.")
            raise ValueError("NER model not supported.")



# --------------------------------------------------------------------------------------------

# Use LUKE as the basemodel for the EntityRecognizer
class RelationExtractor(object):
    def __init__(self, model_name="luke",model_path=None, model_version=None, logger=None, context_mode=None, context_size=None, local_files_only=None ):
        self.model_name = model_name
        self.model_version = model_version
        self.model_path = model_path
        self.logger = logger
        self.local_files_only = local_files_only
        self.context = context_mode
        if self.context == "single":
            self.context_size = 1
        else:
            self.context_size =  context_size

        self.data_converter = DataConverter(logger, context_size = self.context_size)

        self.output_format = None


    def _format_relation(self, rel_id, head_role, head_id, tail_role, tail_id, rel_type, context_start_char, context_end_char):
        relation = dict()
        relation["id"] = rel_id
        relation["node"] = ((head_id,head_role),(tail_id, tail_role))
        relation["metadata"] = {
            "type": rel_type,
            "context_start_char":context_start_char,
            "context_end_char":context_end_char,
            "annotator": f"{self.model_name}-{self.model_version}"
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

        # relations = dict()
        relations = list()
        for rel_idx, relation_dict in enumerate(luke_data):

            entity_spans = [ 
                tuple(relation_dict["head"]),
                tuple(relation_dict["tail"])
            ]
            inputs = tokenizer(relation_dict["text"], entity_spans=entity_spans, return_tensors="pt")
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_idx = int(logits[0].argmax())
            pred_rel = model.config.id2label[predicted_class_idx]
            
            rel = {
                "id": rel_idx,
                "head_role": relation_dict["head_type"],
                "head_id": relation_dict["head_id"],
                "tail_role": relation_dict["tail_type"],
                "tail_id": relation_dict["tail_id"],
                "type": pred_rel,
                "context_start_char": relation_dict["context_start_char"],
                "context_end_char": relation_dict["context_end_char"],
                "context_size": self.context_size,
                "annotator": f"{self.model_name}-{self.model_version}"
            }

            relations.append(rel)
        
        return relations


    def predict(self, document):
        """Create the relation predictions.
        """

        # predict the relation based given a text, the entity offsets and the entity labels
        # return the relation type

        if self.model_name == "luke":
            luke_data = self.data_converter.to_luke(document)
            pred_relations = self._predict_with_luke(luke_data)

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
            model_version = self.config['rel_ext']['version'], 
            model_path = self.config['models']['path'] + self.config['rel_ext']['model_path'],
            local_files_only = self.config['rel_ext']['from_local'],
            logger=logger,
            context_mode=self.config['context']['mode'],
            context_size=self.config['context']['size'])

        self.ent_recognizer = EntityRecognizer(
            model_name = self.config['ner']['model'],
            model_version = self.config['rel_ext']['version'],
            model_config= self.config['ner'],
            logger=logger)


        self.input = None
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


    # ------------------------------------ NER Pipeline --------------------------------------
    def _ner_predict_from_document(self, document):
        for section in document:
            for sentence in section:
                entities = self.ent_recognizer.predict(sentence.text)
                #TODO: Improvment make it optional to add predicted entities later to the document
                # Add entities to the sentence.
                new_entities = []
                for entity in entities:
                    # Note: bconv checks if entity text match with offset selected sentence text
                    self.logger.trace(f"Entity: {vars(entity)}")
                    self.logger.trace(f"Spans: {entity.spans}")
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
                self.logger.trace(f"Document entities:{[ [e.id, e.start, e.end, e.text, e.metadata ] for e in list(document.iter_entities())]}")
                
        elif self.doc_type == "document":
            self.logger.debug(f"Number of document entities: {len(list(self.doc.iter_entities()))}")
            self.logger.trace(f"Document entities:{[ [e.id, e.start, e.end, e.text, e.metadata ] for e in list(self.doc.iter_entities())]}")

        else:
            self.logger.error("Document type is not supported.")
            raise ValueError("Document type is not supported.")

    # ---------------------------------- Relation Ext. Pipeline ------------------------------
    def _drop_relation_duplicates(self, pred_relations , mode="random"):
        self.logger.debug(f"Number of pred. relations before pruning: {len(pred_relations)}")

        if mode == "random":
            df = pd.DataFrame(pred_relations)
            df.drop_duplicates(subset=['head_id', 'tail_id'], keep='last', inplace=True)
            pruned_relations =  df.to_dict("records")
        else:
            self.logger.error("Pruning mode is not supported.")
            raise ValueError("Pruning mode is not supported.")

        self.logger.debug(f"Number of pred. relations after pruning: {len(pruned_relations)}")
        return pruned_relations


    def _format_relation(self, rel):
        relation = dict()
        relation["id"] = rel["id"]
        relation["node"] = ((rel["head_id"],rel["head_role"]),(rel["tail_id"], rel["tail_role"]))
        relation["metadata"] = {
            "type": rel["type"],
            "context_start_char": rel["context_start_char"],
            "context_end_char": rel["context_end_char"],
            "context_size": rel["context_size"],
            "annotator": rel["annotator"]
        }
        return relation


    def _rel_ext_predict_from_document(self, document):
        pred_relations = self.rel_extractor.predict(document)

        # Remove relation duplicate 
        pred_relations = self._drop_relation_duplicates(pred_relations, mode=self.config['rel_ext']['prune_mode'])

        for pred_relation in pred_relations:
            # pred_relation = {
            #     "id": rel_idx,
            #     "head_role": relation_dict["head_type"],
            #     "head_id": relation_dict["head_id"],
            #     "tail_role": relation_dict["tail_type"],
            #     "tail_id": relation_dict["tail_id"],
            #     "type": pred_rel,
            #     "context_start_char": relation_dict["context_start_char"],
            #     "context_end_char": relation_dict["context_end_char"],
            #     "context_size": self.context_size,
            #     "annotator": f"{self.model_name}-{self.model_version}"
            # }
            fmt_rel = self._format_relation(pred_relation)
            new_rel = bconv.Relation(fmt_rel['id'], fmt_rel['node'])
            new_rel.metadata = fmt_rel['metadata']

            document.relations.append(new_rel)
        
        document.sanitize_relations()

    def _rel_ext_predict(self):
        if self.doc_type == "collection":
            for document in self.doc:
                self._rel_ext_predict_from_document(document)                
                
        elif self.doc_type == "document":
            self._rel_ext_predict_from_document(self.doc) 
            
        else:
            self.logger.error("Document type is not supported.")
            raise ValueError("Document type is not supported.")

    def _check_rel_ext_predictions(self):
        if self.doc_type == "collection":
            for document in self.doc:
                self.logger.debug(f"Number of document relations: {len(list(document.iter_relations()))}")
                self.logger.trace(f"Document relations:{[ [e ] for e in list(document.iter_relations())]}")
                
        elif self.doc_type == "document":
            self.logger.debug(f"Number of document relations: {len(list(self.doc.iter_relations()))}")
            self.logger.trace(f"Document relations:{[ [e ] for e in list(self.doc.iter_relations())]}")

        else:
            self.logger.error("Document type is not supported.")
            raise ValueError("Document type is not supported.")

    # ------------------------------------ Main Pipeline ------------------------------------ 
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


    def ner(self):
        """Run the NER prediction pipeline.
        """
        # Produce the Named Entity Recognition  
        self._ner_predict()
        self._check_ner_predictions()
        


    def relation_extraction(self):
        """Run the relation extraction pipeline.
        """

        # Extract the relation 
        self._rel_ext_predict()
        self._check_rel_ext_predictions()


    def write_output(self): 
        if self.config["output"]["format"] in self.input_formats:
            file_path = f'{self.config["output"]["path"]}{self.config["input"]["file"]}.{self.config["output"]["extension"]}'
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
        self.relation_extraction()      
        
        # # Write the output 
        self.write_output()     



    def eval(self):

        # Load annotated file,  copy and remove all entites and relations

        # NER: Iter document entities for both the annotated and predicted documents
        #                 
        # REL: 1) Predict based on original annotations
        #      2) Predict base based on preicted entities
        pass