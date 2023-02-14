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
import torch 
import glob

import bconv
from transformers import LukeTokenizer, LukeModel, LukeForEntityPairClassification
from oger.ctrl.router import Router, PipelineServer


from atminer.new_dataconv import DataConverter

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

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

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        if model_name == "luke":
            self._init_luke_model()

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

    def _init_luke_model(self):
        if self.local_files_only:
            self.logger.info("[Rel. Ext.] Load model from local files.")
            self.model = LukeForEntityPairClassification.from_pretrained(self.model_path, local_files_only=True)
            self.tokenizer = LukeTokenizer.from_pretrained(self.model_path)
        else:
            self.logger.info("[Rel. Ext.] Load model from HuggingFace model hub.")
            self.model = LukeForEntityPairClassification.from_pretrained("studio-ousia/luke-large-finetuned-tacred")
            self.tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-large-finetuned-tacred")

        self.model = self.model.to(self.device)


    def _predict_with_luke(self, luke_data):
        # relations = dict()
        relations = list()
        for rel_idx, relation_dict in enumerate(luke_data):
            try:
                entity_spans = [ 
                    tuple(relation_dict["head"]),
                    tuple(relation_dict["tail"])
                ]
                self.logger.trace(f"Rel. Ext. Input text: {relation_dict['text']}")
                inputs = self.tokenizer(relation_dict["text"], entity_spans=entity_spans, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                logits = outputs.logits
                predicted_class_idx = int(logits[0].argmax())
                pred_rel = self.model.config.id2label[predicted_class_idx]
                
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
            
            except:
                #! 1. Even if error you can return an none relation
                #! 2. Count the amount of errors
                self.logger.error(f"Relation Extraction failed for relation_dict: {relation_dict}")
        
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
    def load_input(self, file_path):
        if self.config["input"]["format"] in self.input_formats:
            self.doc = bconv.load(file_path, fmt=self.config["input"]["format"], byte_offsets=False)
            
            self.logger.debug(f"Input document type: {type(self.doc)}")
            if type(self.doc) == bconv.doc.document.Collection:
                self.doc_type = "collection"
            elif type(self.doc) == bconv.doc.document.Document:
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


    def write_output(self, input_file_path): 
        if self.config["output"]["format"] in self.input_formats:

            if self.config["input"]["type"] == "single":
                file_path = f'{self.config["output"]["path"]}{self.config["input"]["file"]}.{self.config["output"]["extension"]}'
            elif self.config["input"]["type"] == "multiple":
                input_file_name = ".".join(input_file_path.split("/")[-1].split(".")[:-1])
                file_path = f'{self.config["output"]["path"]}{input_file_name}.{self.config["output"]["extension"]}'
            else:
                self.logger.error("Input type not supported.")
                raise ValueError("Input type not supported.")

            with open(file_path, 'w', encoding='utf8') as f:
                #TODO: Might need more specification of the different output formats options
                bconv.dump(self.doc, f, fmt=self.config["output"]["format"])
        else:
            self.logger.error("Output format not supported.")
            raise ValueError("Output format not supported.")

        self.logger.info(f"Wrote output document to file: {file_path}")


    def run(self):
    
        if self.config["input"]["type"] == "single":
            self.logger.info("Loading single document path.")
            input_file_paths = [f'{self.config["input"]["path"]}{self.config["input"]["file"]}.{self.config["input"]["extension"]}']
        elif self.config["input"]["type"] == "multiple":
            self.logger.info("Loading multiple document paths.")
            input_file_paths =  glob.glob(f'{self.config["input"]["path"]}*')
        else:
            self.logger.error("Input type not supported.")
            raise ValueError("Input type not supported.")
            

        for input_file_path in input_file_paths:
            self.logger.info(f"Run ATMiner pipeline for document: {input_file_path}")

            # Load the input
            self.logger.info(f"Loading input file.")
            self.load_input(input_file_path)

            # Run the NER
            self.logger.info(f"Start NER predictions.")
            self.ner()

            # Run the relation extraction
            self.logger.info(f"Start relation extraction.")
            self.relation_extraction()      
            
            # Write the output 
            self.logger.info(f"Writing output to file.")
            self.write_output(input_file_path)     



    def eval(self):

        # Load annotated file,  copy and remove all entites and relations

        # NER: Iter document entities for both the annotated and predicted documents
        #                 
        # REL: 1) Predict based on original annotations
        #      2) Predict base based on preicted entities
        
        
        if self.config["input"]["type"] == "single":
            self.logger.info("Loading single document path.")
            input_file_paths = [f'{self.config["input"]["path"]}{self.config["input"]["file"]}.{self.config["input"]["extension"]}']
        elif self.config["input"]["type"] == "multiple":
            self.logger.info("Loading multiple document paths.")
            input_file_paths =  glob.glob(f'{self.config["input"]["path"]}*')
        else:
            self.logger.error("Input type not supported.")
            raise ValueError("Input type not supported.")
            

        for input_file_path in input_file_paths:
            self.logger.info(f"Run ATMiner pipeline for document: {input_file_path}")

            # Load the input
            self.logger.info(f"Loading input file.")
            self.load_input(input_file_path)
            
            # ------- Set doc to the right level
            if self.doc_type == "collection":
                self.doc = self.doc[0]
                self.doc_type = "document"
                # Else the doc is already is already bconv document

            # ------- Store gold annotations          
            gold_entities = {}
            for e in list(self.doc.iter_entities()):
                # print(f"{e.id}, {e.start}, {e.end}, {e.text}, {e.metadata} ")
                gold_entities[e.id] = {"id":e.id, "start": e.start, "end": e.end, "text": e.text, "type": e.metadata["type"], "annotator":e.metadata["annotator"]}

            gold_relations_with_entities = []
            for r in list(self.doc.iter_relations()):
                # print(f"{r.id}, {r.type}, {r._children}, {r._children[0].refid}, {r.metadata}")
                
                gold_relations_with_entities.append([{
                    "id":r.id, 
                    "type":r.type, 
                    "metadata": r.metadata, 
                    "entities": [{"entity": gold_entities[c.refid], "role":c.role} for c in r._children]
                }])

            self.logger.debug(f"Number of gold entities: {len(gold_entities)}")
            self.logger.debug(f"Number of gold relations: {len(gold_relations_with_entities)}")
            self.logger.debug(f"Sample of gold entities: {list(gold_entities.items())[0]}")
            self.logger.debug(f"Sample of gold relations: {gold_relations_with_entities[0]}")

            # ---------------------------------------------
            #           Relation with gold entities
            # ---------------------------------------------

            # ------- Delete relations 
            self.logger.debug(f"Delete gold relations")
            # Delete relation
            self.doc.relations = []
            self.logger.debug(f"Num doc relations after deletion: {len(list(self.doc.iter_relations()))}")

            # ------- Predict relations based on gold entities
            # Run the relation extraction
            self.logger.info(f"Start relation extraction based on gold entities.")
            self.relation_extraction()  
            
            # ------- Store relations based on gold entities
            pred_relations_with_gold_entities = []
            for r in list(self.doc.iter_relations()):
                # print(f"{r.id}, {r.type}, {r._children}, {r._children[0].refid}, {r.metadata}")
                
                pred_relations_with_gold_entities.append([{
                    "id":r.id, 
                    "type":r.type, 
                    "metadata": r.metadata, 
                    "entities": [{"entity": gold_entities[c.refid], "role":c.role} for c in r._children]
                }])

            self.logger.debug(f"Number of gold entities: {len(gold_entities)}")
            self.logger.debug(f"Number of pred relations with gold entities: {len(pred_relations_with_gold_entities)}")
            self.logger.debug(f"Sample of gold entities: {list(gold_entities.items())[0]}")
            self.logger.debug(f"Sample of pred relations with gold entities: {pred_relations_with_gold_entities[0]}")
            
            # ---------------------------------------------
            #           Relation with pred entities
            # ---------------------------------------------

            # ------- Delete relations and entities


            # Delete relation
            self.logger.debug(f"Delete pred relations with gold entities...")
            self.doc.relations = []
            self.logger.debug(f"Num doc relations after deletion pred relation w. g. e.: {len(list(self.doc.iter_relations()))}")

            # Delete entities
            self.logger.debug(f"Delete gold entities...")
            for pa in self.doc:
                for sent in pa:
                    sent.entities = []
            self.logger.debug(f"Num doc entities after deletion gold entities: {len(list(self.doc.iter_entities()))}")

            # ------- Predict entities and relations 
            
            # Run the NER
            self.logger.info(f"Start NER predictions.")
            self.ner()

            # Run the relation extraction
            self.logger.info(f"Start relation extraction based on predicted entities.")
            self.relation_extraction()  

            # ------- Store entities and relations
            pred_entities = {}
            for e in list(self.doc.iter_entities()):
                print(f"{e.id}, {e.start}, {e.end}, {e.text}, {e.metadata} ")
                pred_entities[e.id] = {"id":e.id, "start": e.start, "end": e.end, "text": e.text, "type": e.metadata["type"], "annotator":e.metadata["annotator"]}

            pred_relations_with_pred_entities = []
            for r in list(self.doc.iter_relations()):
                # print(f"{r.id}, {r.type}, {r._children}, {r._children[0].refid}, {r.metadata}")
                
                pred_relations_with_pred_entities.append([{
                    "id":r.id, 
                    "type":r.type, 
                    "metadata": r.metadata, 
                    "entities": [{"entity": pred_entities[c.refid], "role":c.role} for c in r._children]
                }])

            self.logger.debug(f"Number of pred entities: {len(pred_entities)}")
            self.logger.debug(f"Number of pred relations with pred entities: {len(pred_relations_with_pred_entities)}")
            self.logger.debug(f"Sample of pred entities: {list(pred_entities.items())[0]}")
            self.logger.debug(f"Sample of pred relations with pred entities: {pred_relations_with_pred_entities[0]}")

            # ------- Eval entities 

            # ------- Eval relation based on gold entities

            # ------- Eval relation based on predicted entities


            # ------- Write eval results and report



            # # Run the NER
            # self.logger.info(f"Start NER predictions.")
            # self.ner()

            # # Run the relation extraction
            # self.logger.info(f"Start relation extraction.")
            # self.relation_extraction()      
            
            # # Write the output 
            # self.logger.info(f"Writing output to file.")
            # self.write_output(input_file_path)   