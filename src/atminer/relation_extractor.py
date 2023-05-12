#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Description: ...
Author: ...
Month Year
"""
import uuid

import torch 
from torch.nn import functional as F

from transformers import LukeTokenizer, LukeForEntityPairClassification
from atutils.entity_tagger import EntityTagger

#! replace with data_converter from atutils
from atminer.data_converter import DataConverter
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------
# Use LUKE as the basemodel for the EntityRecognizer
class RelationExtractor(object):
    """A class to represent a Relation Extractor.

    Args:
        model_name (str, optional): The name of the model to use. Defaults to "luke".
        model_path (str, optional): The path to the model. Defaults to None.
        model_version (str, optional): The version of the model to use. Defaults to None.
        logger (loguru.logger, optional): A logger object. Defaults to None.
        context_mode (str, optional): The context mode to use. Defaults to None.
        context_size (int, optional): The context size to use. Defaults to None.
        local_files_only (bool, optional): Whether to load the model from local files only. Defaults to None.
        max_seq_length (int, optional): The maximum sequence length of the model. Defaults to None.
    """    
    def __init__(self, model_name="luke",model_path=None, model_version=None, logger=None, context_mode=None, context_size=None, local_files_only=None, max_seq_length=None, tag_entities=False):
        self.model_name = model_name
        self.model_version = model_version
        self.model_path = model_path
        self.logger = logger
        self.local_files_only = local_files_only
        self.max_seq_length = max_seq_length
        self.tag_entities = tag_entities
        self.context_size =  context_size

        self.data_converter = DataConverter(logger, context_size = self.context_size)

        self.output_format = None

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        if model_name == "luke":
            self._init_luke_model()


    def _format_relation(self, rel_id, head_role, head_id, tail_role, tail_id, rel_type, context_start_char, context_end_char):
        """Format the relation as a dictionary.

        Args:
            rel_id (int): The id of the relation.
            head_role (str): The role of the head entity.
            head_id (str):  The id of the head entity.
            tail_role (str): The role of the tail entity.
            tail_id (str): The id of the tail entity.
            rel_type (str): The type of the relation.
            context_start_char (int): The index of the start character of the context.
            context_end_char (int): The index of the end character of the context.

        Returns:
            dict: A dictionary containing the relation.
        """        
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
        """Initialize the LUKE model."""
        if self.local_files_only:
            self.logger.info("[Rel. Ext.] Load model from local files.")
            self.model = LukeForEntityPairClassification.from_pretrained(self.model_path, local_files_only=True)
            self.tokenizer = LukeTokenizer.from_pretrained(self.model_path)
        else:
            self.logger.info("[Rel. Ext.] Load model from HuggingFace model hub.")
            self.logger.error(f"No model in Huggingface cloud so far ")
            raise ValueError("No local model indicated.")

        self.model = self.model.to(self.device)


    def _predict_with_luke(self, luke_data):
        """Create the relation prediction with LUKE.

        Args:
            luke_data (list): A list of dictionaries containing the article formatted for LUKE.

        Returns:    
            list: A list of dictionaries containing the relations.
        """
        relations = list()
        for  relation_dict in luke_data:
            try:
                entity_spans = [ 
                    tuple(relation_dict["head"]),
                    tuple(relation_dict["tail"])
                ]
                self.logger.trace(f"Rel. Ext. Input text: {relation_dict['sentence']}")
                inputs = self.tokenizer(relation_dict["sentence"], entity_spans=entity_spans, return_tensors="pt", truncation=True, padding="max_length", max_length=self.max_seq_length).to(self.device)
                outputs = self.model(**inputs)
                logits = outputs.logits
                probability_scores = logits.softmax(dim=-1).tolist()[0]
                probability_score = probability_scores[int(logits[0].argmax())]
                


                predicted_class_idx = int(logits[0].argmax())
                pred_rel = self.model.config.id2label[predicted_class_idx]
                
                rel = {
                    "id": uuid.uuid4().hex,
                    "head_role": relation_dict["head_type"],
                    "head_id": relation_dict["head_id"],
                    "tail_role": relation_dict["tail_type"],
                    "tail_id": relation_dict["tail_id"],
                    "type": pred_rel,
                    "context_start_char": relation_dict["context_start_char"],
                    "context_end_char": relation_dict["context_end_char"],
                    "context_size": self.context_size,
                    "annotator": f"{self.model_name}-{self.model_version}",
                    "probability_score": probability_score
                }

                relations.append(rel)
            
            except Exception as e:
                #! 1. Even if error you can return an none relation
                #! 2. Count the amount of errors
                self.logger.error(f"Relation Extraction failed for relation_dict: {relation_dict}")
                self.logger.error(f"Error: {e}")
        return relations


    def predict(self, document):
        """Predict the relations for a given document.

        Args:
            document (bconv.doc.document.Document): The document to predict the relations for.

        Raises:
            ValueError: If the model name is not supported.

        Returns:
            list: A list of dictionaries containing the relations.
        """        
        if self.model_name == "luke":
            luke_data = self.data_converter.to_luke(document)
            
            if self.tag_entities:
                self.logger.debug(f"Tagging entities in LUKE dataset.")
                ent_tagger = EntityTagger()
                luke_data = ent_tagger.tag_luke_data(luke_data)
            #! REMOVE AFTER TESTING  >>>>>>>>>>>>
            # luke_data = luke_data[:30]
            #! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            self.logger.debug(f"Start LUKE relation classification for {len(luke_data)} potential relations.")
            pred_relations = self._predict_with_luke(luke_data)

        else:
            self.logger.error(f"Relation extraction {self.model_name} model not supported.")
            raise ValueError("Relation extraction not supported.")

        return pred_relations
