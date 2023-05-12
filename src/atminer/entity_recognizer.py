#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Description: ...
Author: ...
Month Year
"""

from io import StringIO
import torch 

from transformers import pipeline, AutoConfig, AutoTokenizer, AutoModelForTokenClassification
from oger.ctrl.router import Router, PipelineServer
from atminer.atunits import ATEntity

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------
# Use OGEr as the basemodel for the EntityRecognizer
class EntityRecognizer(object):
    """ A class to represent an Entity Recognizer.

    Args:
        model_name (str, optional): The name of the model to use. Defaults to "oger".
        model_version (str, optional): The version of the model to use. Defaults to None.
        model_config (dict, optional): A dictionary containing the configuration of the Entity Recognizer. Defaults to None.
        logger (loguru.logger, optional): A logger object. Defaults to None.
    """
    def __init__(self, model_name="oger", model_path=None, model_version=None, model_config=None, logger=None, local_files_only=None, max_seq_length=None ):
        
        self.model_name = model_name
        self.model_version = model_version
        self.model_config = model_config
        self.model_path = model_path

        self.max_seq_length = max_seq_length
        self.local_files_only = local_files_only

        self.logger = logger
        
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        if self.model_name == "oger":
            self._init_oger_pipeline()
        elif self.model_name == "base_transformer":
            self._init_base_transformer_pipeline()


    def _init_base_transformer_pipeline(self):
        """Initialize a HuggingFace transformer pipeline for token classification."""
        model_config = AutoConfig.from_pretrained(self.model_path)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, config=model_config,  local_files_only=True, model_max_length=self.max_seq_length)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_path, config=model_config, local_files_only=True)

        #! REMOVE  ignore_labels=[]   ...when the bug is fixed
        #! Aggregation strategy "simple" will allow that subword tokens are predicted with different labels
        self.base_transformer_pipeline = pipeline(task='ner', model=self.model, tokenizer=self.tokenizer, aggregation_strategy=self.model_config["aggregation_strategy"])
       
        
    def _predict_with_base_transformer(self, text):
        """Create the NER prediction with a HuggingFace transformer.

        Args:
            text (str): The text to predict entities for.

        Returns:
            list: A list of ATEntity objects.       
        """        
        predicted_entities = self.base_transformer_pipeline(text)

        #! TODO: Normalize the entities

        entities = []
        for ent in predicted_entities:

        #if not ent["entity_group"] == "O":
        #    assert ent["word"].lower() in text[ent["start"]:ent["end"]].lower(), f"The predicted entity does not match the text: {ent['word']} vs {text[ent['start']:ent['end']]}"

            entities.append(ATEntity(
                text[ent["start"]:ent["end"]],
                [(int(ent["start"]),int(ent["end"]))],
                ent["entity_group"],
                extra_info={
                    "annotator": f"{self.model_name}-{self.model_version}",
                    "probability_score": float(ent["score"])
                }
            ))
        return entities


    def _init_oger_pipeline(self):
        """Initialize the OGER pipeline."""

        conf = Router(settings=self.model_config["settings_path"])
        self.logger.debug(f"OGER conf: {vars(conf)}")
        
        self.logger.debug(f"OGER conf: {vars(conf)}")
        # Initiziate oger pipline
        self.oger_pipeline = PipelineServer(conf, lazy=True)
        self.logger.debug(f"OGER PipelineServer conf: {vars(self.oger_pipeline._conf)}")


    def _predict_with_oger(self, text):
        """Create the NER prediction with OGER.

        Args:
            text (str): The text to predict entities for.

        Returns:
            list: A list of ATEntity objects.       
        """        
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
        """Create the NER prediction.

        Args:
            text (str): The text to predict entities for.

        Raises:
            ValueError: If the model name is not supported.

        Returns:
            list: A list of ATEntity objects.
        """
        if self.model_name == "oger":
                return self._predict_with_oger(text)
        elif self.model_name == "base_transformer":
                return self._predict_with_base_transformer(text)
        
        else:
            self.logger.error(f"NER model {self.model_name} not supported.")
            raise ValueError("NER model not supported.")

