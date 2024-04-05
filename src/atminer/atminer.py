#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Description: ...
Author: ...
Month Year
"""

import json 
from io import StringIO
import os
import uuid
import pandas as pd
import torch 
import glob
import time


import bconv


from atminer.entity_recognizer import EntityRecognizer
from atminer.entity_normalizer import EntityNormalizer
from atminer.relation_extractor import RelationExtractor

from seqeval.metrics import classification_report as seqeval_cls_report
from seqeval.scheme import IOB2
from sklearn.metrics import classification_report as sklearn_cls_report

import numpy as np
from datetime import datetime
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------
def list_get(l, idx, default=None):
    try:
        r = l[idx]
    except IndexError:
        r = default
    return r



# --------------------------------------------------------------------------------------------

# ArthroTraitMiner to run the whole pipline with one command
class ATMiner(object):
    """ The ArthroTraitMiner class.

    Args:
        config (dict): The configuration.
        logger (loguru.logger): The logger. (Default value = None)

    """
    def __init__(self, config, logger):

        self.config = config
        self.logger = logger
        self.apply_entity_normalization = self.config['nen']['apply_entity_normalization']

        self.run_id = datetime.now().strftime('%Y%m%d-%H%M%S%f')

        self.rel_extractor = RelationExtractor(
            model_name = self.config['rel_ext']['model'], 
            model_version = self.config['rel_ext']['version'], 
            model_path = self.config['models']['path'] + self.config['rel_ext']['model_path'],
            local_files_only = self.config['rel_ext']['from_local'],
            logger=logger,
            context_size=self.config['context']['size'],
            max_seq_length=self.config['rel_ext']['max_seq_length'],
            tag_entities=self.config['rel_ext']['tag_entities']
            )

        self.ent_recognizer = EntityRecognizer(
            model_name = self.config['ner']['model'],
            model_version = self.config['ner']['version'],
            model_path = self.config['models']['path'] + self.config['ner']['model_path'],
            local_files_only = self.config['ner']['from_local'],
            max_seq_length=self.config['ner']['max_seq_length'],
            model_config= self.config['ner'],
            logger=logger)

        self.ent_normalizer = EntityNormalizer(
            model_name = config['nen']['model'],
            model_version = config['nen']['version'],
            model_config = config['nen'],
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
        """Predict named entities for a given document.

        Args:
            document (bconv.doc.document.Document): The document to predict the named entities for.
        """        
        for section in document:
            for sentence in section:
                entities = self.ent_recognizer.predict(sentence.text)
                #TODO: Improvment make it optional to add predicted entities later to the document
                # Add entities to the sentence.
                new_entities = []
                for entity in entities:
                    if self.apply_entity_normalization:
                        # Predict the normalized entity
                        normalized_entity = self.ent_normalizer.predict(entity.text, entity.ent_type)
                        # Update the entity
                        if normalized_entity:
                            entity.preferred_form = normalized_entity['preferred_form']
                            entity.resource = normalized_entity['resource']
                            entity.native_id = normalized_entity['native_id']
                            entity.cui = normalized_entity['cui']
                            entity.metadata.update({'nen_annotator': normalized_entity['extra_info']['annotator']})
                       
                    # Note: bconv checks if entity text match with offset selected sentence text
                    self.logger.trace(f"Entity: {vars(entity)}")
                    self.logger.trace(f"Spans: {entity.spans}")

                    new_entities.append(bconv.Entity(entity.id_, entity.text, entity.spans, entity.metadata))
                # Append entites to document sentence
                if new_entities:
                    sentence.add_entities(new_entities)


    def _ner_predict(self):
        """Predict named entities for a given document or collection.

        Raises:
            ValueError: If the document type is not supported.
        """

        if self.doc_type == "collection":
            for document in self.doc:
                self._ner_predict_from_document(document)
                
        elif self.doc_type == "document":
            self._ner_predict_from_document(self.doc)

        else:
            self.logger.error("Document type is not supported.")
            raise ValueError("Document type is not supported.")
            

    def _check_ner_predictions(self):
        """Check the named entity predictions.

        Raises:
            ValueError: If the document type is not supported.
        """

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
    def _custom_mode(self, series):
        """Custom mode function. Return the mode according to a defined order.

        Args:
            series (pandas.Series): The series.

        Returns:
            str: The mode.
        """
        mode_counts = series.value_counts()
        max_count = mode_counts.max()
        # Filter modes that have the max count
        potential_modes = mode_counts[mode_counts == max_count].index.tolist()
        # Define the order
        order = ['hasTrait', 'hasValue', 'hasQualifier', 'hasContinuation', 'none']
        # Return the first according to the defined order
        for o in order:
            if o in potential_modes:
                return o
        return series.mode()[0]  # Fallback, should never be reached due to the order list covering all cases

    def _drop_relation_duplicates(self, pred_relations , mode="random"):
        """Drop duplicate relations.

        Args:
            pred_relations (dict) : The predicted relations.
            mode (str, optional): The mode to drop duplicates. (Default value = "random")

        Raises:
            ValueError: If the mode is not supported.

        Returns:
            list: The pruned relations.
        """        
        self.logger.debug(f"Number of pred. relations before removing duplicates: {len(pred_relations)}")

        if mode == "random":
            df = pd.DataFrame(pred_relations)
            df.drop_duplicates(subset=['head_id', 'tail_id'], keep='last', inplace=True)
            pruned_relations =  df.to_dict("records")

        elif mode == "sorted":
            # Sort so we keep relationships in the sorted order this means none relation will kept if we can't find any other
            df = pd.DataFrame(pred_relations)
            #! REMOVE AFTER DEBUGGING
            df.to_csv(f"../logs/debugging/{datetime.today().strftime('%Y-%m-%d-%H-%M-%S-%f')}_df_before_prunded.tsv", sep="\t")
            self.logger.debug(f"types {df['type'].unique()}")
            self.logger.debug(f"types {df.head(50)}")
            df['type'] = df['type'].astype('category') 
            df['type'] = df['type'].cat.set_categories(['hasTrait', 'hasValue', 'hasQualifier', 'hasContinuation', 'none'], ordered=True) 
            #pandas dataframe sort_values to inflicts order on your categories 
            df.sort_values(['head_id', 'tail_id', 'type'], inplace=True, ascending=True) 
            df.drop_duplicates(subset=['head_id', 'tail_id'], keep='first', inplace=True)
            #! REMOVE AFTER DEBUGGING
            df.to_csv(f"../logs/debugging/{datetime.today().strftime('%Y-%m-%d-%H-%M-%S-%f')}_df_after_prunded.tsv", sep="\t")
            pruned_relations =  df.to_dict("records")
            #! REMOVE AFTER DEBUGGING
            with open(f"../logs/debugging/{datetime.today().strftime('%Y-%m-%d-%H-%M-%S-%f')}_pruned_relations.json", "w") as f:
                json.dump(pruned_relations, f)
            self.logger.debug(f"pruned {pruned_relations}")

        # Keep the type that was predicted the most times for each pair of entities
        elif mode == "most_common":
            df = pd.DataFrame(pred_relations)
            # Get the most common type for each pair of head-tail entities, if there are multiple types with the same count keep first of order list
            df['most_common_type'] = df.groupby(['head_id', 'tail_id'])['type'].transform(self._custom_mode)
            # Drop duplicates
            df.drop('type', axis=1)
            # Rename the most_common_type to type
            df.rename(columns={'most_common_type': 'type'})
            # Drop duplicates, where head_id, tail_id and type are the same
            # df = df.drop_duplicates(subset=['head_id', 'tail_id', 'type']) # Why include type here?
            df.drop_duplicates(subset=['head_id', 'tail_id'], inplace=True)
            pruned_relations = df.to_dict("records")
           
            
        else:
            self.logger.error(f"Removing duplicates mode ({mode}) is not supported.")
            raise ValueError(f"Removing duplicates mode ({mode}) is not supported.")

        self.logger.debug(f"Number of pred. relations after removing duplicates: {len(pruned_relations)}")
        return pruned_relations


    def _format_relation(self, rel):
        """Format the relation.

        Args:
            rel (dict): The relation.

        Returns:
            dict: The formatted relation. 
        """        
        relation = dict()
        relation["id"] = rel["id"]
        relation["node"] = ((rel["head_id"],rel["head_role"]),(rel["tail_id"], rel["tail_role"]))
        relation["metadata"] = {
            "type": rel["type"],
            "context_start_char": rel["context_start_char"],
            "context_end_char": rel["context_end_char"],
            "context_size": rel["context_size"],
            "annotator": rel["annotator"],
            "probability_score": rel["probability_score"]
        }
        return relation


    def _rel_ext_predict_from_document(self, document):
        """Predict relations for a given document.

        Args:
            document (bconv.doc.document.Document): The document.
        """        

        pred_relations = self.rel_extractor.predict(document)

        # Remove relation duplicate 
        if pred_relations:
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
        """Predict relations for a given document or collection of documents.

        Raises:
            ValueError: If the document type is not supported.
        """        
        if self.doc_type == "collection":
            for document in self.doc:
                self._rel_ext_predict_from_document(document)                
                
        elif self.doc_type == "document":
            self._rel_ext_predict_from_document(self.doc) 
            
        else:
            self.logger.error("Document type is not supported.")
            raise ValueError("Document type is not supported.")


    def _check_rel_ext_predictions(self):
        """Check the relations predictions.

        Raises:
            ValueError: If the document type is not supported.
        """        
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
    # --------------------------------------------------------------------------------------- 
    def load_input(self, file_path):
        """Load the input document.

        Args:
            file_path (str): The path to the input file.

        Raises:
            ValueError: Document type is not supported.
            ValueError: Input format not supported.
        """        
        if self.config["input"]["format"] in self.input_formats:
            if self.config["input"]["format"] == "bioc_json":
                self.doc = bconv.load(file_path, fmt=self.config["input"]["format"], byte_offsets=False)
            else:
                self.doc = bconv.load(file_path, fmt=self.config["input"]["format"])
            
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
        """Write the output document.

        Args:
            input_file_path (str): The path to the input file. Use the same directory to save the output file.

        Raises:
            ValueError: Output format not supported.
            ValueError: Input type not supported.
        """         
        if self.config["output"]["format"] in self.input_formats:

            # Create the output path /output_path/run-id/
            output_path = f'{self.config["output"]["path"]}run-{self.run_id}/'
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            if self.config["input"]["type"] == "single":
                #file_path = f'{output_path}{self.config["input"]["file"]}.{self.config["output"]["extension"]}'
                #! Debugging REMOVE !!! >>>>>
                file_path = f'{output_path}{self.config["input"]["file"]}--size-{self.config["context"]["size"]}.{self.config["output"]["extension"]}'
                #! <<<<<<<<<<<<<<<<<<<<<<<<<
            elif self.config["input"]["type"] in ["multiple", "load_list"]:
                input_file_name = ".".join(input_file_path.split("/")[-1].split(".")[:-1])
                file_path = f'{output_path}{input_file_name}.{self.config["output"]["extension"]}'
            else:
                self.logger.error("Input type not supported.")
                raise ValueError("Input type not supported.")

            # if not os.path.exists(file_path):
            #     os.makedirs(file_path)

            if self.config["output"]["format"] == "bioc_json":
                with open(file_path, 'w', encoding='utf8') as f:
                    #TODO: Might need more specification of the different output formats options
                    bconv.dump(self.doc, f, fmt=self.config["output"]["format"], byte_offsets=False)

            elif self.config["output"]["format"] == "pubanno_json.tgz":
                with open(file_path+".pubanno_json.tgz", 'wb') as f:
                    bconv.dump(self.doc, f, fmt=self.config["output"]["format"])

            else:
                with open(file_path, 'w', encoding='utf8') as f:
                    #TODO: Might need more specification of the different output formats options
                    bconv.dump(self.doc, f, fmt=self.config["output"]["format"])
        else:
            self.logger.error("Output format not supported.")
            raise ValueError("Output format not supported.")

        self.logger.info(f"Wrote output document to file: {file_path}")


    def _remove_impermissible_relations(self):
        """Remove impermissible relations.

        Args:
            relations (list): The relations.

        Returns:
            list: The relations.
        """        
        # Remove relations that are not in the allowed relation types
        raise NotImplementedError("Not implemented yet.")
        allowed_relation = self.config['allowed_relations']
        allowed_relation #! TODO: Reformate allowed_relation
        if self.doc_type == "collection":
            for document in self.doc:
                #! TODO: Change the iteration method
                for relation in document.relations:
                    if relation.metadata["type"] not in allowed_relation:
                        #! TODO: Change the remove method
                        document.relations.remove(relation)
                
        elif self.doc_type == "document":
            #! TODO: Change the iteration method
            for relation in self.doc.relations:
                if relation.metadata["type"] not in allowed_relation:
                    #! TODO: Change the remove method
                    document.relations.remove(relation)

        else:
            self.logger.error("Document type is not supported.")
            raise ValueError("Document type is not supported.")
        return self.doc
    

    def post_processing(self):
        """Run the post processing pipeline.
        """
        if self.config['post_processing']['remove_impermissible_relations']:
            self.logger.info("Post-processing: Removing impermissible relations...")
            self.doc = self._remove_impermissible_relations()
        else:
            self.logger.info("No post processing applied.")


    def write_config(self):
        # Create the output path /output_path/run-id/
        output_path = f'{self.config["output"]["path"]}run-{self.run_id}/'
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        with open(f'{output_path}config_run-{self.run_id}.json', 'w') as f:
                 json.dump(self.config, f, indent=4, sort_keys=True)

    def run(self):
        """Run the ATMiner pipeline.

        Raises:
            ValueError: Input type not supported.
        """        
        if self.config["input"]["type"] == "single":
            self.logger.info("Loading single document path.")
            input_file_paths = [f'{self.config["input"]["path"]}{self.config["input"]["file"]}.{self.config["input"]["extension"]}']
        
        elif self.config["input"]["type"] == "multiple":
            self.logger.info("Loading multiple document paths.")
            input_file_paths =  glob.glob(f'{self.config["input"]["path"]}*')
        
        elif self.config["input"]["type"] == "load_list":
            self.logger.info("Loading document paths from list.")
            with open(f'{self.config["input"]["load_list"]}', 'r') as f:
                input_file_paths = f.readlines()
                # Clean and check the paths
            input_file_paths = [p.strip() for p in input_file_paths if p.strip()]
            # Get first 1 path
            self.logger.debug(f"Number of input files: {len(input_file_paths)}")
            self.logger.debug(f"First input file: {input_file_paths[0]}")
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

            # Run post processing
            self.logger.info(f"Start post processing.")
            self.post_processing() 
            
            # Write the output 
            self.logger.info(f"Writing output to file.")
            self.write_output(input_file_path)
        
        # Write the used configuration to file
        self.write_config()     


    # ----------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------
    def _normalize_tag(self, tag):
        """ Normalize the tag.

        Args:
            tag (str): The tag to normalize.

        Returns:
            str: The normalized tag.
        """        
        if "Trait-" in tag:
            return "-".join(tag.split("-")[:2])
        else:
            return tag
        
    def _conll_to_list(self, lines):
        """Convert the CoNLL format to a list.

        Args:
            lines (list): The CoNLL lines.

        Returns:
            list: The CoNLL lines as a list.
        """        
        out_list = []
        d = {
            'tokens':[],
            'tags':[]
        }
        for line in lines:
            if not line.startswith("#"):
                if not line.isspace():
                    e = line.strip().split('\t')
                    d["tokens"].append(e[0]) 
                    #! Traits found by OGER has to been split e.g. Trait-Feeding -> Trait
                    d["tags"].append(self._normalize_tag(e[-1]))
                else:
                    if d["tokens"] and d["tags"]:
                        if d['tokens']:
                            out_list.append(d)
                            d = {
                                'tokens':[],
                                'tags':[]
                            }
        return out_list
    

    def _doc_to_entities(self):
        """Convert the document to a list of entities.

        Returns:
            list: The list of entities.
        """
        fp = StringIO()
        bconv.dump(self.doc, fp, fmt='conll', tagset='IOB', include_offsets=True)
        lines = fp.getvalue().splitlines(True)
        return self._conll_to_list(lines)


    def _get_ner_report(self, gold_ner_tags, pred_ner_tags):
        """Get the NER report.

        Args:
            gold_ner_tags (list): List of gold NER tags.
            pred_ner_tags (list): List of predicted NER tags.

        Returns:
            str: The NER report.
        """        
        
        ner_report = f"***** NER Report *****"
        cls_report_strict = seqeval_cls_report(gold_ner_tags, pred_ner_tags, mode="strict" , scheme=IOB2)
        ner_report += f"\n\nStrict:\n{cls_report_strict}"
        cls_report_conlleval = seqeval_cls_report(gold_ner_tags, pred_ner_tags, mode=None , scheme=IOB2)
        ner_report += f"\n\nCONLLEVAL equivalent:\n{cls_report_conlleval}"
        self.logger.debug(ner_report)
        return ner_report

    def _eval_ner(self, gold_ner_tokens_and_tags, pred_ner_tokens_and_tags):
        """Evaluate the NER predictions.

        Args:
            gold_ner_tokens_and_tags (list): The gold NER tokens and tags.
            pred_ner_tokens_and_tags (list): The predicted NER tokens and tags.

        Raises:
            ValueError: The gold and predicted NER tokens and tags are not the same.

        Returns:
            str: The NER evaluation report.
        """
        if gold_ner_tokens_and_tags and pred_ner_tokens_and_tags:        
            gold_tokens = [e["tokens"] for e in gold_ner_tokens_and_tags]
            pred_tokens = [e["tokens"] for e in pred_ner_tokens_and_tags]

            if gold_tokens == pred_tokens:
                gold_tags = [e["tags"] for e in gold_ner_tokens_and_tags]
                pred_tags = [e["tags"] for e in pred_ner_tokens_and_tags]

                ner_report = self._get_ner_report(gold_tags, pred_tags)

            else:
                raise ValueError("Gold tokens and pred tokens are not the equal.")
        else:
            gold_tags, pred_tags = [], []
            if gold_ner_tokens_and_tags:
                ner_report = "***** NER Report *****\n\nNo pred labels provided."
            elif pred_ner_tokens_and_tags:
                ner_report = "***** NER Report *****\n\nNo gold labels provided."
            else:
                ner_report = "***** NER Report *****\n\nNo gold and no pred labels provided."

        return ner_report, gold_tags, pred_tags

    def _eval_rel_ext(self, gold_relations, pred_relations, labels=["hasValue", "hasTrait"], offset_tolerance = 0):
        """Evaluate the relation extraction predictions.

        Args:
            gold_relations (list): The gold relations.
            pred_relations (list): The predicted relations.
            labels (list, optional): The allowed labels. Defaults to ["hasValue", "hasTrait"].
            offset_tolerance (int, optional): The offset tolerance. Defaults to 0.

        Returns:
            str: The relation extraction evaluation report.
        """
        re_report = "***** Relation Ext. Report *****"

        gold_labels = []
        predicted_labels = []
        for gold_rel in gold_relations:
            gold_rel["entities"] = sorted(gold_rel["entities"], key=lambda d: d["entity"]['start'])
            self.logger.trace(f"GOLD: {gold_rel}")
            found = False

            for pred_rel in pred_relations:
                self.logger.trace(f"PRED: {pred_rel}")
                pred_rel["entities"] = sorted(pred_rel["entities"], key=lambda d: d["entity"]['start'])
                
                if gold_rel["entities"][0]["entity"]["start"] == pred_rel["entities"][0]["entity"]["start"] \
                and gold_rel["entities"][0]["entity"]["end"] == pred_rel["entities"][0]["entity"]["end"] \
                and gold_rel["entities"][1]["entity"]["start"] == pred_rel["entities"][1]["entity"]["start"] \
                and gold_rel["entities"][1]["entity"]["end"] == pred_rel["entities"][1]["entity"]["end"]:
                    
                    self.logger.debug(f"\nMATCH IDS gold: {gold_rel['id']},pred: {pred_rel['id']}\n")
                    if gold_rel["type"] == pred_rel["type"] \
                    and (gold_rel["entities"][0]["role"] != pred_rel["entities"][0]["role"] \
                        or gold_rel["entities"][1]["role"] != pred_rel["entities"][1]["role"]):
                        # Same relation label but the entities are labeled in the opposite way
                        # --> force the relation labels to disagree
                        gold_labels.append(gold_rel["type"])
                        predicted_labels.append("none")
                        found = True
                        break
                    else:
                        gold_labels.append(gold_rel["type"])
                        predicted_labels.append(pred_rel["type"])
                        found = True
                        break
            
            if not found:
                gold_labels.append(gold_rel["type"])
                predicted_labels.append("none")
        
        assert len(gold_relations) == len(gold_labels), "Number of gold relations is unequal to evaluation labels"
        
        re_report = self._get_re_report(gold_labels, predicted_labels, labels)     
        return re_report


    def _get_re_report(self, re_gold_labels, re_pred_labels, labels=["hasValue", "hasTrait"]):
        """Get the relation extraction report.

        Args:
            re_gold_labels (list): The gold relation labels.
            re_pred_labels (list): The predicted relation labels.   
            labels (list, optional): The allowed labels. Defaults to ["hasValue", "hasTrait"].

        Returns:
            str: The relation extraction report.
        """
        
        re_report = "***** Relation Ext. Report *****"
        re_report += f"\n Labels: {labels}"
        try:
            cls_report_zero_div_one = sklearn_cls_report(re_gold_labels, re_pred_labels, labels=labels, zero_division=1)
            re_report += f"\n Relation Ext. (zero_division=1):\n\n{cls_report_zero_div_one}"
        except:
            self.logger.error("No report available for zero_division=1.")
            re_report += f"\n Relation Ext. (zero_division=1):\n\nNo report available."

        try:
            cls_report_zero_div_zero = sklearn_cls_report(re_gold_labels, re_pred_labels, labels=labels, zero_division=0)
            re_report += f"\n Relation Ext. (zero_division=0, labels={labels}):\n\n{cls_report_zero_div_zero}"
        except:
            self.logger.error("No report available for zero_division=0, labels=labels.")
            re_report += f"\n Relation Ext. (zero_division=0, labels={labels}):\n\nNo report available."

        try:
            cls_report_zero_div_zero = sklearn_cls_report(re_gold_labels, re_pred_labels, zero_division=0)
            re_report += f"\n Relation Ext. (zero_division=0, label=undefined):\n\n{cls_report_zero_div_zero}"
        except:
            self.logger.error("No report available for zero_division=0, label=undefined.")
            re_report += f"\n Relation Ext. (zero_division=0, label=undefined):\n\nNo report available."

        self.logger.debug(f"Len gold labels: {len(re_gold_labels)}")        
        self.logger.debug(re_report)

        return re_report

    def _eval_rel_ext_strict(self, gold_relations, pred_relations, labels=None):
        """Evaluate the relation extraction predictions with strict offsets.

        Args:
            gold_relations (list): The gold relations.
            pred_relations (list): The predicted relations.
            labels (list, optional): The allowed labels. Defaults to ["hasValue", "hasTrait"].
            offset_tolerance (int, optional): The offset tolerance. Defaults to 0.

        Returns:
            str: The relation extraction evaluation report.
        """
        
        # ---------------------- Convert dictionaries ------------------------
        if gold_relations and pred_relations:
            dg = []
            for gold_rel in gold_relations:
                gold_rel["entities"] = sorted(gold_rel["entities"], key=lambda d: d["entity"]['start'])

                dg.append(
                    {
                    "id":gold_rel['id'], 
                    "type":gold_rel['type'], 
                    "ent_1_id": gold_rel["entities"][0]['entity']['id'], 
                    "ent_1_start": gold_rel["entities"][0]['entity']['start'],
                    "ent_1_end": gold_rel["entities"][0]['entity']['end'],
                    "ent_1_text": gold_rel["entities"][0]['entity']['text'],
                    "ent_1_type": gold_rel["entities"][0]['entity']['type'],
                    "ent_1_role": gold_rel["entities"][0]["role"],
                    "ent_2_id": gold_rel["entities"][1]['entity']['id'], 
                    "ent_2_start": gold_rel["entities"][1]['entity']['start'],
                    "ent_2_end": gold_rel["entities"][1]['entity']['end'],
                    "ent_2_text": gold_rel["entities"][1]['entity']['text'],
                    "ent_2_type": gold_rel["entities"][1]['entity']['type'],
                    "ent_2_role": gold_rel["entities"][1]["role"],
                    }
                )

            dp = []
            for pred_rel in pred_relations:
                pred_rel["entities"] = sorted(pred_rel["entities"], key=lambda d: d["entity"]['start'])

                dp.append(
                    {
                    "id":pred_rel['id'], 
                    "type":pred_rel['type'], 
                    "ent_1_id": pred_rel["entities"][0]['entity']['id'], 
                    "ent_1_start": pred_rel["entities"][0]['entity']['start'],
                    "ent_1_end": pred_rel["entities"][0]['entity']['end'],
                    "ent_1_text": pred_rel["entities"][0]['entity']['text'],
                    "ent_1_type": pred_rel["entities"][0]['entity']['type'],
                    "ent_1_role": pred_rel["entities"][0]["role"],
                    "ent_2_id": pred_rel["entities"][1]['entity']['id'], 
                    "ent_2_start": pred_rel["entities"][1]['entity']['start'],
                    "ent_2_end": pred_rel["entities"][1]['entity']['end'],
                    "ent_2_text": pred_rel["entities"][1]['entity']['text'],
                    "ent_2_type": pred_rel["entities"][1]['entity']['type'],
                    "ent_2_role": pred_rel["entities"][1]["role"],
                    }
                )
            # ---------------------- ---------------------- ------------------------

            # conver to dataframes
            dfgold = pd.DataFrame(dg)
            self.logger.debug(f"dfgold shape: {dfgold.shape}")
            self.logger.debug(f"dfm head 50: {dfgold.head(20)}")
            self.logger.debug(f"dfm tail 50: {dfgold.tail(20)}")
            #! REMOVE AFTER DEBUGGING
            dfgold.to_csv(f"../logs/debugging/{datetime.today().strftime('%Y-%m-%d-%H-%M-%S-%f')}_dfgold.tsv", sep='\t', encoding='utf-8')


            dfpred = pd.DataFrame(dp)
            self.logger.debug(f"dfpred shape: {dfpred.shape}")
            self.logger.debug(f"dfm head 50: {dfpred.head(20)}")
            self.logger.debug(f"dfm tail 50: {dfpred.tail(20)}")
            #! REMOVE AFTER DEBUGGING
            dfpred.to_csv(f"../logs/debugging/{datetime.today().strftime('%Y-%m-%d-%H-%M-%S-%f')}_dfpred.tsv", sep='\t', encoding='utf-8')


            # merge datasets
            dfm = pd.merge(dfgold, dfpred, on=["ent_1_end", "ent_1_start", "ent_2_end", "ent_2_start"], how="outer", suffixes=('_gold', '_pred'))
            self.logger.debug(f"dfm shape: {dfm.shape}") 
            #! REMOVE AFTER DEBUGGING
            dfm.to_csv(f"../logs/debugging/{datetime.today().strftime('%Y-%m-%d-%H-%M-%S-%f')}_test_df_0.tsv", sep='\t', encoding='utf-8')

            # Remove all "none" predictions with no equivalent gold labels
            #     --> rows where pred == "none" and gold == "NaN" 
            dfm = dfm[~ (pd.isna(dfm["type_gold"]) & (dfm['type_pred'] == 'none'))].copy()
            self.logger.debug(f"dfm shape after rm none pred with nan gold: {dfm.shape}") 
            #! REMOVE AFTER DEBUGGING
            dfm.to_csv(f"../logs/debugging/{datetime.today().strftime('%Y-%m-%d-%H-%M-%S-%f')}_test_df_1.tsv", sep='\t', encoding='utf-8')


            # Check if the role of the entities in the corresponding relation is the same
            dfm['type_pred'] = dfm.apply(lambda row: "none" if not pd.isna(row['id_gold']) and ((row['ent_1_role_gold'] != row['ent_1_role_pred']) or (row['ent_2_role_gold'] != row['ent_2_role_pred'])) else row['type_pred'], axis = 1)

            #! REMOVE AFTER DEBUGGING
            dfm.to_csv(f"../logs/debugging/{datetime.today().strftime('%Y-%m-%d-%H-%M-%S-%f')}_test_df_2.tsv", sep='\t', encoding='utf-8')

            self.logger.debug(f"dfm head 50: {dfm.head(50)}")
            self.logger.debug(f"dfm tail 50: {dfm.tail(50)}")

            gold_labels = dfm['type_gold'].to_list()
            pred_labels = dfm['type_pred'].to_list()

            re_report = self._get_re_report(gold_labels, pred_labels)
        
        else:
            gold_labels, pred_labels = [], []
            if gold_relations:
                re_report = "***** Relation Ext. Report *****\n\nNo pred labels provided."
            elif pred_relations:
                re_report = "***** Relation Ext. Report *****\n\nNo gold labels provided."
            else:
                re_report = "***** Relation Ext. Report *****\n\nNo gold and no pred labels provided."

        return re_report, gold_labels, pred_labels

    def _write_eval_report(self, out_file, ner_report, re_report_with_gold_ner, re_report_with_pred_ner, stats):
        # Create the output path /output_path/run-id/
        output_path = f'{self.config["output"]["path"]}run-{self.run_id}/'
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        with open(f'{output_path}file-{out_file}_ner-report_run-{self.run_id}.json', 'w') as f:
                 f.write(ner_report)

        with open(f'{output_path}file-{out_file}_re-report-with-gold-entities_run-{self.run_id}.json', 'w') as f:
                 f.write(re_report_with_gold_ner)
        
        with open(f'{output_path}file-{out_file}_re-report-with-pred-entities_run-{self.run_id}.json', 'w') as f:
                 f.write(re_report_with_pred_ner)
        
        with open(f'{output_path}file-{out_file}_stats_run-{self.run_id}.json', 'w') as f:
            json.dump(stats, f, indent=4, sort_keys=True)

        self.logger.info(f"Wrote evaluation report for file: {out_file}")

    def eval(self):
        """Evaluate the pipeline.

        Raises:
            ValueError: If the input type is not supported.
        """        
        t0_all = time.time()
        if self.config["input"]["type"] == "single":
            self.logger.info("Loading single document path.")
            input_file_paths = [f'{self.config["input"]["path"]}{self.config["input"]["file"]}.{self.config["input"]["extension"]}']
        
        elif self.config["input"]["type"] == "multiple":
            self.logger.info("Loading multiple document paths.")
            input_file_paths =  glob.glob(f'{self.config["input"]["path"]}*/*')
        
        elif self.config["input"]["type"] == "load_list":
            self.logger.info("Loading document paths from list.")
            with open(f'{self.config["input"]["load_list"]}', 'r') as f:
                input_file_paths = f.readlines()
                # Clean and check the paths
            input_file_paths = [p.strip() for p in input_file_paths if p.strip()]
            # Get first 1 path
            self.logger.debug(f"Number of input files: {len(input_file_paths)}")
            self.logger.debug(f"First input file: {input_file_paths[0]}")
        else:
            self.logger.error("Input type not supported.")
            raise ValueError("Input type not supported.")
            

        ner_all_gold_labels = []
        ner_all_pred_labels = []

        re_all_gold_labels_with_gold_ner = []
        re_all_pred_labels_with_gold_ner = []

        re_all_gold_labels_with_pred_ner = []
        re_all_pred_labels_with_pred_ner = []

        stats_all = {}

        for input_file_path in input_file_paths:
            stats_single = {}
            t0_single = time.time()
            self.logger.info(f"Run ATMiner pipeline for document: {input_file_path}")

            # Load the input
            self.logger.info(f"Loading input file.")
            self.load_input(input_file_path)
            
            # ------- Set doc to the right level
            if self.doc_type == "collection":
            #     self.doc = self.doc[0]
            #     self.doc_type = "document"
                # Else the doc is already is already bconv document
                self.logger.warning("Single document collections not supported yet. Only the first document will be processed.")

            # ------- Store gold annotations

            gold_ner_tokens_and_tags = self._doc_to_entities()
            gold_entities = {}
            for e in list(self.doc[0].iter_entities()):
                # print(f"{e.id}, {e.start}, {e.end}, {e.text}, {e.metadata} ")
                gold_entities[e.id] = {"id":e.id, "start": e.start, "end": e.end, "text": e.text, "type": e.metadata["type"], "annotator":e.metadata["annotator"]}

            gold_relations_with_entities = []
            for r in list(self.doc[0].iter_relations()):
                # print(f"{r.id}, {r.type}, {r._children}, {r._children[0].refid}, {r.metadata}")
                
                gold_relations_with_entities.append({
                    "id":r.id, 
                    "type":r.type, 
                    "metadata": r.metadata, 
                    "entities": [{"entity": gold_entities[c.refid], "role":c.role} for c in r._children]
                })

            self.logger.debug(f"Number of gold entities: {len(gold_entities)}")
            self.logger.debug(f"Number of gold relations: {len(gold_relations_with_entities)}")
            self.logger.debug(f"Sample of gold entities: {list_get(list(gold_entities.items()), 0)}")
            self.logger.debug(f"Sample of gold relations: {list_get(gold_relations_with_entities,0)}")

            # ---------------------------------------------
            #           Relation with gold entities
            # ---------------------------------------------

            # ------- Delete relations 
            self.logger.debug(f"Delete gold relations")
            # Delete relation
            self.doc[0].relations = []
            self.logger.debug(f"Num doc relations after deletion: {len(list(self.doc[0].iter_relations()))}")

            # ------- Predict relations based on gold entities
            # Run the relation extraction
            self.logger.info(f"Start relation extraction based on gold entities.")
            self.relation_extraction()  
            
            # ------- Store relations based on gold entities
            pred_relations_with_gold_entities = []
            for r in list(self.doc[0].iter_relations()):
                # print(f"{r.id}, {r.type}, {r._children}, {r._children[0].refid}, {r.metadata}")
                
                pred_relations_with_gold_entities.append({
                    "id":r.id, 
                    "type":r.type, 
                    "metadata": r.metadata, 
                    "entities": [{"entity": gold_entities[c.refid], "role":c.role} for c in r._children]
                })

            self.logger.debug(f"Number of gold entities: {len(gold_entities)}")
            self.logger.debug(f"Number of pred relations with gold entities: {len(pred_relations_with_gold_entities)}")
            self.logger.debug(f"Sample of gold entities: {list_get(list(gold_entities.items()),0)}")
            # self.logger.debug(f"Sample of pred relations with gold entities: {pred_relations_with_gold_entities[0]}")
            
            # ---------------------------------------------
            #           Relation with pred entities
            # ---------------------------------------------

            # ------- Delete relations and entities


            # Delete relation
            self.logger.debug(f"Delete pred relations with gold entities...")
            self.doc[0].relations = []
            self.logger.debug(f"Num doc relations after deletion pred relation w. g. e.: {len(list(self.doc[0].iter_relations()))}")

            # Delete entities
            self.logger.debug(f"Delete gold entities...")
            for pa in self.doc[0]:
                for sent in pa:
                    sent.entities = []
            self.logger.debug(f"Num doc entities after deletion gold entities: {len(list(self.doc[0].iter_entities()))}")

            # ------- Predict entities and relations 
            
            # Run the NER
            self.logger.info(f"Start NER predictions.")
            self.ner()

            # Run the relation extraction
            self.logger.info(f"Start relation extraction based on predicted entities.")
            self.relation_extraction()  

            # ------- Store entities and relations
            pred_ner_tokens_and_tags = self._doc_to_entities()

            pred_entities = {}
            for e in list(self.doc[0].iter_entities()):
                # print(f"{e.id}, {e.start}, {e.end}, {e.text}, {e.metadata} ")
                pred_entities[e.id] = {"id":e.id, "start": e.start, "end": e.end, "text": e.text, "type": e.metadata["type"], "annotator":e.metadata["annotator"]}

            pred_relations_with_pred_entities = []
            for r in list(self.doc[0].iter_relations()):
                # print(f"{r.id}, {r.type}, {r._children}, {r._children[0].refid}, {r.metadata}")
                
                pred_relations_with_pred_entities.append({
                    "id":r.id, 
                    "type":r.type, 
                    "metadata": r.metadata, 
                    "entities": [{"entity": pred_entities[c.refid], "role":c.role} for c in r._children]
                })

            self.logger.debug(f"Number of pred entities: {len(pred_entities)}")
            self.logger.debug(f"Number of pred relations with pred entities: {len(pred_relations_with_pred_entities)}")
            self.logger.debug(f"Sample of pred entities: {list_get(list(pred_entities.items()),0)}")
            # self.logger.debug(f"Sample of pred relations with pred entities: {pred_relations_with_pred_entities[0]}")

            # ------- Eval entities
            ner_report, ner_gold_labels, ner_pred_labels = self._eval_ner(gold_ner_tokens_and_tags, pred_ner_tokens_and_tags)
            ner_all_gold_labels += ner_gold_labels
            ner_all_pred_labels += ner_pred_labels

            # ------- Eval relation based on gold entities
            re_report_with_gold_entities, re_gold_labels_with_gold_ner, re_pred_labels_with_gold_ner = self._eval_rel_ext_strict(gold_relations_with_entities, pred_relations_with_gold_entities)
            re_all_gold_labels_with_gold_ner += re_gold_labels_with_gold_ner
            re_all_pred_labels_with_gold_ner += re_pred_labels_with_gold_ner

            #re_report_with_gold_entities = self._eval_rel_ext_v2(gold_relations_with_entities, gold_relations_with_entities)
                        

            # ------- Eval relation based on predicted entities
            re_report_with_pred_entities, re_gold_labels_with_pred_ner, re_pred_labels_with_pred_ner = self._eval_rel_ext_strict(gold_relations_with_entities, pred_relations_with_pred_entities)
            re_all_gold_labels_with_pred_ner += re_gold_labels_with_pred_ner
            re_all_pred_labels_with_pred_ner += re_pred_labels_with_pred_ner


            # ------- Write eval results and report
            t1_single = time.time()
            stats_single['total_time_single'] = t1_single - t0_single

            self.logger.info(f"NER report: {ner_report}")
            self.logger.info(f"RE report with gold entities: {re_report_with_gold_entities}")
            self.logger.info(f"RE report with pred. entities: {re_report_with_pred_entities}")

            time_now = datetime.today().strftime('%Y-%m-%d-%H-%M-%S-%f')

            # with open(f'../logs/debugging/{time_now}-REPORT_STATS-{self.config["input"]["file"]}.txt', 'w') as f:
            #      json.dump(stats_single, f)
            # with open(f'../logs/debugging/{time_now}-NER_REPORT-{self.config["input"]["file"]}.txt', 'w') as f:
            #      f.write(ner_report)
            # with open(f'../logs/debugging/{time_now}-RE_REPORT_WITH_GOLD-{self.config["input"]["file"]}.txt', 'w') as f:
            #      f.write(re_report_with_gold_entities)
            # with open(f'../logs/debugging/{time_now}-NER_REPORT_WITH_PRED-{self.config["input"]["file"]}.txt', 'w') as f:
            #      f.write(re_report_with_pred_entities)

            self._write_eval_report(input_file_path.split("/")[-1], 
                                ner_report, 
                                re_report_with_gold_entities, 
                                re_report_with_pred_entities, 
                                stats_single)  
            
            # Write the output 
            self.logger.info(f"Writing eval predicted articles to output file...")
            self.write_output(input_file_path)  
        
        # Write the used configuration to file
        self.write_config()

        # Final report
        t1_all = time.time()
        stats_all['total_time_all'] = t1_all - t0_all

        ner_all_report = self._get_ner_report(ner_all_gold_labels, ner_all_pred_labels)
        re_all_report_with_gold_ner = self._get_re_report(re_all_gold_labels_with_gold_ner, re_all_pred_labels_with_gold_ner)
        re_all_report_with_pred_ner = self._get_re_report(re_all_gold_labels_with_pred_ner, re_all_pred_labels_with_pred_ner)
        
        self._write_eval_report("all", 
                                ner_all_report, 
                                re_all_report_with_gold_ner, 
                                re_all_report_with_pred_ner, 
                                stats_all)    
