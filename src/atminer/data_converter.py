#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Description: ...
Author: ...
Month Year
"""

import itertools
import spacy

class DataConverter():
    def __init__(self, logger, nlp_type, nlp_model, context_size=None):
        
        self.logger = logger
        self.context_size = context_size
        
        if nlp_type == "spacy":
            self.nlp = spacy.load(nlp_model)
        else:
            self.logger.error("NLP model is not supported.")
            raise ValueError("NLP model is not supported.")
        

    def _context_indices(self, i, context_indices):
        #a = [[0,5],[6,13],[14,45],[46,67]] # end char
        indices = []
        for idx, c in enumerate(context_indices):
            if c[0] <= i <= c[1]:
                indices.append(idx)
        return indices


    def _start_char_anno(self, locations, psg_offset):
        s = 999999999999
        for l in locations:
            s = min(l['offset'] - psg_offset, s)
        return s


    def _end_char_anno(self, locations, psg_offset):
        e = 0
        for l in locations:
            e = max((l['offset'] + l['length']) - psg_offset, e)
        return e


    def _create_n_grams(self, list_obj, n=3):
        if len(list_obj) < n:
            return [list_obj]
        return [list_obj[i:i+n] for i in range(len(list_obj)-n+1)]


    def _get_luke_relation(self, relation_type, sentence, head_start, head_end, tail_start, tail_end, 
     head_type, head_id, tail_type, tail_id, context_start_char, context_end_char):
        relation = dict()
        relation["sentence"] = sentence                  
        relation['head'] = [head_start, head_end]
        relation['tail'] = [tail_start, tail_end]
        relation['relation'] = relation_type

        relation['head_type'] = head_type
        relation['head_id'] = head_id

        relation['tail_type'] = tail_type
        relation['tail_id'] = tail_id

        relation['context_start_char'] = context_start_char
        relation['context_end_char'] = context_end_char
        return relation

    def _get_contexts(self, text, context_size):
        doc = self.nlp(text)

        # sent_indices = [[sent.start_char, sent.end_char] for sent in doc.sents]
        # sentences = [str(sent).strip() for sent in doc.sents]

        sents = [sent for sent in doc.sents]
        ngrams_sents = self._create_n_grams(sents, n=context_size)
        contexts = [{
            'start_char': ngram_sents[0].start_char, 
            'end_char': ngram_sents[-1].end_char, 
            'text': "".join([str(sent).strip() for sent in ngram_sents]),
            'entities': [],
        } for ngram_sents in ngrams_sents]

        return contexts


    def _get_relations(self, passage):
        # Maximum length of documents
        if len(passage["text"]) >= 1000000:
            print(f"ERROR: Document to long: len = {len(passage['text'])}")
            return [], 0, 0

        contexts = self._get_contexts(passage["text"], self.context_size)
        context_indices = [[context["start_char"], context["end_char"]] for context in contexts]

        # get for each context the corresponding entities
        tmp_entities = []
        for anno in passage["annotations"]:
            anno_start_char = self._start_char_anno(anno['locations'], passage['offset'])
            anno_end_char = self._end_char_anno(anno['locations'], passage['offset'])

            anno_context_idx_start = self._context_indices(anno_start_char, context_indices)
            anno_context_idx_end = self._context_indices(anno_end_char, context_indices)
            anno_context_idx = list(set(anno_context_idx_start) & set(anno_context_idx_end))

            tmp_entities.append({
                'id': anno["id"],
                'start_char': anno_start_char,
                'end_char': anno_end_char,
                'type': anno['infons']['type']
            })
            
            for context_idx in anno_context_idx:
                contexts[context_idx]['entities'].append(anno['id'])


        relations = []
        for context in contexts:

            for ent_ids in itertools.combinations(context["entities"], 2):

                # Entities = the combination of Arthropod and Trait
                ent_0_type = tmp_entities[ent_ids[0]]["type"].split("-")[0]
                ent_1_type = tmp_entities[ent_ids[1]]["type"].split("-")[0]
                if frozenset([ent_0_type, ent_1_type]) == frozenset(["Arthropod", "Trait"]):

                    relation = self._get_luke_relation( 
                        "undefined", 
                        context["text"] , 
                        tmp_entities[ent_ids[0]]['start_char']-context['start_char'], 
                        tmp_entities[ent_ids[0]]['end_char']-context['start_char'],

                        tmp_entities[ent_ids[1]]['start_char']-context['start_char'], 
                        tmp_entities[ent_ids[1]]['end_char']-context['start_char'],
                        
                        tmp_entities[ent_ids[0]]["type"],
                        tmp_entities[ent_ids[0]]['id'],
                        tmp_entities[ent_ids[1]]["type"],
                        tmp_entities[ent_ids[1]]['id'],
                        
                        context['start_char'] + passage['offset'],
                        context['end_char'] + passage['offset']
                        )
                    relations.append(relation)
                
        return relations


    def bioc_to_luke(self, data):
        out_data = []
        for doc in data['documents']:

            relations = []
            for passage in doc["passages"]:
                relations +=  self._get_relations(passage)
                 
            out_data.append({
                "id": doc["id"],
                "relations": relations
            })
        
        #* TEST >>>>>>>>>>>>>
        import json
        with open("./test_dv.json", "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=4, sort_keys=True)
        #* <<<<<<<<<<<<<<< TEST 

        return out_data  