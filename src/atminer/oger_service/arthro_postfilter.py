#!/usr/bin/env python3
# coding: utf8

# Author: Joseph Cornelius, 2021

'''
OGER postfilter for extracting annotated age mentions.
'''

from oger.doc import document
from quantulum3 import parser
import spacy

class PostEntityRecognizer:
    def __init__(self):
        super().__init__()

        self.quant_parser = parser
        self.nlp = spacy.load('en_core_web_lg')
    

    def _get_quants_candidates(self, text):
        quants = self.quant_parser.parse(text)
        candidates = []
        for quant in quants:
            entry = [
                'Value', 
                f"{quant.value} {quant.unit.name}",
                'Quantulum',
                'QUANT0000',
                'CUI-less'
            ]
            candidates.append({
                "start": quant.span[0],
                "end": quant.span[1],
                "entry": entry
            })
        return candidates


    def _get_location_candidates(self, text):
        doc = self.nlp(text)
        candidates = []

        for ent in doc.ents:
            if (ent.label_ == 'GPE') or (ent.label_ == 'LOC'):
                entry = [
                    'Value', 
                    ent.text,
                    'SpaCy',
                    'LOC0000',
                    'CUI-less'
                ]
                candidates.append({
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "entry": entry
                })
        return candidates

    def recognize_entities(self, text):
        # entry[0] = Entity-type field
        # entry[1] = Preferred-form field
        # entry[2] = Original-resource field
        # entry[3] = Concept-ID field (defined by DB)
        # entry[4] = UMLS CUI field
        # entry[5:] = Any additional fields
        # e.g. entry = set(['Arthropod', 'Lachnodius', 'CoL', '62VND', 'CUI-less'])

        candidates_quant = self._get_quants_candidates(text)
        candidates_loc = self._get_location_candidates(text)

        candidates = candidates_quant + candidates_loc

        for candidate in candidates:
            # position = (start, end) with in the sentence
            # e.g. entry = set(['Arthropod', 'Lachnodius', 'CoL', '62VND', 'CUI-less'])
            position = (candidate['start'], candidate['end'])
            entry = candidate["entry"]

            yield position, tuple(entry)



def find_values(content):

    entity_recognizer = PostEntityRecognizer()

    for sentence in content.get_subelements(document.Sentence):
        #+ print(sentence.text, sentence.start)
        
        # Get Values 
        sentence.recognize_entities(entity_recognizer)

    #+ for entity in content.iter_entities():
    #+     print(entity.id_, entity.text, entity.start, entity.end, entity.info)

    
#! TODO Write a filter that detects all Genus names in a DOCUMENT and finds the Corresponding abbreviation 

#! TODO Write a filter that detects all Country and Location names 

# Gets called by default if not method is defined 
def postfilter(content):
    print("In postfilter")
    pass
    

if __name__ == '__main__':
    postfilter("dummy")