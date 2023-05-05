import itertools


RELATIONSHIPS = {
    "hasTrait" : [{"Arthropod", "Trait"}],
    "hasValue" : [{"Trait", "Value"}],
    "hasContinuation": [{"Arthropod", "Arthropod"},{"Trait", "Trait"}, {"Value", "Value"}, {"Qualifier", "Qualifier"}],
    "hasQualifier": [{"Qualifier", "Arthropod"}]
}

RELATIONSHIPS_TRAIT_VALUES = {
    "hasTrait" : [{"Arthropod", "Trait"}],
    "hasValue" : [{"Trait", "Value"}],
}

RELATIONSHIPS_LIST = sum(RELATIONSHIPS_TRAIT_VALUES.values(), [])


class DataConverter():
    def __init__(self, logger, context_size=None):
        
        self.logger = logger
        self.context_size = context_size
        
    def _get_luke_relation(self, relation_type=None, text=None, head_start=None, head_end=None, tail_start=None, tail_end=None, 
        head_type=None, head_id=None, tail_type=None, tail_id=None, context_start_char=None, context_end_char=None, head_text=None, tail_text=None):
        """Creates a relation formatted for the LUKE model.
        """
        
        relation = dict()
        relation["sentence"] = text
        relation['relation'] = relation_type

        relation['context_start_char'] = context_start_char
        relation['context_end_char'] = context_end_char

        # Sort so that head is first entity in text
        if head_start < tail_start:                  
            relation['head'] = [head_start, head_end]
            relation['tail'] = [tail_start, tail_end]
            relation['head_type'] = head_type
            relation['head_id'] = head_id

            relation['tail_type'] = tail_type
            relation['tail_id'] = tail_id
            
            relation['head_text'] = head_text
            relation['tail_text'] = tail_text

        else:
            relation['tail'] = [head_start, head_end]
            relation['head'] = [tail_start, tail_end]
            relation['tail_type'] = head_type
            relation['tail_id'] = head_id

            relation['head_type'] = tail_type
            relation['head_id'] = tail_id
            
            relation['tail_text'] = head_text
            relation['head_text'] = tail_text

        

        
        return relation


    def _create_n_grams(self, list_obj, n=3):
        """Creates a list of n-grams.
        """
        if len(list_obj) < n:
            return [list_obj]
        return [list_obj[i:i+n] for i in range(len(list_obj)-n+1)]


    def _get_contexts(self, doc, context_size):
        """Get sliding contexts windows with **context_size** number of sentences. 

        Args:
            doc (bconv.doc.document.Document): Instance of bconv Document.
            context_size (int): The number of sentences a context window should contain.

        Returns:
            list: A list of list of context. A context contains n consecutive sentences. 
        """
        sents = []
        
        for section in doc:
            for sentence in section:
                sents.append(sentence)

        ngrams_sents = self._create_n_grams(sents, n=context_size)

        contexts = [{
            'start_char': ngram_sents[0]._start, 
            'end_char': ngram_sents[-1]._end, 
            'text': "".join([sent.text for sent in ngram_sents]),
            #! Change to e.metadata["ent_type"] for use with ATMiner
            'entities': [ {"id": e.id, "start_char": e.start, "end_char": e.end, "type": e.metadata["type"], "text":e.text } for sent in ngram_sents for e in list(sent.iter_entities())],
        } for ngram_sents in ngrams_sents]

        return contexts


    #!
    #! Change to generator and yield a single new luke datapoint
    #!

    def to_luke(self, doc):
        """Get potential relations in LUKE format. A relation exists between two annotations in the same context window.

        Args:
            passage (dictionary): The BioC passage dictionary.

        Returns:
            list: List of dictionaries of potential relations in LUKE format.
        """
        # Maximum length of documents

        self.logger.debug(f"RELATIONSHIPS_LIST: {RELATIONSHIPS_LIST}")
        
        if len(doc.text) >= 1000000:
            self.logger.error(f"ERROR: Document to long: len = {len(doc.text)}")
            return [], 0, 0

        contexts = self._get_contexts(doc, self.context_size)


        luke_input_data = []
        for context in contexts:

            for entity_tuple in itertools.combinations(context["entities"], 2):

                # Entities = the combination of e.g. Arthropod and Trait
                # O = Head , 1 = Tail but the order doesn't matter (just for clarity)
                ent_0_type = entity_tuple[0]["type"].split("-")[0] 
                ent_1_type = entity_tuple[1]["type"].split("-")[0]

                if any({ent_0_type, ent_1_type} == rel for rel in RELATIONSHIPS_LIST):
                    
                    # sanity check
                    start_ent_0 = entity_tuple[0]['start_char']-context['start_char']
                    end_ent_0 = entity_tuple[0]['end_char']-context['start_char']
                    start_ent_1 = entity_tuple[1]['start_char']-context['start_char']
                    end_ent_1 = entity_tuple[1]['end_char']-context['start_char']

                    assert entity_tuple[0]['text'] == context['text'][start_ent_0:end_ent_0], f"Entity 0 text doesn't match offsets: {entity_tuple[0]['text']} != {context['text'][start_ent_0:end_ent_0]}"
                    assert entity_tuple[1]['text'] == context['text'][start_ent_1:end_ent_1], f"Entity 1 text doesn't match offsets: {entity_tuple[1]['text']} != {context['text'][start_ent_1:end_ent_1]}"
                   
                    
                    luke_input = self._get_luke_relation( 
                        relation_type="undefined", 
                        text=context["text"] , 

                        head_start=start_ent_0,
                        head_end=end_ent_0,
                        tail_start=start_ent_1, 
                        tail_end=end_ent_1,

                        head_type=entity_tuple[0]["type"],
                        head_id=entity_tuple[0]['id'],
                        tail_type=entity_tuple[1]["type"],
                        tail_id=entity_tuple[1]['id'],

                        context_start_char=context['start_char'] ,
                        context_end_char=context['end_char'],

                        head_text=entity_tuple[0]["text"],
                        tail_text=entity_tuple[1]["text"],
                        )
                    luke_input_data.append(luke_input)
                
        return luke_input_data
