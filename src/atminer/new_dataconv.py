import itertools


RELATIONSHIPS = {
    "hasTrait" : [{"Arthropod", "Trait"}],
    "hasValue" : [{"Trait", "Value"}],
    "hasContinuation": [{"Arthropod", "Arthropod"},{"Trait", "Trait"}, {"Value", "Value"}, {"Qualifier", "Qualifier"}],
    "hasQualifier": [{"Qualifier", "Arthropod"}]
}

RELATIONSHIPS_LIST = sum(RELATIONSHIPS.values(), [])

class DataConverter():
    def __init__(self, logger, context_size=None):
        
        self.logger = logger
        self.context_size = context_size
        
    def _get_luke_relation(self, relation_type, text, head_start, head_end, tail_start, tail_end, 
        head_type, head_id, tail_type, tail_id, context_start_char, context_end_char, head_text, tail_text):
        """Creates a relation formatted for the LUKE model.
        """
        relation = dict()
        relation["text"] = text                  
        relation['head'] = [head_start, head_end]
        relation['tail'] = [tail_start, tail_end]
        relation['relation'] = relation_type

        relation['head_type'] = head_type
        relation['head_id'] = head_id

        relation['tail_type'] = tail_type
        relation['tail_id'] = tail_id

        relation['context_start_char'] = context_start_char
        relation['context_end_char'] = context_end_char

        relation['head_text'] = head_text
        relation['tail_text'] = tail_text
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
            'entities': [ {"id": e.id, "start_char": e.start, "end_char": e.end, "type": e.metadata["name"], "text":e.text } for sent in ngram_sents for e in list(sent.iter_entities())],
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
        if len(doc.text) >= 1000000:
            print(f"ERROR: Document to long: len = {len(doc.text)}")
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

                    luke_input = self._get_luke_relation( 
                        "undefined", 
                        context["text"] , 
                        entity_tuple[0]['start_char']-context['start_char'], 
                        entity_tuple[0]['end_char']-context['start_char'],

                        entity_tuple[1]['start_char']-context['start_char'], 
                        entity_tuple[1]['end_char']-context['start_char'],
                        
                        entity_tuple[0]["type"],
                        entity_tuple[0]['id'],
                        entity_tuple[1]["type"],
                        entity_tuple[1]['id'],
                        
                        context['start_char'] ,
                        context['end_char'],

                        entity_tuple[0]["text"],
                        entity_tuple[1]["text"],
                        )
                    luke_input_data.append(luke_input)
                
        return luke_input_data
