# Should be able to keep track of the different resources need and to produce automatic updates
class ResourceManager(object):
    def __init__(self,):
        self.model_name = model_name
        

# Use OGEr as the basemodel for the EntityRecognizer
class EntityRecognizer(object):
    def __init__(self, model_name="OGER"):
        self.model_name = model_name
        
    
    def predict(self):
        # predict the entities based given a text
        # return the entity offsets, lables and ids
        pass


# Use LUKE as the basemodel for the EntityRecognizer
class RelationExtractor(object):
    def __init__(self, model_name="LUKE"):
        self.model_name = model_name
        
    
    def predict(self):
        # predict the relation based given a text, the entity offsets and the entity labels
        # return the relation type
        pass


# ArthroTraitMiner to run the whole pipline with one command
class ATMiner(object):
    def __init__(self, config, logger):

        self.config = config
        self.logger = logger

        self.rel_extractor = RelationExtractor()
        self.ent_recognizer = EntityRecognizer()

        
    
    def load(self):
        # Load a plain text or XML file
        pass

    def ner(self):
        # Produce the Named Entity Recognition 
        pass

    def split(self, strategy=None):
        # Use a strategy/model to split a text into sentences of context windows 
        pass

    def relation_extraction(self):
        # Extract the relation 
        pass

    def write(self):
        # Write the results to an annoted file or produce the database outputs
        pass 

