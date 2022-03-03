# Arthropod-Trait-Miner

Arthropods traits prediction pipeline.

## Setup

We use python 3.7 for this project.

To install the required dependencies can use `pip install -r requirements.txt`

## Run

### Configuration
   
To determine all the necessary configuration before you run the pipeline adjust the `./src/atminer/config.yaml` file.

### Run

To run the pipeline switch into the `./src/atminer` directory and run:
`python run_atm.py`.

## Pipeline Description

### Repository structure

```
|--	data						# Contains all the data need to run ATMiner
|	|-- resources				# Contains the static resources 
|	|	|-- luke_model			# Contains LUKE relation extraction model 
|	|	|-- termlists			# Contains term lists for OGER (Trait, Arth.) 
|	|-- tmp					    # Contains files created while running  ATMiner
|		|-- input
|		|-- output
|		
|		
|--	src						    # Contains all the ATMiner scripts	
|-- oger_service			    # Contains scripts for the OGER NER model
	|-- 	atminer				# Contains all the ATMiner scripts	
		|-- atminer.py			# The ATMiner , EntityRecognizer and RelExtractor class
		|-- config.py			# Loads and stores the configuration 
		|-- config.yaml		    # Configuration file
		|-- data_converter.py	# Class to converter the data into different formats
		|-- run_atm.py			# Script to run the ATMiner
```

### Classes
#### ATMiner Class
The ATMiner class determines the complete prediction pipeline.

It is also responsible for loading and writing the data in the correct format.
```python
class ATMiner(object):
    def __init__(...):
        self.rel_extractor = RelationExtractor(..)
        self.ent_recognizer = EntityRecognizer(...)

    def run(...):
        self.load()
        self.ner()
        self.rel_extraction() 
        self.write()  
```

#### EntityRecognizer Class

The EntityRecognizer class manages the different models for the named entity recognition and provides a uniform interface.

It is also responsible to ensure that we receive the data in the requested output format.

```python
class EntityRecognizer(object):
    def __init__(...):
        self.model_name = model_name

    def predict(...):
        ...
        if self.model_name == "oger":
            ...
	    return ner_predictions
```
#### RelationExtractor Class

The RelationExtractor class manages the different models for relation extraction and provides a uniform interface.

It is also responsible to ensure that we receive the data in the requested output format.

```python
class RelationExtractor(object):
    def __init__(...):
        self.model_name = model_name

    def predict(...):
        ...
        if self.model_name == "luke":
            ...
        return rel_predictions

```
#### DataConverter Class
The DataConverter class is responsible for converting the input and output data to the correct format.

We also use the class to convert the data between the NER and Relation Extraction models.
```python
class DataConverter(object):
    def __init__(...):
        self.nlp = spacy.load(nlp_model)
    
    def bioc_to_luke(...):
        return luke_data
        
    def bioc_to_tsv(...):
        return tsv_data
    
    def bioc_to_txt(...):
        return txt_data
    
    def xml_to_txt(...):
        return txt_data

```

### Configuration Details
- Input Format
  - *Path: `./data/tmp/input`*
  -   .txt
  -  BIOC JSON 		# ToDo
- Output Format
  - *Path: `./data/tmp/output/`*
  - BIOC JSON
  - TSV (relations triples)
- Entity Recognizer Model
  - [OGER](https://github.com/OntoGene/OGER)
- Relation Extraction Model
  - *Path: `./data/models/luke/`*
  - [LUKE](https://github.com/studio-ousia/luke)
- Term Lists
  - *Path: `./data/static/termlists/`*
  - COL Arthropods
  - Feeding Traits
  - Habitat Traits
  - Morphology Traits
  - ~~EOL Arthropods~~   (not used)
  - ~~EOL Traits~~		(not used)
