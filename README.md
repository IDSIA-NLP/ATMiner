# Arthropod-Trait-Miner

Arthropods traits prediction pipeline.

**Table of Content:**
- [Arthropod-Trait-Miner](#arthropod-trait-miner)
  - [Setup](#setup)
  - [Run](#run)
    - [Configuration](#configuration)
    - [Run](#run-1)
  - [Pipeline Description](#pipeline-description)
    - [Repository structure](#repository-structure)
    - [Classes](#classes)
      - [ATMiner Class](#atminer-class)
      - [EntityRecognizer Class](#entityrecognizer-class)
      - [RelationExtractor Class](#relationextractor-class)
      - [DataConverter Class](#dataconverter-class)
    - [Configuration Details](#configuration-details)
  - [Documentation](#documentation)
    - [Sphinx](#sphinx)
      - [Update Documentation](#update-documentation)
      - [Serve the Spinx documentation locally](#serve-the-spinx-documentation-locally)

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
|-- src						    # Contains all the ATMiner scripts	
|-- oger_service			    # Contains scripts for the OGER NER model
	|-- atminer			    	# Contains all the ATMiner scripts	
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
  - *Path: `./data/tmp/input/`*
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


## Documentation
### Sphinx
We use sphinx for the detailed documentation of the ATMiner pipeline.
The documentation in Markdown format can be found in the `./doc` folder. 

#### Update Documentation
After updating the docstrings or documentation in the `./doc` folder run the following commands to update the Spinx documentation.
Change to the root folder of this repository and run: `sphinx-apidoc -f -o docs/source ./src/atminer` .
This will add update `.rst` files in the `./doc/build/` folder.

To convert the `.rst` files to `.md` files run: `rst2myst convert docs/**/*.rst`
Now, you can delete all the `.rst` files in the `./doc/build/` folder.

#### Serve the Spinx documentation locally
Change to the root folder of this repository and run: ` sphinx-autobuild docs/source docs/build/html`

Alternatively, switch to the `./doc/build/home` folder and run: `python3 -m http.server 8000` 