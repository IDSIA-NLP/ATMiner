#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Description: ...
Author: ...
Month Year
"""
# --------------------------------------------------------------------------------------------
#                                           IMPORT
# --------------------------------------------------------------------------------------------

from loguru import logger
import numpy as np
import spacy
import json
import time
import pandas as pd
import os
from pathlib import Path

import matplotlib.pyplot as plt

from atminer.config import Config
from atminer.atminer import ATMiner


import glob
import bconv

_config = Config(config_file_path="../configs/config.yaml")

# --------------------------------------------------------------------------------------------
#                                            MAIN
# --------------------------------------------------------------------------------------------


# ----------------------------------------- Functions ----------------------------------------
def human_format(num):
    """
    Converts a number into a human-readable format with SI prefixes.

    Args:
        num (float): The number to be converted.

    Returns:
        str: The human-readable format of the number.
    """
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])


def load_output_files(predicted_output_path):
    """
    Load all predicted output documents from the specified path.
    
    Args:
        predicted_output_path (str): The path to the directory containing the predicted output documents.
        
    Returns:
        dict: A dictionary where the keys are the stem of the file names and the values are the loaded documents.
    """
    logger.debug(f'Load all predicted output documents from {predicted_output_path} ...')
    file_path = Path(predicted_output_path)
    return {f.stem: bconv.load(f, fmt="bioc_json", byte_offsets=False) for f in file_path.glob("*.bioc.json")}


def write_to_json(file_path, data):
    """
    Write data to a json file.

    Args:
        file_path (str): The path to the file to be written.
        data (dict): The data to be written.

    Returns:
        None
    """    
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def create_and_save_bar_plot(data, entity_type, output_path):
    """
    Create and save a bar plot for the 50 native_ids with the most entities variety.

    Args:
        data (dict): The data to be plotted.
        entity_type (str): The entity type.
        output_path (str): The path to the directory where the plot should be saved.

    Returns:
        None
    """
    data_sorted = {k: v for k, v in sorted(data.items(), key=lambda item: item[1], reverse=True)}
    data_sorted_top50 = dict(list(data_sorted.items())[0:50])
    plt.bar(data_sorted_top50.keys(), data_sorted_top50.values())
    plt.xticks(rotation=90)
    plt.savefig(f"{output_path}name_variety_per_type_and_native_id_{entity_type}.png")
    plt.close()


def create_and_save_normalized_bar_plot(categories, original_data, found_data, original_labels, found_labels, title, output_path):
    """
    Create and save a normalized bar plot.

    Args:
        categories (list): The categories.
        original_data (list): The original data.
        found_data (list): The found data.
        original_labels (list): The original labels.
        found_labels (list): The found labels.
        title (str): The title of the plot.
        output_path (str): The path to the directory where the plot should be saved.

    Returns:    
        None
    """
    plt.bar(categories, original_data, label="Original")
    plt.bar(categories, found_data, label="Found")
    plt.xticks(fontsize=8)
    plt.legend()
    for i, v in enumerate(original_labels):
        plt.text(i-0.2, 1.05 , f"n={human_format(v)}", color='black', fontweight='bold')
    for i, v in enumerate(found_labels):
        plt.text(i-0.2, -0.05, f"k={human_format(v)}", color='black', fontweight='bold')
    plt.ylim(-0.1, 1.1)
    plt.title(title)
    plt.savefig(f"{output_path}{title.replace(' ', '_')}_normalized.png")
    plt.close()


def create_entity_dict(e):
    """
    Create a dictionary from an entity.

    Args:
        e (bconv.doc.entity.Entity): The entity to be converted.

    Returns:
        dict: The entity as a dictionary.
    """
    extracted_entity = {   
                "id":e.id, 
                "text": e.text, 
                "type": e.metadata["type"], 
                "preferred_form": e.metadata["preferred_form"] if e.metadata["preferred_form"] != "" else None,
                "resource": e.metadata["resource"] if e.metadata["resource"] != "" else None,
                "native_id": e.metadata["native_id"] if e.metadata["native_id"] != "" else None,
                "cui": e.metadata["cui"] if e.metadata["cui"] != "" else None,
                "annotator":e.metadata["annotator"],
            }
    if "nen_annotator" in e.metadata:
        extracted_entity["nen_annotator"] = e.metadata["nen_annotator"]
    return extracted_entity


def create_output_path(output_dir, run_id):
    """
    Create the output path for the assessment.

    Args:
        output_dir (str): The path to the directory where the output should be saved.
        run_id (str): The run id.

    Returns:
        str: The output path.
    """
    output_path = f"{output_dir}{run_id}/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    return output_path


def create_dataframe_from_entities(output_files):
    """
    Create a dataframe from all entities.

    Args:
        output_files (dict): The output files.

    Returns:
        pd.DataFrame: The dataframe containing all entities.
    """
    all_entities = []
    for filename, element in output_files.items():
        if isinstance(element, bconv.doc.document.Collection):
            doc = element[0]     
        elif isinstance(element, bconv.doc.document.Document):
            doc = element
        else:
            raise TypeError("Document type not supported")
        
        for e in doc.iter_entities():
            extracted_entity = create_entity_dict(e)
            
            all_entities.append(extracted_entity)
    
    # Create dataframe from all entities
    df_entities = pd.DataFrame(all_entities)
    df_entities = df_entities.fillna(value=np.nan)
    return df_entities


def write_native_id_entity_type_report(df_entities, output_path):
    """
    Write for each entity type the number of entities variety and the number of entities per native_id.

    Args:
        df_entities (pd.DataFrame): The dataframe containing all entities.
        output_path (str): The path to the directory where the output should be saved.

    Returns:
        dict: The number of entities per native_id.
    """

    native_id_entity_type_report = {}
    native_id_entity_type_counts = {}
    for entity_type in df_entities["type"].unique():
        df_entities_type = df_entities[df_entities["type"] == entity_type]
        native_id_entity_type_report[entity_type] = {}
        native_id_entity_type_counts[entity_type] = {}
        for native_id in df_entities_type["native_id"].dropna().unique():
            df_entities_type_native_id = df_entities_type[df_entities_type["native_id"] == native_id]
            # all lower case
            native_id_entity_type_report[entity_type][native_id] = df_entities_type_native_id["text"].str.lower().unique().tolist()
            native_id_entity_type_counts[entity_type][native_id] = len(df_entities_type_native_id["text"].str.lower().unique().tolist())

    # Write native_id_entity_type_report to file
    write_to_json(f"{output_path}name_variety_per_type_and_native_id.json", native_id_entity_type_report)
    return native_id_entity_type_counts


def write_entity_native_id_report(df_entities, output_path):
    """
    Write for each entity type the percentage and number of entities that don't have a native_id.

    Args:
        df_entities (pd.DataFrame): The dataframe containing all entities.
        output_path (str): The path to the directory where the output should be saved.

    Returns:
        None
    """
    native_id_report = {}
    for entity_type in df_entities["type"].unique():
        df_entities_type = df_entities[df_entities["type"] == entity_type]
        # Join percentage and counts
        df_entities_type["native_id"].value_counts().to_frame().join(df_entities_type["native_id"].value_counts(normalize=True).to_frame(), lsuffix='_count', rsuffix='_percent').to_csv(f"{output_path}native_id_counts_{entity_type}.csv")

        non_termlist_entity_type = df_entities[(df_entities["type"] == entity_type) & (df_entities["native_id"].isna())]
        percent_non_termlist_entity_type = len(non_termlist_entity_type) / len(df_entities[df_entities["type"] == entity_type])
        native_id_report[entity_type] = {
            f"percent_non_termlist_{entity_type}": percent_non_termlist_entity_type,
            f"count_non_termlist_{entity_type}": len(non_termlist_entity_type),
            f"count_{entity_type}": len(df_entities[df_entities["type"] == entity_type])
        }

    # Write native_id_report to file
    write_to_json(f"{output_path}non_termlist_entities_report.json", native_id_report)


def write_frequency_counts(df_entities, output_path):
    """
    Write frequency counts in numbers and percentage to file per entity type.

    Args:
        df_entities (pd.DataFrame): The dataframe containing all entities.
        output_path (str): The path to the directory where the output should be saved.

    Returns:
        None
    """
    for entity_type in df_entities["type"].unique():
        df_entities_type = df_entities[df_entities["type"] == entity_type]
        # Join percentage and counts
        df_entities_type["text"].value_counts().to_frame().join(df_entities_type["text"].value_counts(normalize=True).to_frame(), lsuffix='_count', rsuffix='_percent').to_csv(f"{output_path}counts_{entity_type}.csv")


def calculate_percentage(df_entities, termlist, entity_type, termlist_name):
        """
        Calculate the percentage of termlist found.

        Args:
            df_entities (pd.DataFrame): The dataframe containing all entities.
            termlist (pd.DataFrame): The termlist.
            entity_type (str): The entity type.
            termlist_name (str): The name of the termlist.

        Returns:
            tuple: The unique original ids and the found termlist.
        """
        unique_termlist_original_ids = termlist["original_id"].unique()
        termlist_found = df_entities[(df_entities["type"] == entity_type) & (df_entities["native_id"].isin(unique_termlist_original_ids))]
        
        # Just log the percentage
        percent_termlist_found = len(termlist_found["native_id"].unique()) / len(unique_termlist_original_ids)
        logger.debug(f'Percent of {termlist_name} termlist that is found entities with type "{entity_type}": {percent_termlist_found}')
        return unique_termlist_original_ids,  termlist_found


def generate_entities_with_concept_id_plot(df_entities, 
                                           output_path, 
                                           arthropod_termlist_found, 
                                           traits_feeding_termlist_found, 
                                           traits_habitat_termlist_found, 
                                           traits_morphology_termlist_found):
    """
    Generate a plot for entities with concept id vs all annotated entities.

    Args:
        df_entities (pd.DataFrame): The dataframe containing all entities.  
        output_path (str): The path to the directory where the plot should be saved.
        arthropod_termlist_found (pd.DataFrame): The found arthropod termlist.
        traits_feeding_termlist_found (pd.DataFrame): The found feeding termlist.
        traits_habitat_termlist_found (pd.DataFrame): The found habitat termlist.
        traits_morphology_termlist_found (pd.DataFrame): The found morphology termlist.
        categories (list): The categories.

    Returns:
        None
    """
    categories = ["Arthropod", "Feeding", "Habitat", "Morphology"]

    original_data = [
        len(df_entities[df_entities["type"] == "Arthropod"]) / len(df_entities[df_entities["type"] == "Arthropod"]),
        len(df_entities[df_entities["type"] == "Trait"]) / len(df_entities[df_entities["type"] == "Trait"]),
        len(df_entities[df_entities["type"] == "Trait"]) / len(df_entities[df_entities["type"] == "Trait"]),
        len(df_entities[df_entities["type"] == "Trait"]) / len(df_entities[df_entities["type"] == "Trait"])
    ]
    found_data = [
        len(arthropod_termlist_found) / len(df_entities[df_entities["type"] == "Arthropod"]),
        len(traits_feeding_termlist_found) / len(df_entities[df_entities["type"] == "Trait"]),
        len(traits_habitat_termlist_found) / len(df_entities[df_entities["type"] == "Trait"]),
        len(traits_morphology_termlist_found) / len(df_entities[df_entities["type"] == "Trait"])
    ]   
    original_labels = [
        len(df_entities[df_entities["type"] == "Arthropod"]),
        len(df_entities[df_entities["type"] == "Trait"]),
        len(df_entities[df_entities["type"] == "Trait"]),
        len(df_entities[df_entities["type"] == "Trait"])
    ]   
    found_labels = [
        len(arthropod_termlist_found),
        len(traits_feeding_termlist_found),
        len(traits_habitat_termlist_found),
        len(traits_morphology_termlist_found)
    ]   
    create_and_save_normalized_bar_plot(categories, original_data, found_data, original_labels, found_labels, "Entities with concept id vs all annotated entities", output_path)


def generate_percentage_of_termlist_entities_plot(output_path, 
                                                  unique_arthropod_termlist_original_ids, 
                                                  arthropod_termlist_found, 
                                                  unique_traits_feeding_termlist_original_ids, 
                                                  traits_feeding_termlist_found, 
                                                  unique_traits_habitat_termlist_original_ids, 
                                                  traits_habitat_termlist_found, 
                                                  unique_traits_morphology_termlist_original_ids, 
                                                  traits_morphology_termlist_found):
    """
    Generate a plot for the percentage of termlist entities found.

    Args:
        output_path (str): The path to the directory where the plot should be saved.
        unique_arthropod_termlist_original_ids (list): The unique original ids of the arthropod termlist.
        arthropod_termlist_found (pd.DataFrame): The found arthropod termlist.
        unique_traits_feeding_termlist_original_ids (list): The unique original ids of the feeding termlist.
        traits_feeding_termlist_found (pd.DataFrame): The found feeding termlist.
        unique_traits_habitat_termlist_original_ids (list): The unique original ids of the habitat termlist.
        traits_habitat_termlist_found (pd.DataFrame): The found habitat termlist.

    Returns:
        None
    """
    categories = ["Arthropod", "Feeding", "Habitat", "Morphology"]

    original_data = [
        len(unique_arthropod_termlist_original_ids) / len(unique_arthropod_termlist_original_ids), 
        len(unique_traits_feeding_termlist_original_ids) / len(unique_traits_feeding_termlist_original_ids), 
        len(unique_traits_habitat_termlist_original_ids) / len(unique_traits_habitat_termlist_original_ids), 
        len(unique_traits_morphology_termlist_original_ids) / len(unique_traits_morphology_termlist_original_ids)
    ]
    found_data = [
        len(arthropod_termlist_found["native_id"].unique()) / len(unique_arthropod_termlist_original_ids), 
        len(traits_feeding_termlist_found["native_id"].unique()) / len(unique_traits_feeding_termlist_original_ids), 
        len(traits_habitat_termlist_found["native_id"].unique()) / len(unique_traits_habitat_termlist_original_ids), 
        len(traits_morphology_termlist_found["native_id"].unique()) / len(unique_traits_morphology_termlist_original_ids)
    ]
    original_labels = [
        len(unique_arthropod_termlist_original_ids), 
        len(unique_traits_feeding_termlist_original_ids), 
        len(unique_traits_habitat_termlist_original_ids), 
        len(unique_traits_morphology_termlist_original_ids)
    ]
    found_labels = [
        len(arthropod_termlist_found["native_id"].unique()), 
        len(traits_feeding_termlist_found["native_id"].unique()), 
        len(traits_habitat_termlist_found["native_id"].unique()), 
        len(traits_morphology_termlist_found["native_id"].unique())
    ]
    create_and_save_normalized_bar_plot(categories, original_data, found_data, original_labels, found_labels, "Termlist found vs original", output_path)



def assess_entity_normalization(predicted_output_path="../data/tmp/output_test/run-20230517-190035054078/", output_dir="../data/assessments/"):
    """
    Assessing Entity Normalization with the Taxon and Trait Dictionaries.

    Goals:
    1. For each entity determine how many of the listed entities in the termlist are found 
        in on of the automated documents.
    2. For all of the found entities determine the frequence counts of the specific term
        found and the number of variations of the term
        (e.g. feeds, feeding, feed --> 3 variations found)

    Args:
        predicted_output_path (str): The path to the directory containing the predicted output documents.
        output_dir (str): The path to the directory where the output should be saved.

    Returns:
        None
    """
    logger.info(f'Assessing Entity Normalization with the Taxon and Trait Dictionaries ...')


    # Use pathlib.Path to iterate over the files
    run_id = predicted_output_path.split("/")[-2]
    output_files = load_output_files(predicted_output_path)

    logger.info(f'Loaded {len(output_files)} output files')

    # Load all entities from predicted output documents
    df_entities = create_dataframe_from_entities(output_files)

    # Log some information about the entities
    logger.debug(f'DF_Entities: Loaded {len(df_entities)} entities')
    logger.debug(f'DF_Entities: Columns: {df_entities.columns}')
    logger.debug(f'DF_Entities: Head: {df_entities.head()}')
    logger.debug(f'DF_Entities: Most frequent entities (top 10)')
    logger.debug(f'{df_entities["text"].value_counts().head(10)}')

    # Create output path if not exists
    output_path = create_output_path(output_dir, run_id)
    
    # Write frequency counts in numbers and percentage to file per entity type
    logger.debug(f'Write frequency counts in numbers and percentage to file per entity type')
    write_frequency_counts(df_entities, output_path)
 
    # Write for each entity type the percentage and number of entities that don't have a native_id
    logger.debug(f'Write for each entity type the percentage and number of entities with a native_id and the number of entities without a native_id')
    write_entity_native_id_report(df_entities, output_path)

    # Get all text of entities with the same native_id and entity type and write to file
    logger.debug(f'Get all text of entities with the same native_id and entity type and write to file')
    native_id_entity_type_counts = write_native_id_entity_type_report(df_entities, output_path)
 
    # Create historpgram for 50 native_ids type with most entities variety and write to file for each entity
    logger.debug(f'Create historpgram for 50 native_ids type with most entities variety and write to file for each entity')
    
    for entity_type in df_entities["type"].unique():
        create_and_save_bar_plot(native_id_entity_type_counts[entity_type], entity_type, output_path)

    # -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  
    # Compare entities with termlists 
        
    # Load orignal termlists
    arthropod_termlist = pd.read_csv("../data/resources/termlists/col_arthropods.tsv", sep="\t")
    traits_feeding_termlist = pd.read_csv("../data/resources/termlists/traits_feeding.normalized.tsv", sep="\t")
    traits_habitat_termlist = pd.read_csv("../data/resources/termlists/traits_habitat.normalized.tsv", sep="\t")
    traits_morphology_termlist = pd.read_csv("../data/resources/termlists/traits_morphology.normalized.tsv", sep="\t")

    # Calculate percentage of termlist found
    unique_arthropod_termlist_original_ids, arthropod_termlist_found = calculate_percentage(df_entities, arthropod_termlist, "Arthropod", "Arthropod")
    unique_traits_feeding_termlist_original_ids, traits_feeding_termlist_found = calculate_percentage(df_entities, traits_feeding_termlist, "Trait", "Feeding")
    unique_traits_habitat_termlist_original_ids, traits_habitat_termlist_found = calculate_percentage(df_entities, traits_habitat_termlist, "Trait", "Habitat")
    unique_traits_morphology_termlist_original_ids, traits_morphology_termlist_found = calculate_percentage(df_entities, traits_morphology_termlist, "Trait", "Morphology")
   
    # Plotting termlist found vs original
    logger.debug(f'Plotting termlist found vs original')
    generate_percentage_of_termlist_entities_plot(output_path, 
                                                  unique_arthropod_termlist_original_ids, 
                                                  arthropod_termlist_found, 
                                                  unique_traits_feeding_termlist_original_ids, 
                                                  traits_feeding_termlist_found, 
                                                  unique_traits_habitat_termlist_original_ids, 
                                                  traits_habitat_termlist_found, 
                                                  unique_traits_morphology_termlist_original_ids, 
                                                  traits_morphology_termlist_found)

    # Entities with concept id vs all annotated entities
    logger.debug(f'Plotting Entities with concept id vs all annotated entities ...')

    generate_entities_with_concept_id_plot(df_entities, 
                      output_path, 
                      arthropod_termlist_found, 
                      traits_feeding_termlist_found, 
                      traits_habitat_termlist_found, 
                      traits_morphology_termlist_found)
    
def main(logger):
    """ Get the configuration and run the ATMiner pipeline.

    Args:
        logger (logger instance): Instance of the loguru logger

    Returns:
        None
    """

    logger.info(f'Start assessments ...')

    # Assessing Entity Normalization with the Taxon and Trait Dictionaries
    assess_entity_normalization()


# --------------------------------------------------------------------------------------------
#                                          RUN
# --------------------------------------------------------------------------------------------

if __name__ == '__main__':

    # Setup logger
    logger.add("../logs/assessment.log", rotation="1 MB", level=_config()["logger"]["level"])
    logger.info(f'Start ...')

    # Time the execution
    start = time.time()

    # Run main
    main(logger)

    # End timing and log in minutes
    end = time.time()
    logger.info(f'Finished in {round((end - start) / 60, 2)} minutes')