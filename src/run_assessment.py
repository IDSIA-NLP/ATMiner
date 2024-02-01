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
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])


def nen_vs_termlists():
    # Assessing Entity Normalization with the Taxon and Trait Dictionaries
    # Goals
    # 1. For each entity determine how many of the listed entities in the termlist are found 
    #     in on of the automated documents. 
    # 2. For all of the found entities determine the frequence counts of the specific term 
    #     found and the number of variations of the term 
    #     (e.g. feeds, feeding, feed --> 3 variations found)

    logger.info(f'Assessing Entity Normalization with the Taxon and Trait Dictionaries ...')

    # Load all predicted output documents
    file_path = "../data/tmp/output_test/run-20230517-190035054078/"
    run_id = file_path.split("/")[-2]

    logger.debug(f'Load all predicted output documents from {file_path} ...')

    # Load all predicted output documents
    output_files = {}
    for filename in glob.glob(file_path + "*.bioc.json"):
        output_files[filename.split("/")[-1].split(".")[0]] = bconv.load(filename, fmt="bioc_json", byte_offsets=False)

    logger.info(f'Loaded {len(output_files)} output files')

    # Load all entities from predicted output documents
    all_entities = []
    for filename, element in output_files.items():
        logger.info(f'key: {filename} value: {element}')

        if type(element) == bconv.doc.document.Collection:
            doc = element[0]     
        elif type(element) == bconv.doc.document.Document:
            doc = element
        else:
            raise TypeError("Document type not supported")

        # "type": "Arthropod",
        # "preferred_form": "Longitarsus",
        # "resource": "CoL",
        # "native_id": "62Y2N",
        # "cui": "CUI-less",
        # "annotator": "base_transformer-v1.0.1",
        # "probability_score": 0.7465987205505371,
        # "nen_annotator": "oger-v1.0.1"
        for e in doc.iter_entities():
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
            
            all_entities.append(extracted_entity)
            
    # Create dataframe from all entities
    df_entities = pd.DataFrame(all_entities)
    df_entities = df_entities.fillna(value=np.nan)

    logger.debug(f'DF_Entities: Loaded {len(df_entities)} entities')
    logger.debug(f'DF_Entities: Columns: {df_entities.columns}')
    logger.debug(f'DF_Entities: Head: {df_entities.head()}')

    # Most frequent entities (top 10)
    logger.debug(f'DF_Entities: Most frequent entities (top 10)')
    logger.debug(f'{df_entities["text"].value_counts().head(10)}')

    # Write frequency counts in numbers and percentage to file per entity type
    logger.debug(f'DF_Entities: Write frequency counts in numbers and percentage to file per entity type')
    # Create output path if not exists
    output_path = f"../data/assessments/{run_id}/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    for entity_type in df_entities["type"].unique():
        df_entities_type = df_entities[df_entities["type"] == entity_type]
        # Join percentage and counts
        df_entities_type["text"].value_counts().to_frame().join(df_entities_type["text"].value_counts(normalize=True).to_frame(), lsuffix='_count', rsuffix='_percent').to_csv(f"{output_path}counts_{entity_type}.csv")
 

    # Percent of entities with type "Arthropod" that don't have a native_id
    logger.debug(f'DF_Entities: Percent of entities with type "Arthropod" that don\'t have a native_id')
    

    non_termlist_arthropods = df_entities[(df_entities["type"] == "Arthropod") & (df_entities["native_id"].isna())]
    percent_non_termlist_arthropods = len(non_termlist_arthropods) / len(df_entities[df_entities["type"] == "Arthropod"])
    logger.debug(f'{percent_non_termlist_arthropods}')

    # Write for each entity type the percentage and number of entities that don't have a native_id
    logger.debug(f'DF_Entities: Write for each entity type the percentage and number of entities with a native_id and the number of entities without a native_id')
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
    with open(f"{output_path}non_termlist_entities_report.json", "w") as f:
        json.dump(native_id_report, f, indent=4)


    # Get all text of entities with the same native_id and entity type and write to file
    logger.debug(f'DF_Entities: Get all text of entities with the same native_id and entity type and write to file')
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
    with open(f"{output_path}name_variety_per_type_and_native_id.json", "w") as f:
        json.dump(native_id_entity_type_report, f, indent=4)

    # Create historpgram for 50 native_ids type with most entities variety and write to file for each entity
    logger.debug(f'DF_Entities: Create historpgram for 50 native_ids type with most entities variety and write to file for each entity')
    
    for entity_type in df_entities["type"].unique():
        df_entities_type = df_entities[df_entities["type"] == entity_type]
        native_id_entity_type_counts_sorted = {k: v for k, v in sorted(native_id_entity_type_counts[entity_type].items(), key=lambda item: item[1], reverse=True)}
        native_id_entity_type_counts_sorted_top50 = dict(list(native_id_entity_type_counts_sorted.items())[0:50])
        plt.bar(native_id_entity_type_counts_sorted_top50.keys(), native_id_entity_type_counts_sorted_top50.values())
        plt.xticks(rotation=90)
        plt.savefig(f"{output_path}name_variety_per_type_and_native_id_{entity_type}.png")
        plt.close()

    #  --------------------------------------------------------------------------------------
    #  -------------------- Compare entities with termlists ---------------------------------
    #  --------------------------------------------------------------------------------------
    # Load orignal termlists
    arthropod_termlist = pd.read_csv("../data/resources/termlists/col_arthropods.tsv", sep="\t")
    traits_feeding_termlist = pd.read_csv("../data/resources/termlists/traits_feeding.normalized.tsv", sep="\t")
    traits_habitat_termlist = pd.read_csv("../data/resources/termlists/traits_habitat.normalized.tsv", sep="\t")
    traits_morphology_termlist = pd.read_csv("../data/resources/termlists/traits_morphology.normalized.tsv", sep="\t")
            
    # Get percentage of arthropod termlist that is found entities with entity type "Arthropod" 
    logger.debug(f'DF_Entities: Get percentage of arthropod termlist that is found entities with type "Arthropod"')
    unique_arthropod_termlist_original_ids =  arthropod_termlist["original_id"].unique()
    arthropod_termlist_found = df_entities[df_entities["type"] == "Arthropod"][df_entities["native_id"].isin(unique_arthropod_termlist_original_ids)]
    percent_arthropod_termlist_found = len(arthropod_termlist_found["native_id"].unique()) / len(unique_arthropod_termlist_original_ids)
    logger.debug(f'Percent of arthropod termlist that is found entities with type "Arthropod": {percent_arthropod_termlist_found}')

    # Get percentage of traits_feeding termlist that is found entities with entity type "Trait"
    logger.debug(f'DF_Entities: Get percentage of traits_feeding termlist that is found entities with type "Trait"')
    unique_traits_feeding_termlist_original_ids =  traits_feeding_termlist["original_id"].unique()
    traits_feeding_termlist_found = df_entities[df_entities["type"] == "Trait"][df_entities["native_id"].isin(unique_traits_feeding_termlist_original_ids)]
    percent_traits_feeding_termlist_found = len(traits_feeding_termlist_found["native_id"].unique()) / len(unique_traits_feeding_termlist_original_ids)
    logger.debug(f'Percent of traits_feeding termlist that is found entities with type "Trait": {percent_traits_feeding_termlist_found}')

    # Get percentage of traits_habitat termlist that is found entities with entity type "Trait"
    logger.debug(f'DF_Entities: Get percentage of traits_habitat termlist that is found entities with type "Trait"')
    unique_traits_habitat_termlist_original_ids =  traits_habitat_termlist["original_id"].unique()
    traits_habitat_termlist_found = df_entities[df_entities["type"] == "Trait"][df_entities["native_id"].isin(unique_traits_habitat_termlist_original_ids)]
    percent_traits_habitat_termlist_found = len(traits_habitat_termlist_found["native_id"].unique()) / len(unique_traits_habitat_termlist_original_ids)
    logger.debug(f'Percent of traits_habitat termlist that is found entities with type "Trait": {percent_traits_habitat_termlist_found}')

    # Get percentage of traits_morphology termlist that is found entities with entity type "Trait"
    logger.debug(f'DF_Entities: Get percentage of traits_morphology termlist that is found entities with type "Trait"')
    unique_traits_morphology_termlist_original_ids =  traits_morphology_termlist["original_id"].unique()
    traits_morphology_termlist_found = df_entities[df_entities["type"] == "Trait"][df_entities["native_id"].isin(unique_traits_morphology_termlist_original_ids)]
    percent_traits_morphology_termlist_found = len(traits_morphology_termlist_found["native_id"].unique()) / len(unique_traits_morphology_termlist_original_ids)
    logger.debug(f'Percent of traits_morphology termlist that is found entities with type "Trait": {percent_traits_morphology_termlist_found}')

   
    # Alternative plot with normalized stacked bar chart with annotation amount of termlist found compared to original termlist size
    logger.debug(f'DF_Entities: Alternative plot with normalized stacked bar chart with annotation amount of termlist found compared to original termlist size')
    plt.bar(["Arthropod", "Feeding", "Habitat", "Morphology"],
            [
                len(unique_arthropod_termlist_original_ids) / len(unique_arthropod_termlist_original_ids), 
                len(unique_traits_feeding_termlist_original_ids) / len(unique_traits_feeding_termlist_original_ids), 
                len(unique_traits_habitat_termlist_original_ids) / len(unique_traits_habitat_termlist_original_ids), 
                len(unique_traits_morphology_termlist_original_ids) / len(unique_traits_morphology_termlist_original_ids)
            ],
            label="Original")
    plt.bar(["Arthropod", "Feeding", "Habitat", "Morphology"], 
            [
                len(arthropod_termlist_found["native_id"].unique()) / len(unique_arthropod_termlist_original_ids), 
                len(traits_feeding_termlist_found["native_id"].unique()) / len(unique_traits_feeding_termlist_original_ids), 
                len(traits_habitat_termlist_found["native_id"].unique()) / len(unique_traits_habitat_termlist_original_ids), 
                len(traits_morphology_termlist_found["native_id"].unique()) / len(unique_traits_morphology_termlist_original_ids)
            ],
            label="Found")
    plt.xticks(fontsize=8)
    plt.legend()
    for i, v in enumerate([
                len(unique_arthropod_termlist_original_ids), 
                len(unique_traits_feeding_termlist_original_ids), 
                len(unique_traits_habitat_termlist_original_ids), 
                len(unique_traits_morphology_termlist_original_ids)
            ]):
        plt.text(i-0.2, 1.05 , f"n={human_format(v)}", color='black', fontweight='bold')
    for i, v in enumerate([
                len(arthropod_termlist_found["native_id"].unique()), 
                len(traits_feeding_termlist_found["native_id"].unique()), 
                len(traits_habitat_termlist_found["native_id"].unique()), 
                len(traits_morphology_termlist_found["native_id"].unique())
            ]):
        plt.text(i-0.2, -0.05, f"k={human_format(v)}", color='black', fontweight='bold')

    plt.ylim(-0.1, 1.1)



    plt.title("Termlist found vs original")
    plt.savefig(f"{output_path}termlist_found_vs_original_normalized.png")
    plt.close()


    # Plot with normalized stacked bar chart with annotation amount of termlist found compared to original entities size
    logger.debug(f'DF_Entities: Plot with normalized stacked bar chart with annotation amount of termlist found compared to original entities size')    
    plt.bar(["Arthropod", "Feeding", "Habitat", "Morphology"],
            [
                len(df_entities[df_entities["type"] == "Arthropod"]) / len(df_entities[df_entities["type"] == "Arthropod"]), 
                len(df_entities[df_entities["type"] == "Trait"]) / len(df_entities[df_entities["type"] == "Trait"]), 
                len(df_entities[df_entities["type"] == "Trait"]) / len(df_entities[df_entities["type"] == "Trait"]), 
                len(df_entities[df_entities["type"] == "Trait"]) / len(df_entities[df_entities["type"] == "Trait"])
            ],
            label="Original")
    plt.bar(["Arthropod", "Feeding", "Habitat", "Morphology"], 
            [
                len(arthropod_termlist_found) / len(df_entities[df_entities["type"] == "Arthropod"]), 
                len(traits_feeding_termlist_found) / len(df_entities[df_entities["type"] == "Trait"]), 
                len(traits_habitat_termlist_found) / len(df_entities[df_entities["type"] == "Trait"]), 
                len(traits_morphology_termlist_found) / len(df_entities[df_entities["type"] == "Trait"])
            ],
            label="Found")
    plt.xticks(fontsize=8)
    plt.legend()
    for i, v in enumerate([
                len(df_entities[df_entities["type"] == "Arthropod"]), 
                len(df_entities[df_entities["type"] == "Trait"]), 
                len(df_entities[df_entities["type"] == "Trait"]), 
                len(df_entities[df_entities["type"] == "Trait"])
            ]):
        plt.text(i-0.2, 1.05 , f"n={human_format(v)}", color='black', fontweight='bold')
    for i, v in enumerate([
                len(arthropod_termlist_found), 
                len(traits_feeding_termlist_found), 
                len(traits_habitat_termlist_found), 
                len(traits_morphology_termlist_found)
            ]):
        plt.text(i-0.2, -0.05, f"k={human_format(v)}", color='black', fontweight='bold')

    plt.ylim(-0.1, 1.1)



    plt.title("Entities with concept id vs all annotated entities")
    plt.savefig(f"{output_path}entities_with_concept_id_vs_all_annotated_entities_normalized.png")
    plt.close()

    
def main(logger):
    """ Get the configuration and run the ATMiner pipeline.

    Args:
        logger (logger instance): Instance of the loguru logger

    Returns:
        None
    """

    logger.info(f'Start assessments ...')

    # Assessing Entity Normalization with the Taxon and Trait Dictionaries
    nen_vs_termlists()
    

    # miner = ATMiner( 
    #     _config(), 
    #     logger)

    # miner.run()



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