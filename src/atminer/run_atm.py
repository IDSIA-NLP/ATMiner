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

from config import Config
from atminer import ATMiner

_config = Config()

# --------------------------------------------------------------------------------------------
#                                            MAIN
# --------------------------------------------------------------------------------------------


# ----------------------------------------- Functions ----------------------------------------

def main(logger):
    """[summary]

    Args:
        logger ([type]): [description]

    Returns:
        [type]: [description]
    """

    logger.info(f'VERY IMPORTANT LOG NOTE')
    logger.info(_config.get("input_path"))

    miner = ATMiner( 
        _config(), 
        logger)

    miner.run()



# --------------------------------------------------------------------------------------------
#                                          RUN
# --------------------------------------------------------------------------------------------

if __name__ == '__main__':

    # Setup logger
    logger.add("../../logs/dataset_builder.log", rotation="1 MB")
    logger.info(f'Start ...')

   # Run main
    main(logger)