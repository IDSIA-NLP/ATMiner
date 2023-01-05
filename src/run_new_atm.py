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

from atminer.config import Config
from atminer.new_atminer import ATMiner

_config = Config(config_file_path="../configs/config.yaml")

# --------------------------------------------------------------------------------------------
#                                            MAIN
# --------------------------------------------------------------------------------------------


# ----------------------------------------- Functions ----------------------------------------

def main(logger):
    """ Get the configuration and run the ATMiner pipeline.

    Args:
        logger (logger instance): Instance of the loguru logger

    Returns:
        None
    """

    logger.info(f'Load and run ATMiner ...')

    miner = ATMiner( 
        _config(), 
        logger)

    miner.run()



# --------------------------------------------------------------------------------------------
#                                          RUN
# --------------------------------------------------------------------------------------------

if __name__ == '__main__':

    # Setup logger
    logger.add("../../logs/run_atm.log", rotation="1 MB")
    logger.info(f'Start ...')

   # Run main
    main(logger)