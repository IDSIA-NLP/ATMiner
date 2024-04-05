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

from atminer.config import Config
from atminer.atminer import ATMiner

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
    logger.remove() # Remove default logger
    logger.add("../logs/run_atm.log", rotation="1 MB", level=_config()["logger"]["level"])
    logger.info(f'Start ...')

    # Time the execution
    start = time.time()

    # Run main
    main(logger)

    # End timing and log in minutes
    end = time.time()
    logger.info(f'Finished in {round((end - start) / 60, 2)} minutes')