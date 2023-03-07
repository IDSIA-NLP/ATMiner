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
from atminer.atminer import ATMiner

_config = Config(config_file_path="../configs/config_eval.yaml")

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

    miner.eval()



# --------------------------------------------------------------------------------------------
#                                          RUN
# --------------------------------------------------------------------------------------------

if __name__ == '__main__':

    # Setup logger
    logger.add("../logs/run_atm.log", rotation="1 MB", retention=5, level=_config()["logger"]["level"])
    logger.info(f'Start ...')

   # Run main
    main(logger)