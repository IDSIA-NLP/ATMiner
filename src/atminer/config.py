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

import yaml

# --------------------------------------------------------------------------------------------
#                                            MAIN
# --------------------------------------------------------------------------------------------

class Config(object):

    def __init__(self, config_file_path="./config.yaml"):

        with open(config_file_path, 'r') as stream:
            conf = yaml.safe_load(stream)

        self._config = conf # set it to conf

    def get(self, property_name):
        """_summary_

        Args:
            property_name (_type_): _description_

        Returns:
            _type_: _description_
        """
        if property_name not in self._config.keys(): # we don't want KeyError
            return None  # just return None if not found
        return self._config[property_name]

    def __call__(self):
        return self._config

if __name__ == '__main__':

    # Setup logger
    #logger.add("../../logs/dataset_builder.log", rotation="5 MB")
    #logger.info(f'Start ...')

   # Run main
    _config = Config()
    #main(logger)
    print(_config())
    print(_config.get("nlp"))

    print(_config()["nlp"]["name"])