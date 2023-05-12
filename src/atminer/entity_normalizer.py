
"""
Description: ...
Author: ...
Month Year
"""

from oger.ctrl.router import Router, PipelineServer
from io import StringIO

class EntityNormalizer(object):
    """ A class to represent an Entity Normalizer.

    Args:
        model_name (str, optional): The name of the model to use. Defaults to "oger".
        model_version (str, optional): The version of the model to use. Defaults to None.
        model_config (dict, optional): A dictionary containing the configuration of the Entity Recognizer. Defaults to None.
        logger (loguru.logger, optional): A logger object. Defaults to None.
    """
    def __init__(self, model_name="oger", model_version=None, model_config=None, merge_strategy=None, logger=None):
        
        self.model_name = model_name
        self.model_version = model_version
        self.model_config = model_config
        self.merge_strategy = model_config['merge_strategy']

        self.logger = logger
        
        if self.model_name == "oger":
            self._init_oger_pipeline()
        else:
            raise NotImplementedError(f"Entity Normalizer {self.model_name} not implemented.")



    def _init_oger_pipeline(self):
        """Initialize the OGER pipeline."""

        conf = Router(settings=self.model_config["settings_path"])
        # Initiziate oger pipline
        self.oger_pipeline = PipelineServer(conf, lazy=True)
        self.logger.debug(f"OGER PipelineServer conf: {vars(self.oger_pipeline._conf)}")


    def _predict_with_oger(self, text, ent_type):
        """Create the NER prediction with OGER.

        Args:
            text (str): The text to predict entities for.

        Returns:
            list: A list of ATEntity objects.       
        """        
        doc = self.oger_pipeline.load_one(StringIO(text), 'txt')

        self.oger_pipeline.process(doc)
        self.oger_pipeline.postfilter(doc)

        if self.merge_strategy == "first":
            ent_list = list(doc.iter_entities())
            if ent_list:
                ent = ent_list[0]

                return {
                    "text": ent.text,
                    "offsets": [(ent.start,ent.end)],
                    "type": ent.type,
                    "preferred_form": ent.pref, 
                    "resource": ent.db, 
                    "native_id": ent.cid, 
                    "cui": ent.cui,
                    "extra_info": {"annotator": f"{self.model_name}-{self.model_version}"}
                }
            else:
                return None
        
        else:
            raise NotImplementedError(f"Merge strategy {self.merge_strategy} not implemented.")

    

    def predict(self, text, ent_type):
        """Create the NEN prediction.

        Args:
            text (str): The text to predict entities for.

        Raises:
            ValueError: If the model name is not supported.

        Returns:
            list: A list of ATEntity objects.
        """
        if self.model_name == "oger":
                return self._predict_with_oger(text, ent_type)
        
    

if "__main__" == __name__:
    from loguru import logger
    import bconv
    import yaml

    class Config(object):

        def __init__(self, config_file_path="../../configs/config.yaml"):

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

    _config = Config(config_file_path="../../configs/config.yaml")
    config = _config()

    logger.add("../../logs/entity_normalizer.log", rotation="100 KB")
    ent_normalizer = EntityNormalizer(
            model_name = config['nen']['model'],
            model_version = config['nen']['version'],
            model_config = config['nen'],
            logger=logger)

    doc = bconv.load('../../data/tmp/output_test/PMC3428702.bioc.json', fmt='bioc_json', byte_offsets=False)

    for ent in doc[0].iter_entities():
        print(ent.text, ent.metadata)
        print(ent_normalizer.predict(ent.text, ent.metadata["type"]))
        print("\n\n")