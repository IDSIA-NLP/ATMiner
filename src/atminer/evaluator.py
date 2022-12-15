from atminer import ATMiner


class Evaluator(ATMiner):
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

        self.input = None
        self.data = None    #BioC format