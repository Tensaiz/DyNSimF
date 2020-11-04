from enum import Enum

__author__ = "Mathijs Maijer"
__email__ = "m.f.maijer@gmail.com"


class UpdateType(Enum):
    STATE = 0
    UTILITY = 1
    NETWORK = 2

class UpdateConfiguration(object):
    """
    Configuration for Updates
    TODO: Validate attributes
    """
    def __init__(self, iterable=(), **kwargs):
        self.set_default()
        self.__dict__.update(iterable, **kwargs)
        self.validate()

    def set_default(self):
        self.arguments = {}
        self.condition = None
        self.get_nodes = None
        self.update_type = UpdateType.STATE

    def validate(self):
        if not isinstance(self.update_type, UpdateType):
            raise ValueError('Update type should of enum UpdateType')


class Update(object):
    """
    Update class
    """

    def __init__(self, fun, config=None):
        self.function = fun
        self.config = config if config else UpdateConfiguration()
        self.arguments = self.config.arguments
        self.condition = self.config.condition
        self.get_nodes = self.config.get_nodes
        self.update_type = self.config.update_type

    def execute(self, nodes=None):
        if self.get_nodes:
            output = self.function(nodes, **self.arguments)
        else:
            output = self.function(**self.arguments)
        return output
