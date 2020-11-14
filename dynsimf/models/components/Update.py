from enum import Enum
from dynsimf.models.helpers.ConfigValidator import ConfigValidator
from dynsimf.models.components.conditions.Condition import Condition

__author__ = "Mathijs Maijer"
__email__ = "m.f.maijer@gmail.com"


class UpdateType(Enum):
    STATE = 0
    UTILITY = 1
    NETWORK = 2
    EDGE_VALUES = 3

class UpdateConfiguration(object):
    """
    Configuration for Updates
    """
    def __init__(self, config=None):
        self.set_config(config)
        self.validate()

    def set_config(self, config):
        self.config = config if config else {}
        self.arguments = config['arguments'] if 'arguments' in self.config else {}
        self.condition = config['condition'] if 'condition' in self.config else None
        self.get_nodes = config['get_nodes'] if 'get_nodes' in self.config else None
        self.update_type = config['update_type'] if 'update_type' in self.config else UpdateType.STATE

    def validate(self):
        ConfigValidator.validate('arguments', self.arguments, dict)
        ConfigValidator.validate('condition', self.condition, Condition, optional=True)
        ConfigValidator.validate('get_nodes', self.get_nodes, bool, optional=True)
        ConfigValidator.validate('update_type', self.update_type, UpdateType)
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
