from enum import Enum
from dynsimf.models.helpers.ConfigValidator import ConfigValidator
from dynsimf.models.components.conditions.Condition import Condition

__author__ = "Mathijs Maijer"
__email__ = "m.f.maijer@gmail.com"


class UpdateType(Enum):
    '''
    An Enum to specify the type of the update
    '''
    STATE = 0
    NETWORK = 1
    EDGE_VALUES = 2


class UpdateConfiguration(object):
    '''
    Configuration for Updates

    :var config: The dictionary containing the key/value pairs of the members of this class,
        if no key/value pair is provided, a default value is used instead
    :vartype config: dict
    :var arguments: A dictionary with arguments for the update function, defaults to empty dict
    :vartype arguments: dict
    :var condition: A condition for nodes that must be met before the update is executed on them
    :vartype condition: Condition or None
    :var get_nodes: A boolean indicating whether the update function should receive a list of sampled nodes as argument,
        defaults to None
    :vartype get-nodes: bool or None
    :var update_type: A value from the `UpdateType` enum, indicating what kind of update is being performed,
        defaults to `UpdateType.STATE`
    :vartype update_type: UpdateType
    '''
    def __init__(self, config=None):
        self.set_config(config)
        self.validate()

    def set_config(self, config):
        '''
        Set the values for the members of the class by reading them from the config or setting their default values

        :param dict config: The configuration dictionary with the key/value pairs for the class members
        '''
        self.config = config if config else {}
        self.arguments = config['arguments'] if 'arguments' in self.config else {}
        self.condition = config['condition'] if 'condition' in self.config else None
        self.get_nodes = config['get_nodes'] if 'get_nodes' in self.config else None
        self.update_type = config['update_type'] if 'update_type' in self.config else UpdateType.STATE

    def validate(self):
        '''
        Validate the update configuration

        :raises ValueError: if the `update_type` member is not of type `UpdateType`
        '''
        ConfigValidator.validate('arguments', self.arguments, dict)
        ConfigValidator.validate('condition', self.condition, Condition, optional=True)
        ConfigValidator.validate('get_nodes', self.get_nodes, bool, optional=True)
        ConfigValidator.validate('update_type', self.update_type, UpdateType)
        if not isinstance(self.update_type, UpdateType):
            raise ValueError('Update type should of enum UpdateType')


class Update(object):
    """
    Update class

    :var fun: The update function that should be executed
    :vartype fun: function
    :var config: UpdateConfiguration object, defaults to a new UpdateConfiguration object
    :vartype config: UpdateConfiguration
    :var arguments: A dictionary with arguments for the update function, defaults to empty dict
    :vartype arguments: dict
    :var condition: A condition for nodes that must be met before the update is executed on them
    :vartype condition: Condition or None
    :var get_nodes: A boolean indicating whether the update function should receive a list of sampled nodes as argument, 
        defaults to None
    :vartype get-nodes: bool or None
    :var update_type: A value from the `UpdateType` enum, indicating what kind of update is being performed,
        defaults to `UpdateType.STATE`
    :vartype update_type: UpdateType
    """

    def __init__(self, fun, config=None):
        '''
        Initialise the update by setting the class members to their values defined in the config

        :param function fun: The update function to execute
        :param config: The configuration object containing the values for the class members
        :type config: UpdateConfiguration, optional
        '''
        self.function = fun
        self.config = config if config else UpdateConfiguration()
        self.arguments = self.config.arguments
        self.condition = self.config.condition
        self.get_nodes = self.config.get_nodes
        self.update_type = self.config.update_type

    def execute(self, nodes=None):
        '''
        Execute the update function with or without the sampled nodes from the scheme/condition 
        and return the output

        :param nodes: An optional list of nodes that the update function should be applied on.
            The given nodes are filtered by the schemes and conditions
        :type nodes: list, optional
        '''
        if self.get_nodes:
            output = self.function(nodes, **self.arguments)
        else:
            output = self.function(**self.arguments)
        return output
