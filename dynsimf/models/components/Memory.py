from enum import Enum

from dynsimf.models.helpers.ConfigValidator import ConfigValidator

__author__ = "Mathijs Maijer"
__email__ = "m.f.maijer@gmail.com"


class MemoryConfigurationType(Enum):
    '''
    Enumeration for the type of the memory configuration
    '''
    STATE = 0
    ADJACENCY = 1
    UTILITY = 2
    EDGE_VALUES = 3

class MemoryConfiguration(object):
    '''
    A configuration for storing and writing utility/states/graphs

    :var memory_type: Indicates what type the config specifies using the `MemoryConfigurationType` enum
    :vartype memory_type: MemoryConfigurationType
    :var save_disk: A boolean indicating whether to save the states to the disk,
        defaults to False
    :vartype save_disk: bool
    :var memory_size: The amount of iterations to keep in memory, -1 to not keep any, 0 to store every iteration,
        defaults to -1
    :vartype memory_size: int
    :var save_interval: The amount of iterations between a save,
        defaults to 0
    :vartype save_interval: int
    :var memory_interval: The amount of iterations between the storing of memory,
        defaults to 1
    :vartype memory_interval: int
    :var path: If save_disk is set, the path to save the data to,
        defaults to: `./output/memory_type_name.txt`
        where memory_type_name is the name of the memory type enum
    :vartype path: str
    '''

    def __init__(self, memory_type, cfg=None):
        '''
        Initialize the memory configuration by setting all the members and their default values if no value is provided

        :param MemoryConfigurationType memory_type:
            Indicates what type the config specifies using the `MemoryConfigurationType` enum
        :param cfg: Dictionary containing the values for the members of the configuration class,
            When certain keys/value pairs are not provided, default values are used
        :type cfg: dict, optional
        '''
        self.memory_type = memory_type
        cfg = cfg if cfg else {}
        cfg_keys = list(cfg.keys())
        self.save_disk = False if 'save_disk' not in cfg_keys else cfg['save_disk']
        self.memory_size = -1 if 'memory_size' not in cfg_keys else cfg['memory_size']
        self.save_interval = 0 if 'save_interval' not in cfg_keys else cfg['save_interval']
        self.memory_interval = 1 if 'memory_interval' not in cfg_keys else cfg['memory_interval']
        self.path = './output/' + self.memory_type.name + '.txt' if 'path' not in cfg_keys else cfg['path']
        self.validate()

    def validate(self):
        '''
        Validate the memory configuration

        :raises ValueError: if any of the members does not match their specified type
        :raises ValueError: if the save interval is negative when the `save_disk` member is set to `True`
        :raises ValueError: if the `memory_interval` member is less than -1
        '''
        ConfigValidator.validate('save_disk', self.save_disk, bool)
        ConfigValidator.validate('memory_size', self.memory_size, int)
        ConfigValidator.validate('save_interval', self.save_interval, int)
        ConfigValidator.validate('memory_interval', self.memory_interval, int)
        ConfigValidator.validate('path', self.path, str)

        if self.save_disk and self.save_interval < 1:
            raise ValueError('Save interval should be a positive integer when save disk is true')

        if self.memory_interval < -1:
            raise ValueError('Memory interval should be a valid value, [-1 <= x <= n]')

    def get_description_string(self, n, rows, cols):
        '''
    	Returns a description for the output when writing to disk

        :param int n: Total amount of simulation iterations
        :param int rows: Amount of rows to write per iteration
        :param int cols: Amount of columns to write per iteration
        :return: A string containing data about the output that can be read in to parse the information after
        :rtype: str
        '''
        return '# Output array shape:\n#({0}, {1}, {2})\n'.format(int(n / self.save_interval), rows, cols)
