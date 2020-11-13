from enum import Enum

from dynsimf.models.helpers.ConfigValidator import ConfigValidator

__author__ = "Mathijs Maijer"
__email__ = "m.f.maijer@gmail.com"


class MemoryConfigurationType(Enum):
    STATE = 0
    ADJACENCY = 1
    UTILITY = 2
    EDGE_VALUES = 3

class MemoryConfiguration(object):
    """A configuration for storing and writing utility/states/graph

    Attributes:
        save_disk: A boolean indicating whether to save the states to the disk
        path: If save_disk is set, the path to save the data to
        memory_size: The amount of iterations to keep in memory, -1 to not keep any, 0 to store every iteration
        save_interval: The amount of iterations between a save
        memory_interval: The amount of iterations between the storing of memory
    """
    memory_type: MemoryConfigurationType
    save_disk: bool
    memory_size: int
    save_interval: int
    memory_interval: int
    path: str

    def __init__(self, memory_type, cfg=None):
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
        """
    	Returns a description for the output
        :param n int: Total amount of simulation iterations
        :param rows: Amount of rows to write per iteration
        :param cols: Amount of columns to write per iteration
        """
        return '# Output array shape:\n#({0}, {1}, {2})\n'.format(int(n / self.save_interval), rows, cols)
