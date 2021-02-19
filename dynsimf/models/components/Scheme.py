from typing import Callable, List
import types

from dynsimf.models.components.Update import Update
from dynsimf.models.helpers.ConfigValidator import ConfigValidator

__author__ = "Mathijs Maijer"
__email__ = "m.f.maijer@gmail.com"


class Scheme(object):
    '''
    A class used to select iteration bounds for an update and to sample a select amount of nodes to apply updates on
    
    :var sample_function: The sample function to select the nodes to apply the updates on
    :vartype sample_function: function
    :var args: The arguments to provide for the sample function, defaults to an empty dictionary
    :vartype args: dict
    :var lower_bound: The lower bound of the updates to be executed, defaults to None
    :vartype lower_bound: int or None
    :var upper_bound: The upper bound of the updates to be executed, defaults to None
    :vartype upper_bound: int or None
    :var updates: A list of `Update` objects, defaults to empty list
    :vartype updates: list 
    '''
    def __init__(self, sample_function: Callable, config: dict):
        '''
        Initialise the scheme

        :param function sample_function: The sample function used to select the nodes to apply updates on
        :param dict config: The dictionary containing the settings for the class members
        '''
        self.sample_function: Callable = sample_function
        self.args: dict = config['args'] if 'args' in config else {}
        self.lower_bound: int = config['lower_bound'] if 'lower_bound' in config else None
        self.upper_bound: int = config['upper_bound'] if 'upper_bound' in config else None
        self.updates: List[Update] = config['updates'] if 'updates' in config else []
        self.validate()

    def validate(self):
        '''
        Validate the config using the `ConfigValidator` class

        :raises ValueError: if any member does not match its type or range
        '''
        ConfigValidator.validate('sample_function', self.sample_function, types.FunctionType)
        ConfigValidator.validate('args', self.args, dict, optional=True)
        ConfigValidator.validate('lower_bound', self.lower_bound, int, optional=True, variable_range=(0,))
        ConfigValidator.validate('upper_bound', self.upper_bound, int, optional=True)
        ConfigValidator.validate('updates', self.updates, list, optional=True)

    def add_update(self, update: Update) -> None:
        '''
        Add a Update object to the list of updates of the scheme

        :param Update update: The update to add to the list of updates
        '''
        self.updates.append(update)

    def set_bounds(self, lower: int, upper: int) -> None:
        '''
        Set the bounds of the scheme

        :param int lower: The lower bound to set
        :param int upper: The upper bound to set
        '''
        self.lower_bound = lower
        self.upper_bound = upper

    def sample(self):
        ''' 
        Run the sample function with the configured arguments and return the result

        :return: A list of nodes sampled by the `sample_function` member
        :rtype: list
        '''
        return self.sample_function(**self.args)
