from typing import Callable, List
import types

from dynsimf.models.components.Update import Update
from dynsimf.models.helpers.ConfigValidator import ConfigValidator

__author__ = "Mathijs Maijer"
__email__ = "m.f.maijer@gmail.com"


class Scheme(object):
    def __init__(self, sample_function: Callable, args: dict = None,
                 lower_bound: int = None, upper_bound: int = None,
                 updates: list = None):
        self.sample_function: Callable = sample_function
        self.args: dict = args if args else {}
        self.lower_bound: int = lower_bound
        self.upper_bound: int = upper_bound
        self.updates: List[Update] = updates if updates else []
        self.validate()

    def validate(self):
        ConfigValidator.validate('sample_function', self.sample_function, types.FunctionType)
        ConfigValidator.validate('args', self.args, dict, optional=True)
        ConfigValidator.validate('lower_bound', self.lower_bound, int, optional=True, variable_range=(0,))
        ConfigValidator.validate('upper_bound', self.upper_bound, int, optional=True)
        ConfigValidator.validate('updates', self.updates, list, optional=True)

    def add_update(self, update: Update) -> None:
        self.updates.append(update)

    def set_bounds(self, lower: int, upper: int) -> None:
        self.lower_bound = lower
        self.upper_bound = upper

    def sample(self) -> int:
        return self.sample_function(**self.args)
