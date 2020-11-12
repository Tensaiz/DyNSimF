__author__ = "Mathijs Maijer"
__email__ = "m.f.maijer@gmail.com"

class ConfigValidator(object):
    """
    A class that can be used to validate variable types and values

    Mostly used to validate user specified values in a config

    A value error will be thrown if a variable does not match with the expected type or value
    """

    @classmethod
    def validate(cls, name, variable, variable_type, variable_range=None, optional=False):
        """
        Validate a variable type, optionality and range if set.

        Args:
            name (str): Name of the variable.
            variable (any): The variable to validate.
            variable_range (tuple(start, end), optional): The range a variable value should lie in.
            optional (bool, optional): Boolean indicating whether the variable can be optional.

        Raises:
            ValueError: if any of the expected variable type or values does not match
        """
        cls.check_optionality(name, variable, optional)
        if optional and not variable:
            return
        cls.check_type(name, variable, variable_type)
        if variable_range:
            cls.check_range(name, variable, variable_range)

    @staticmethod
    def check_optionality(name, variable, optional):
        """
        Check whether a variable may be None, otherwise throw a value error

        Args:
            name (str): Name of the variable.
            variable (any): The variable to check the optionality for.
            optional (bool): Boolean indicating whether the variable can be optional

        Raises:
            ValueError: if the variable is None while optional is False
        """
        if not optional and not variable:
            raise ValueError('The variable: ' + name + ' is not optional and should not be None!')

    @staticmethod
    def check_type(name, variable, variable_type):
        """
        Check whether a variable type matches with an expected type, otherwise throw a value error

        Args:
            name (str): Name of the variable.
            variable (any): The variable to check the type for.
            variable_type (any): The expected variable type

        Raises:
            ValueError: if the variable type does not match the expected type
        """
        if not isinstance(variable, variable_type):
            raise ValueError('The variable: ' + name + ' should be of type: ' + str(variable_type))

    @staticmethod
    def check_types(name, variable, variable_types):
        """
        Check whether a variable type matches with any of the expected types, otherwise throw a value error

        Args:
            name (str): Name of the variable.
            variable (any): The variable to check the type for.
            variable_types (list[any]): The expected variable types

        Raises:
            ValueError: if the variable type does not match with any of the expected types
        """
        if type(variable) not in variable_types:
            raise ValueError('The variable: ' + name +
                            ' should be one of the types: ' + str(variable_types) +
                            ' it is currently: ' + type(variable))

    @staticmethod
    def check_range(name, variable, variable_range):
        """
        Check whether a variable lies within an expected range, otherwise throw a value error

        Args:
            name (str): Name of the variable.
            variable (any): The variable to check whether the value lies in the expected range.
            variable_range (tuple(from(comparable), to(comparable(optional)))): The expected variable range

        Raises:
            ValueError: if the variable does not lie in the expected range
        """
        if not variable >= variable_range[0]:
            raise ValueError('The variable ' + name + ' should be larger than or equal to ' +
                            variable_range[0] + ' it currently is: ' + variable)

        if len(variable_range) == 2:
            if not variable <= variable_range[1]:
                raise ValueError('The variable ' + name + ' should be smaller than or equal to ' +
                                variable_range[1] + ' it currently is: ' + variable)
