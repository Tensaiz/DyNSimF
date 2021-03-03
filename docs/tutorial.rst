********
Tutorial
********

This section will describe the main modeling flow and the relevant concepts that will appear.
After following this section, it will be clear on how to implement your own model, or run one of the examples.

The main goal of this framework, is to create an environment where a network model, whether static or dynamic can be created.
Nodes inside a network can have multiple states that can be updated based on certain conditions. Next to this, the edges can also be updated.

----------------------
General modeling steps
----------------------

The general modeling flow is as follows:

1. Define a graph
2. Add (continuous) internal states
3. Define constants and intial values
4. Create update conditions
5. Add iteration schemes (optional)
6. Create a utility/cost layer (optional)

    a. Define a utility and cost
    b. Choose an iteration method
    c. Choose a sample method

7. Simulate
8. Optional steps(Visualize/Sensitivity analysis)

By following these steps, almost any network that relies on state updating using conditions can be created and simulated.

------------------------------------
Graph, internal states and constants
------------------------------------

The graphs should be ``Networkx`` graphs, either defined by yourself or generated using one of their built-in functions.
Attributes in the graph can still be accessed and used to update functions.

After a graph is defined, the model can be initialized and internal states can be added to the model. The model constructor takes in a ``ModelConfiguration`` object,
which functions as a config. This object can be used to specify what information should be saved / written to disk.

When the model is initalized,
states can be added using ``set_states([state1, state2, state3, ..., stateN])`` function, where the argument is a list containing strings indicating the states.

If the model requires certain constant values, these can be added by setting them manually.
It should be a dictionary where the key corresponds to the constant name and the value to the constant value.
Adding constants is completely optional.

Example:


.. code-block:: python

    import networkx as nx
    from dynsimf.models.Model import Model
    from dynsimf.models.Model import ModelConfiguration

    # Create an initial graph
    g = nx.erdos_renyi_graph(n=1000, p=0.1)

    # Create the constants dictionary
    constants = {
        'constant_1': 0.1,
        'constant_2': 2.5
    }

    cfg = ModelConfiguration() # A configuration object for the model
    model = Model(g, ModelConfiguration(cfg)) # Initialize a model object
    model.constants = constants # Set the constants
    model.set_states(['state1', 'state2']) # Add the states to the model


-------------
Intial values
-------------

After the graph has been created, the model has been initalized, and the internal states have been added,
the next step is to define the intial values of the states.

This is done by creating a dictionary, that maps a state name to a list of initial values per node, or a constant value.
This value has to be a continous value, that can be statically set, or it can be a function that will return a list of all the initial values.

After creating the dictionary, it can be added to the model using the ``set_initial_state(initial_dict, param_dict)`` function.
The param_dict argument is a dictionary that specifies which input the functions inside the initial_dict should receive as parameters.

The example below will create a model with 3 states. Every node in the model will initialize `status_1` with a value returned by the `initial_status_1` function,
which will results in all nodes getting a random uniform value between 0 and 0.5. The same happens for the internal state `status_2`.
The third state is constant and thus the same for every node.

.. code-block:: python

    import networkx as nx
    import numpy as np
    from dynsimf.models.Model import Model

    constants = {
        'x': 0.2,
        'N', 100
    }

    # Returns constants[N] amount of random numbers between 0 and 0.5
    def initial_state_1(constants):
        return np.random.uniform(0, 0.5, constants['N'])

    # Returns constants[N] amount of random numbers between 0.5 and 1 + constants[x] (0.2)
    def initial_state_2(constants):
        return constants['x'] + np.random.uniform(0.5, 1, constants['N'])

    initial_states = {
        'state_1': initial_state_1, # A function returning initial values for every node for their state 1
        'state_2': initial_state_2, # A function returning initial values for every node for their state 2
        'state_3': 2 # A constant
    }

    g = nx.erdos_renyi_graph(n=100, p=0.1)

    model = Model(g)
    model.constants = constants
    model.add_states['state_1', 'state_2', 'state_3']

    # The paramaters we want to receive in our initalization functions
    initial_params = {
        'constants': model.constants
    }
    model.set_initial_state(initial_state, initial_params)

-------
Updates
-------

-----------------
Update conditions
-----------------

Another important part of the model is creating conditions and update rules. There are multiple levels a condition could apply to, these levels can be selected using the ConditionType class:

    - State: A condition based on a node's state
    - Adjacency: Condition based on the number of neighbors of a node
    - Utility: A condition based on the current utility of the node
    - Edge values: A custom condition based on the edge values between nodes

Next to that, there are multiple ways of evaluating whether a condition is True:

    - StochasticCondition: Takes in a number and makes a random draw, if the number is smaller than the argument, it will satisfy the condition.
    - ThresholdCondition: A condition to check whether a value specified using ConditionType, is greater than, greater or equal than, lesser than, lesser or equal than a certain value.
    - Custom: A custom condition that could be anything based on the values in the framework

In the example below, the states are updated when the condition ``NodeStochastic(1)`` is true, which is always the case, so the update functions are called every iteration.
Here the state `state_1` will be updated every iteration by setting it equal to `state_2` + 0.1. The same is done for `state_2`, but in this case it is set equal to `state_1` + 0.5.
As can be seen in the example, the states can be retrieved using the function ``get_state(state_name)``.

.. code-block:: python

    import networkx as nx
    import numpy as np
    from dynsimf.models.Model import Model

    from dynsimf.models.components.conditions.Condition import ConditionType
    from dynsimf.models.components.conditions.StochasticCondition import StochasticCondition

    g = nx.erdos_renyi_graph(n=100, p=0.1)

    model = Model(g)

    model.add_states(['state_1', 'state_2']

    # Creating the condition
    s = StochasticCondition(ConditionType.STATE, 1)

    # Update functions
    def update_1():
        return {'state_1': model.get_state('state_2') + 0.1}

    def update_2():
        return {'state_2': model.get_state('state_1') + 0.5}

    # Rules
    model.add_update(update_1, condition=s)
    model.add_update(update_2, condition=s)

Note that conditions can be chained together as well. As could be seen above, when the condition was initialized, the type was provided: ``StochasticCondition(ConditionType.STATE, 1)``. 
But every condition can also take a `chained_condition` parameter, that takes in another condition object. All nodes that pass the current condition, will then be passed to the next condition in the chain. 
The chained condition can also have another condition chained to it, allowing any set of conditions. The documentation on the conditions can be read to see which parameters can be used for which condition.

-----------------
Iteration schemes
-----------------

Another addition to the model, are iteration schemes. These can be used for two things:

1. Specify nodes to update
2. Specify iteration range when updates should take place

This allows to only update a select amount of nodes during a specific time in the iteration.
Under the hood, when schemes are not defined,
a default scheme is used for every rule that is active during each iteration and selects all nodes.

To create a scheme, simply create a scheme object that takes

	- A sample function: a function that returns the nodes that should be updated. ary with State -> Value mappings

    - A dictionary with as keys:

        - 'args': maps to a dictionary with arguments for the sample function

        - 'lower_bound' (optional): maps to an integer indicating from which iteration the scheme should apply

        - 'upper_bound' (optional): maps to an integer indicating until which iteration the scheme should apply

        - 'updates': maps to a list of update objects that should be executed during this scheme

After the scheme object is created, it can be added to the model by using the `add_scheme function`:
``model.add_scheme(Scheme(sample_function, {'args': {'argument': argument_value}, 'updates': [update_object]}))``.

In the example below, the update_1 function is only being evaluated when ``lower ≤ iteration < upper``,
in this case only when the second iteration is occurring.
In this case a node is selected based on the weighted `status_1` value.
Because no scheme has been added to the second rule, it will be evaluated and executed for every node, each iteration.

The output of the model will provide 0.6 as value for `status_2`, as 0.1 is added each iteration to the initial value of 0.1.
Furthermore, at iteration 2, one node is sampled by the sample function, for which status_2 is taken and then increased by 0.1. This value is then updated for its status_1 value.
This means that for the end result only one node will have a 0.4 value for status_1, while the other nodes will have their initial value of 0.1 after 5 iterations.


.. code-block:: python

    import networkx as nx
    import numpy as np

    from dynsimf.models.Model import Model
    from dynsimf.models.Model import ModelConfiguration
    from dynsimf.models.components.Update import Update
    from dynsimf.models.components.Update import UpdateType
    from dynsimf.models.components.Update import UpdateConfiguration
    from dynsimf.models.components.Scheme import Scheme

    g = nx.erdos_renyi_graph(n=10, p=0.1)
    model = Model(g, ModelConfiguration())

    # Define schemes
    def sample_state_weighted(graph):
        probs = []
        status_1 = model.get_state('status_1')
        factor = 1.0/sum(status_1)
        for s in status_1:
            probs.append(s * factor)
        return np.random.choice(graph.nodes, size=1, replace=False, p=probs)

    initial_state = {
        'status_1': 0.1,
        'status_2': 0.1,
    }

    model.set_states(['status_1', 'status_2'])

    # Update functions
    def update_1(nodes):
        node = nodes[0]
        node_update = model.get_state('status_2')[node] + 0.1
        return {'status_1': {node: node_update}}

    def update_2():
        s2 = model.get_state('status_2') + 0.1
        return {'status_2': s2}

    # Add update 2 to the model, which will increase status_1 by 0.5 each iteration
    model.add_update(update_2)

    update_cfg = UpdateConfiguration({
        'arguments': {}, # No other arguments
        'get_nodes': True, # We want the nodes as argument
        'update_type': UpdateType.STATE
    })
    u = Update(update_1, update_cfg) # Create an Update object that contains the object function
    # Create a scheme with the correct sample function, parameters, bounds, and update function
    model.add_scheme(Scheme(sample_state_weighted, {'args': {'graph': model.graph}, 'lower_bound': 2, 'upper_bound': 3, 'updates': [u]}))

    model.set_initial_state(initial_state)
    output = model.simulate(5)

------------------
Utility Cost model
------------------

This section will describe the extra utility cost layer that can be added to a model. More about this concept can be read in the paper `A Structural Model of Segregation in Social Networks` by Angelo Mele.
Basically, by utilizing 3 functions of the package, the whole network will reconfigure itself in a way to maximise the utility for each node, while not exceeding a set cost threshold.
Every node link has a calculated utility and a calculated cost, which are both defined by the user. 
The package will automatically try to configure the network in such a way to maximise the utility for each node, while staying below a set cost threshold.

The first step is to initialize the model:

.. code-block:: python

    from dynsimf.models.UtilityCostModel import FunctionType
    from dynsimf.models.UtilityCostModel import SampleMethod
    from dynsimf.models.UtilityCostModel import UtilityCostModel
    from dynsimf.models.Model import ModelConfiguration
    from dynsimf.models.components.Memory import MemoryConfiguration
    from dynsimf.models.components.Memory import MemoryConfigurationType

    cfg = {
        'adjacency_memory_config': \
            MemoryConfiguration(MemoryConfigurationType.ADJACENCY, {
                'memory_size': 0
            }),
        'utility_memory_config': \
            MemoryConfiguration(MemoryConfigurationType.UTILITY, {
                'memory_size': 0
            })
    }
    cost_threshold = 1
    model = UtilityCostModel(g, cost_threshold, ModelConfiguration(cfg))

A few things are happening here. First an actual configuration dictionary is used for the `ModelConfiguration` class. 
This allows the storage of the utility and adjacency values each iteration. A value of `0` means that it should be stored for every round. 
`-1` would mean that the values should not be stored.
Any value above 0 indicates the amount of consecutive iterations the values should be stored for, which might be helpful if you need the values of the previous N iterations.

Then, when the model is initialized, the ``UtilityCostModel`` class is used instead of the regular ``Model`` class. 
It also takes a `cost_threshold` argument, that indicates the max sum of the costs a node can have. After the initialization of the model, three functions must be set by the user.

.. code-block:: python

    def utility_calculation():
        return np.random.random((n_nodes, n_nodes))

    def cost_calculation():
        adjacency = model.get_adjacency()
        cost = np.ones((n_nodes, n_nodes)) * 0.1
        return cost

    model.add_utility_function(utility_calculation, FunctionType.MATRIX)
    model.add_cost_function(cost_calculation, FunctionType.MATRIX)
    model.set_sampling_function(SampleMethod.NEIGHBORS_OF_NEIGHBORS)

Several things are happening here. First the utility and cost functions are defined. 
There are two possibilities when defining these functions, either they return a matrix, or a single value but take in 2 nodes as arguments.
If a matrix is returned, it should have the dimensions of the amount of nodes x the amount of nodes. 
When providing the defined functions, the second argument defines which type it is, pairwise or matrix, which is done by using the ``FunctionType`` enumeration.
The ``add_utility_function`` and ``add_cost_function`` do just that, they add the user defined functions to the model.

The final step is to select a sampling function, that defines which nodes may connect with each other. 

.. code-block:: python

    class SampleMethod(Enum):
        ALL = 0
        NEIGHBORS_OF_NEIGHBORS = 1
        CUSTOM = 2

By using ``SampleMethod.ALL``, every node can potentially connect to another node, while the `NEIGHBORS_OF_NEIGHBORS` option only allows nodes to connect to their neighbors of neighbors.
If a custom sample method is used, it should return a matrix, where each row and column represent a node. Every `1` in a column represents that the node in that row can form a connection with that node in the column.

----------
Simulation
----------

After everything has been specified and added to the model, it can be ran using the ``simulate(iterations)`` function.
It will run the model iterations amount of times and return the regular output as shown in other models before.

.. ------------------------
.. Optional functionalities
.. ------------------------

.. There are several extra configurations and options:

.. .. toctree::
..    :maxdepth: 2

..    optional/ModelRunner.rst
..    optional/Visualization.rst

.. The ``ContinuousModelRunner`` can be used to simulate a model mutliple times using different parameters.
.. It also includes sensitivity analysis functionalities.

.. The visualization section explains how visualizations can be configured, shown, and saved.

----------------
Helper functions
----------------

There are several functions that can be used during the simulation in the update functions that can be of help.

	- ``get_state(state_name)`` Gets all the values of a state.

    - ``get_previous_nodes_states(n)`` Get all the nodes' states from the n'th previous saved iteration

    - ``get_adjacency()`` Gets the adjacency matrix

    - ``get_neighbors(node)`` Get all neighbors of a ndoe

    - ``get_neighbors_neighbors_adjacency_matrix()`` Get the adjacency matrix, where a 1 represents a neighbor of a neighbor.

    - ``get_neighbors_neighbors(node)`` Get the neighbors of all neighbors of a node.

    - ``get_utility()`` Get the current node utility.

    - ``get_previous_nodes_utility(iteration)`` Get previous node utility where iteration is the n'th previous iteration.

--------
Examples
--------

Two examples have been added that reproduce models shown in two different papers.

.. toctree::
   :maxdepth: 2

   examples/ControlVsCraving.rst
   examples/HIOM.rst