from dynsimf.models.ExampleRunner import ExampleRunner
# from dynsimf.models.tools.Visualizer import Visualizer

__author__ = "Mathijs Maijer"
__email__ = "m.f.maijer@gmail.com"

if __name__ == "__main__":
    # print('Running craving vs self control model')
    # m = 'Craving vs Self control'
    # model = ExampleRunner(m)
    # output = model.simulate(100)
    # model.visualize(output)
    # # states = Visualizer.read_states_from_file('./out.txt')
    # # model.visualize(states)

    print('Running HIOM model')
    m = 'HIOM'
    model = ExampleRunner(m)
    output = model.simulate(15000)
    model.visualize(output)
    # states = Visualizer.read_states_from_file('./out.txt')
    # model.visualize(states)