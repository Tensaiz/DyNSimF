import unittest

from dynsimf.models.Update import Update
from dynsimf.models.Update import UpdateType
from dynsimf.models.Update import UpdateConfiguration

__author__ = "Mathijs Maijer"
__email__ = "m.f.maijer@gmail.com"


class UpdateTest(unittest.TestCase):
    def test_init(self):
        cfg_options = {
            'arguments': None,
            'condition': None,
            'get_nodes': False,
            'update_type': UpdateType.STATE
        }
        update_cfg = UpdateConfiguration(cfg_options)
        u = Update(lambda x: x, update_cfg)
        self.assertTrue(isinstance(u, Update))

    def test_execute(self):
        cfg_options = {
            'arguments': {'x': [1, 2, 3]},
            'condition': None,
            'get_nodes': False,
            'update_type': UpdateType.STATE
        }
        update_cfg = UpdateConfiguration(cfg_options)
        u = Update(lambda x: x, update_cfg)
        self.assertEqual(u.execute(), [1, 2, 3])

        cfg_options = {
            'arguments': {'y': [3, 4, 5]},
            'condition': None,
            'get_nodes': True,
            'update_type': UpdateType.STATE
        }
        update_cfg = UpdateConfiguration(cfg_options)
        u = Update(lambda x, y: x + y, update_cfg)
        self.assertEqual(u.execute([1, 2]), [1, 2, 3, 4, 5])
