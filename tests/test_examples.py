#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Test examples inside agents/examples
"""

import os
import unittest
import parlai.utils.testing as testing_utils


class TestTorchRankerAgentExamples(unittest.TestCase):
    """
    Checks that transformer_ranker can learn some very basic tasks.
    """

    @testing_utils.retry(ntries=3)
    def test_repeater(self):
        """
        Test a simple repeat-after-me model.
        """
        with testing_utils.tempdir() as tmpdir:
            model_file = os.path.join(tmpdir, 'model')
            stdout, valid, test = testing_utils.train_model(
                dict(
                    task='integration_tests',
                    model_file=model_file,
                    model='examples/tra',
                    eps=1,
                    bs=100,
                )
            )
            self.assertEqual(
                test['exs'],
                100,
                'test examples = {}\nLOG:\n{}'.format(valid['exs'], stdout),
            )
