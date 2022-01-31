# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from logging import Logger
from typing import List

from simple.measure import Measure

class Metric(Measure):
    """
    This class extends `Measure` by adding a better print function for showing the metrics.
    """
    def __init__(self):
        super(Metric, self).__init__()

    def view(self, setting: List, _log: Logger):
        """
        Displays the metrics of the model on the console.

        Args:
            setting (List): Specifies the metric setting i.e. 'fil', 'raw'.
            _log (Logger): Specifies the logger that logs the results.
        """
        for raw_or_fil in setting:
            _log.info(raw_or_fil.title() + " setting:")
            _log.info("\tHit@1 = %f" % self.hit1[raw_or_fil])
            _log.info("\tHit@3 = %f" % self.hit3[raw_or_fil])
            _log.info("\tHit@10 = %f" % self.hit10[raw_or_fil])
            _log.info("\tMR = %f" % self.mr[raw_or_fil])
            _log.info("\tMRR = %f" % self.mrr[raw_or_fil])
