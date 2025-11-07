import os
import re
import signal
import sys
import time
from multiprocessing import Process
from typing import List
from concurrent_metrics_checker import ConcurrentMetricsChecker

import numpy as np


class ConcurrentMetricsCheckerExtended(ConcurrentMetricsChecker):

    def __init__(self, output_path: str, metrics_api_url: str, available_loras: List[str]):
        super().__init__(output_path, metrics_api_url, available_loras)

        self.available_loras = available_loras

        self.running_by_adapter = {}
        self.waiting_by_adapter = {}
        for index in range(len(self.available_loras) + 1):
            self.running_by_adapter[index] = []
            self.waiting_by_adapter[index] = []

    def extract_metrics(self, prometheus_output: str):
        super().extract_metrics(prometheus_output)

        for index in range(len(self.available_loras) + 1):
            self.running_by_adapter[index].append(
                self.find_prometheus_metric_value(
                    f"running_by_adapter_{index}",
                    prometheus_output
                )
            )
            self.waiting_by_adapter[index].append(
                self.find_prometheus_metric_value(
                    f"waiting_by_adapter_{index}",
                    prometheus_output
                )
            )

    def save_metrics(self):
        super().save_metrics()

        for index in range(len(self.available_loras) + 1):
            running_by_adapter = np.asarray(self.running_by_adapter[index])
            if np.count_nonzero(running_by_adapter) > 0:
                np.save(
                    os.path.join(self.output_path, f'running_by_adapter_{index}'),
                    running_by_adapter
                )
            waiting_by_adapter = np.asarray(self.waiting_by_adapter[index])
            if np.count_nonzero(waiting_by_adapter) > 0:
                np.save(
                    os.path.join(self.output_path, f'waiting_by_adapter_{index}'),
                    waiting_by_adapter
                )
