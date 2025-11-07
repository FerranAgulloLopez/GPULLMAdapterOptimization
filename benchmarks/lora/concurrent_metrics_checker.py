import os
import re
import signal
import sys
import time
from multiprocessing import Process
from typing import Optional

import numpy as np
import requests


class ConcurrentMetricsChecker(Process):

    def __init__(self, output_path: str, metrics_api_url: str, title: Optional[str] = ''):
        super(ConcurrentMetricsChecker, self).__init__()
        self.output_path = output_path
        self.metrics_api_url = metrics_api_url
        self.title = title
        signal.signal(signal.SIGTERM, self.__signal_term_handler)

        self.time = []
        self.gpu_cache_usage_perc = []
        self.num_running = []
        self.num_waiting = []
        self.num_preempted = []
        self.finished = []
        self.processed_tokens_prompt = []
        self.processed_tokens_generation = []

    def run(self):
        while True:
            try:
                prometheus_output = requests.get(self.metrics_api_url).text
                self.extract_metrics(prometheus_output)
            except Exception as e:
                print(f'Error while running usage checker: {e}')
            finally:
                time.sleep(2)

    def __signal_term_handler(self, signal, frame):
        self.save_metrics()
        sys.exit(0)

    def find_prometheus_metric_value(self, label: str, metrics_response: str):
        try:
            pattern = f"vllm:{label}(.+?) ([+-]?([0-9]*[.])?[0-9]+)\n"
            value = re.search(pattern, metrics_response).group(2)
        except:
            # print(f'Pattern error with label {label}')
            return 0
        return float(value)

    def extract_metrics(self, prometheus_output: str):
        self.gpu_cache_usage_perc.append(
            self.find_prometheus_metric_value(
                f"gpu_cache_usage_perc",
                prometheus_output
            )
        )

        self.num_running.append(
            self.find_prometheus_metric_value(
                f"num_requests_running",
                prometheus_output
            )
        )

        self.num_waiting.append(
            self.find_prometheus_metric_value(
                f"num_requests_waiting",
                prometheus_output
            )
        )

        self.num_preempted.append(
            self.find_prometheus_metric_value(
                f"num_preemptions_total",
                prometheus_output
            )
        )

        self.finished.append(
            self.find_prometheus_metric_value(
                f"request_success_total",
                prometheus_output
            )
        )

        self.processed_tokens_prompt.append(
            self.find_prometheus_metric_value(
                f"prompt_tokens_total",
                prometheus_output
            )
        )
        self.processed_tokens_generation.append(
            self.find_prometheus_metric_value(
                f"generation_tokens_total",
                prometheus_output
            )
        )
        self.time.append(time.perf_counter())

    def save_metrics(self):
        np.save(
            os.path.join(self.output_path, f'time_{self.title}'),
            np.asarray(self.time)
        )
        np.save(
            os.path.join(self.output_path, f'gpu_cache_usage_perc_{self.title}'),
            np.asarray(self.gpu_cache_usage_perc)
        )
        np.save(
            os.path.join(self.output_path, f'num_running_{self.title}'),
            np.asarray(self.num_running)
        )
        np.save(
            os.path.join(self.output_path, f'num_waiting_{self.title}'),
            np.asarray(self.num_waiting)
        )
        np.save(
            os.path.join(self.output_path, f'num_preempted_{self.title}'),
            np.asarray(self.num_preempted)
        )
        np.save(
            os.path.join(self.output_path, f'finished_{self.title}'),
            np.asarray(self.finished)
        )
        np.save(
            os.path.join(self.output_path, f'processed_tokens_prompt_{self.title}'),
            np.asarray(self.processed_tokens_prompt)
        )
        np.save(
            os.path.join(self.output_path, f'processed_tokens_token_{self.title}'),
            np.asarray(self.processed_tokens_generation)
        )
