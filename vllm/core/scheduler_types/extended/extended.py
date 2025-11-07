from typing import Dict, Optional

from prometheus_client import (Gauge)

from vllm.config import CacheConfig, LoRAConfig, SchedulerConfig
from vllm.core.scheduler import Scheduler
from vllm.engine.metrics import StatLogger
from vllm.logger import init_logger

logger = init_logger(__name__)


class ExtendedScheduler(Scheduler):

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        lora_config: Optional[LoRAConfig],
    ) -> None:
        super().__init__(scheduler_config, cache_config, lora_config)
        if lora_config is None:
            raise ValueError("This scheduler only works with LoRA enabled")
        self.extended_initialized_visualization: bool = False
        self.max_cpu_loras = lora_config.max_cpu_loras

    def do_alternative_logging(self, stat_logger: StatLogger) -> None:
        super().do_alternative_logging(stat_logger)
        if not self.extended_initialized_visualization:
            stat_logger.metrics.extended_running_by_adapter = {}
            stat_logger.metrics.extended_waiting_by_adapter = {}
            self.extended_initialized_visualization = True

        extended_running_by_adapter: Dict[int, int] = {}
        extended_waiting_by_adapter: Dict[int, int] = {}
        for index in range(self.max_cpu_loras + 1):
            extended_running_by_adapter[index] = 0
            extended_waiting_by_adapter[index] = 0

        for seq_group in self.running:
            lora_int_id = seq_group.lora_int_id
            extended_running_by_adapter[lora_int_id] += 1

        for seq_group in self.waiting:
            lora_int_id = seq_group.lora_int_id
            extended_waiting_by_adapter[lora_int_id] += 1

        for lora_int_id, running in extended_running_by_adapter.items():
            if lora_int_id not in stat_logger.metrics.extended_running_by_adapter:
                stat_logger.metrics.extended_running_by_adapter[lora_int_id] = Gauge(
                    name=f"vllm:running_by_adapter_{lora_int_id}",
                    documentation=f"Number of running requests of adapter {lora_int_id}.",
                    labelnames=list(stat_logger.labels.keys())
                )
            stat_logger.metrics.extended_running_by_adapter[lora_int_id].labels(**stat_logger.labels).set(running)

        for lora_int_id, waiting in extended_waiting_by_adapter.items():
            if lora_int_id not in stat_logger.metrics.extended_waiting_by_adapter:
                stat_logger.metrics.extended_waiting_by_adapter[lora_int_id] = Gauge(
                    name=f"vllm:waiting_by_adapter_{lora_int_id}",
                    documentation=f"Number of waiting requests of adapter {lora_int_id}.",
                    labelnames=list(stat_logger.labels.keys())
                )
            stat_logger.metrics.extended_waiting_by_adapter[lora_int_id].labels(**stat_logger.labels).set(waiting)
