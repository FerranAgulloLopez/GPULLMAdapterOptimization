import copy
import shlex
import argparse
import asyncio
import json
import os
import csv
import random
import time
import warnings
import requests
import subprocess
from subprocess import Popen
from dataclasses import dataclass
from datetime import datetime
from typing import AsyncGenerator, List, Optional, Tuple, Dict, Set

import numpy as np
from backend_request_func import (ASYNC_REQUEST_FUNCS, RequestFuncInput,
                                  RequestFuncOutput)
from transformers import PreTrainedTokenizerBase

from vllm.transformers_utils.tokenizer import get_tokenizer
from benchmarks.lora.concurrent_metrics_checker import ConcurrentMetricsChecker
from benchmarks.lora.placement_algorithm.interface import \
    PlacementAlgorithmInterface
from benchmarks.lora.placement_algorithm.factory import \
    check_subclass as check_placement_algorithm_subclass, \
    get_subclass as get_placement_algorithm_subclass


@dataclass
class BenchmarkMetrics:
    completed: int
    completed_by_adapter: Dict[str, int]
    total_input: int
    total_output: int
    request_throughput: float
    input_throughput: float
    output_throughput: float
    mean_ttft_ms: float
    mean_ttfts_ms_by_adapter: Dict[str, float]
    median_ttft_ms: float
    p99_ttft_ms: float
    mean_tpot_ms: float
    median_tpot_ms: float
    p99_tpot_ms: float
    mean_itl_ms: float
    median_itl_ms: float
    p99_itl_ms: float


def sample_sharegpt_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int] = None,
) -> List[Tuple[str, int, int]]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [(data["conversations"][0]["value"],
                data["conversations"][1]["value"]) for data in dataset]

    # Shuffle the dataset.
    random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int, int]] = []
    for i in range(len(dataset)):
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = dataset[i][0]
        prompt_token_ids = tokenizer(prompt).input_ids
        completion = dataset[i][1]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = len(completion_token_ids
                         ) if fixed_output_len is None else fixed_output_len
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))

    return filtered_dataset


def calculate_metrics(
    input_requests: List[Tuple[str, int, int]],
    input_requests_loras: List[str],
    outputs: List[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
) -> Tuple[BenchmarkMetrics, List[int]]:
    actual_output_lens = []
    total_input = 0
    completed = 0
    completed_by_adapter: Dict[str, int] = {}
    itls = []
    tpots = []
    ttfts = []
    ttfts_by_lora: Dict[str, List[float]] = {}
    for i in range(len(outputs)):
        if outputs[i] is not None and outputs[i].success:
            # We use the tokenizer to count the number of output tokens for all
            # serving backends instead of looking at len(outputs[i].itl) since
            # multiple output tokens may be bundled together
            # Note: this may inflate the output token count slightly
            output_len = len(
                tokenizer(outputs[i].generated_text,
                          add_special_tokens=False).input_ids)
            actual_output_lens.append(output_len)
            total_input += input_requests[i][1]
            if output_len > 1:
                tpots.append(
                    (outputs[i].latency - outputs[i].ttft) / (output_len - 1))
            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)
            request_lora: str = input_requests_loras[i]
            if request_lora in ttfts_by_lora:
                ttfts_by_lora[request_lora] += [outputs[i].ttft]
            else:
                ttfts_by_lora[request_lora] = [outputs[i].ttft]
            completed += 1
            if request_lora not in completed_by_adapter:
                completed_by_adapter[request_lora] = 0
            completed_by_adapter[request_lora] += 1
        else:
            actual_output_lens.append(0)

    mean_ttfts_by_lora: Dict[str, float] = {}
    for user, values in ttfts_by_lora.items():
        mean_ttfts_by_lora[user] = np.mean(values or 0) * 1000

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
            stacklevel=2)
    metrics = BenchmarkMetrics(
        completed=completed,
        completed_by_adapter=completed_by_adapter,
        total_input=total_input,
        total_output=sum(actual_output_lens),
        request_throughput=completed / dur_s,
        input_throughput=total_input / dur_s,
        output_throughput=sum(actual_output_lens) / dur_s,
        mean_ttft_ms=np.mean(ttfts or 0) *
        1000,  # ttfts is empty if streaming is not supported by backend
        mean_ttfts_ms_by_adapter=mean_ttfts_by_lora,
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        p99_ttft_ms=np.percentile(ttfts or 0, 99) * 1000,
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        p99_tpot_ms=np.percentile(tpots or 0, 99) * 1000,
        mean_itl_ms=np.mean(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        p99_itl_ms=np.percentile(itls or 0, 99) * 1000,
    )

    return metrics, actual_output_lens


class Server:

    def __init__(
            self,
            server_args: str,
            output_path: str,
            port: int,
            server_index: int,
            gpu_id: str,
            max_loras: int,
            max_cpu_loras: int,
            max_lora_rank: int,
            lora_modules: List[Tuple[str, str]]
    ):
        super(Server, self).__init__()
        self.server_args = server_args
        self.output_path = output_path
        self.server_out = None
        self.server_err = None
        self.server_index = server_index
        self.port = port
        self.gpu_id = gpu_id
        self.max_loras = max_loras
        self.max_cpu_loras = max_cpu_loras
        self.max_lora_rank = max_lora_rank

        self.lora_modules = ''
        for adapter_id, adapter_path in lora_modules:
            self.lora_modules += f' {adapter_id}={adapter_path}'

        assert '--port' not in self.server_args
        assert '--enable-lora' not in self.server_args
        assert '--max-loras' not in self.server_args
        assert '--max-cpu-loras' not in self.server_args
        assert '--max-lora-rank' not in self.server_args
        assert '--lora-modules' not in self.server_args

    def run(self) -> Popen:
        try:
            self.server_out = open(os.path.join(self.output_path, f'server_out_{self.server_index}.log'), 'w')
            self.server_err = open(os.path.join(self.output_path, f'server_err_{self.server_index}.log'), 'w')
            command = f'' \
                      f'python3 -m vllm.entrypoints.openai.api_server ' \
                      f'--enable-lora ' \
                      f'--max-loras {self.max_loras} ' \
                      f'--max-cpu-loras {self.max_cpu_loras} ' \
                      f'--max-lora-rank {self.max_lora_rank} ' \
                      f'--lora-modules{self.lora_modules} ' \
                      f'--port={self.port} ' \
                      f'{self.server_args}'
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = self.gpu_id
            print('Run server with command', command)
            open_subprocess = subprocess.Popen(
                shlex.split(command),
                shell=False,
                cwd='/',
                stdout=self.server_out,
                stderr=self.server_err,
                env=env
            )
            return open_subprocess
        except Exception as e:
            print(e)
            if self.server_out:
                self.server_out.close()
            if self.server_err:
                self.server_err.close()
            raise e

    def terminate(self, open_subprocess: Popen) -> None:
        open_subprocess.kill()
        open_subprocess.terminate()
        open_subprocess.wait()
        if self.server_out:
            self.server_out.close()
        if self.server_err:
            self.server_err.close()


async def get_request(
    adapter_requests: List[Tuple[str, int, int]],
    adapter_rate: float,
    finish_time: float
) -> AsyncGenerator[Tuple[str, int, int], None]:
    request_iterator = iter(adapter_requests)

    #send first request after interval
    interval = np.random.exponential(1.0 / adapter_rate)
    if (interval + time.perf_counter()) >= finish_time:
        yield None, None, None, None
        return
    await asyncio.sleep(interval)

    interval = np.random.exponential(1.0 / adapter_rate)
    for index, (prompt, prompt_len, output_len) in enumerate(request_iterator):
        yield prompt, prompt_len, output_len, interval
        await asyncio.sleep(interval)
        interval = np.random.exponential(1.0 / adapter_rate)

    raise ValueError('Task without enough requests to send')


async def send_adapter_requests(
        adapter: str,
        start_time: float,
        finish_time: float,
        adapter_requests: List[Tuple[str, int, int]],
        adapter_rate: float,
        api_url: str,
        request_func
):
    inputs = []
    tasks = []
    arrivals = []
    await asyncio.sleep(start_time - time.perf_counter())
    async for (prompt, prompt_len, output_len, next_interval) in get_request(adapter_requests, adapter_rate, finish_time):
        if next_interval is None:
            break
        inputs.append((prompt, prompt_len, output_len))
        request_func_input = RequestFuncInput(
            model=adapter,
            prompt=prompt,
            api_url=api_url,
            prompt_len=prompt_len,
            output_len=output_len
        )
        tasks.append(
            asyncio.create_task(
                request_func(request_func_input=request_func_input)))
        arrivals.append([time.perf_counter(), prompt_len, output_len, adapter, api_url])
        if (time.perf_counter() + next_interval) >= finish_time:
            break

    await asyncio.sleep(finish_time - time.perf_counter())

    outputs = []
    for index in range(len(tasks)):
        if tasks[index].done():
            outputs.append(tasks[index].result())
        else:
            outputs.append(None)
        tasks[index].cancel()
    await asyncio.gather(*tasks)
    return inputs, outputs, arrivals


async def benchmark(
    backend: str,
    model_id: str,
    tokenizer: PreTrainedTokenizerBase,
    total_time: int,
    adapters: List[str],
    adapters_rates: List[float],
    adapters_prompts: List[List[Tuple[str, int, int]]],
    adapters_api_urls: List[str]
):
    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS.get(backend)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    test_prompt, test_prompt_len, test_output_len = adapters_prompts[0][0]
    unique_api_urls = set(adapters_api_urls)
    for api_url in unique_api_urls:
        print("Starting initial single prompt test run in base model for", api_url)
        test_input = RequestFuncInput(
            model=model_id,
            prompt=test_prompt,
            api_url=api_url,
            prompt_len=test_prompt_len,
            output_len=test_output_len,
        )
        test_output = await request_func(request_func_input=test_input)
        if not test_output.success:
            raise ValueError(
                "Initial test run failed - Please make sure benchmark arguments "
                f"are correctly specified. Error: {test_output.error}")

    waiting_time: float = 5  # to leave time for all adapters to get ready to send requests
    benchmark_start_time = time.perf_counter() + waiting_time
    benchmark_finish_time = benchmark_start_time + total_time
    init_time: float = time.perf_counter()
    tasks = []
    for adapter_index, adapter in enumerate(adapters):
        tasks.append(asyncio.create_task(
            send_adapter_requests(
                adapter=adapter,
                start_time=benchmark_start_time,
                finish_time=benchmark_finish_time,
                adapter_requests=adapters_prompts[adapter_index],
                adapter_rate=adapters_rates[adapter_index],
                api_url=adapters_api_urls[adapter_index],
                request_func=request_func
            )
        ))
    await asyncio.gather(*tasks)
    real_total_time: float = time.perf_counter() - (init_time + waiting_time)

    inputs = []
    inputs_adapters = []
    outputs = []
    all_arrivals = []
    for index in range(len(tasks)):
        if tasks[index].done():
            _input, output, arrivals = tasks[index].result()
            inputs_adapters += [adapters[index]] * len(_input)
            inputs += _input
            outputs += output
            all_arrivals += arrivals
        else:
            raise ValueError('Some adapter sending of request did not work properly')

    metrics, actual_output_lens = calculate_metrics(
        input_requests=inputs,
        input_requests_loras=inputs_adapters,
        outputs=outputs,
        dur_s=real_total_time,
        tokenizer=tokenizer,
    )
    result = {
        "start_time": benchmark_start_time,
        "end_time": benchmark_finish_time,
        "duration": real_total_time,
        "completed": metrics.completed,
        "completed_by_adapter": metrics.completed_by_adapter,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "input_throughput": metrics.input_throughput,
        "output_throughput": metrics.output_throughput,
        "mean_ttft_ms": metrics.mean_ttft_ms,
        "mean_ttfts_ms_by_adapter": metrics.mean_ttfts_ms_by_adapter,
        "median_ttft_ms": metrics.median_ttft_ms,
        "p99_ttft_ms": metrics.p99_ttft_ms,
        "mean_tpot_ms": metrics.mean_tpot_ms,
        "median_tpot_ms": metrics.median_tpot_ms,
        "p99_tpot_ms": metrics.p99_tpot_ms,
        "mean_itl_ms": metrics.mean_itl_ms,
        "median_itl_ms": metrics.median_itl_ms,
        "p99_itl_ms": metrics.p99_itl_ms,
    }

    print("{s:{c}^{n}}".format(s=' Serving Benchmark Intermediate Result ', n=50, c='='))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", real_total_time))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total generated tokens:",
                                 metrics.total_output))
    print("{:<40} {:<10.2f}".format("Request throughput (req/s):",
                                    metrics.request_throughput))
    print("{:<40} {:<10.2f}".format("Input token throughput (tok/s):",
                                    metrics.input_throughput))
    print("{:<40} {:<10.2f}".format("Output token throughput (tok/s):",
                                    metrics.output_throughput))
    print("{s:{c}^{n}}".format(s='Time to First Token', n=50, c='-'))
    print("{:<40} {:<10.2f}".format("Mean TTFT (ms):", metrics.mean_ttft_ms))
    print("{:<40} {:<10.2f}".format("Median TTFT (ms):",
                                    metrics.median_ttft_ms))
    print("{:<40} {:<10.2f}".format("P99 TTFT (ms):", metrics.p99_ttft_ms))
    print("{s:{c}^{n}}".format(s='Time per Output Token (excl. 1st token)',
                               n=50,
                               c='-'))
    print("{:<40} {:<10.2f}".format("Mean TPOT (ms):", metrics.mean_tpot_ms))
    print("{:<40} {:<10.2f}".format("Median TPOT (ms):",
                                    metrics.median_tpot_ms))
    print("{:<40} {:<10.2f}".format("P99 TPOT (ms):", metrics.p99_tpot_ms))
    print("{s:{c}^{n}}".format(s='Inter-token Latency', n=50, c='-'))
    print("{:<40} {:<10.2f}".format("Mean ITL (ms):", metrics.mean_itl_ms))
    print("{:<40} {:<10.2f}".format("Median ITL (ms):", metrics.median_itl_ms))
    print("{:<40} {:<10.2f}".format("P99 ITL (ms):", metrics.p99_itl_ms))
    print("=" * 50)
    return result, all_arrivals


def get_available_adapters_on_server(
        models_url: str,
        main_model_id: str
) -> Tuple[List[str], List[str]]:
    response = requests.get(models_url).json()
    available_loras: Dict[str, str] = {model['id']: model['root'] for model in response['data']}
    if main_model_id not in available_loras:
        raise ValueError('Benchmarking with a model that is not available in the server')
    del available_loras[main_model_id]
    if len(available_loras) == 0:
        raise ValueError("No available LoRAs in the server")
    available_loras_ids = list(available_loras.keys())
    available_loras_paths = list(available_loras.values())
    return available_loras_ids, available_loras_paths


def assign_ranks_to_adapters(
        adapters: List[str],
        values_to_use: List[str]
) -> Tuple[List[int], List[str]]:
    assert len(adapters) >= len(values_to_use)
    random.shuffle(values_to_use)

    # extract rank from each adapter
    values_to_use_ranks: List[int] = []
    for path in values_to_use:
        if os.path.exists(os.path.join(path, 'config.json')):
            adapter_config_path = os.path.join(path, 'config.json')
        elif os.path.exists(os.path.join(path, 'adapter_config.json')):
            adapter_config_path = os.path.join(path, 'adapter_config.json')
        else:
            raise ValueError('Provided adapter', path, 'has not config.json or adapter_config.json file')
        with open(adapter_config_path) as f:
            adapter_config = json.load(f)
        if 'rank' in adapter_config:
            adapter_rank = adapter_config['rank']
        elif 'r' in adapter_config:
            adapter_rank = adapter_config['r']
        else:
            raise ValueError('Provided adapter', path, 'has not rank specified in its configuration')
        values_to_use_ranks.append(adapter_rank)

    # distribute
    adapters_ranks: List[int] = []
    adapters_paths: List[str] = []
    index = 0
    while len(adapters_ranks) < len(adapters):
        adapters_ranks.append(values_to_use_ranks[index])
        adapters_paths.append(values_to_use[index])
        index += 1
        if index >= len(values_to_use):
            index = 0

    # shuffle
    aux_shuffled_list = list(zip(adapters_ranks, adapters_paths))
    random.shuffle(aux_shuffled_list)
    adapters_ranks, adapters_paths = zip(*aux_shuffled_list)

    return adapters_ranks, adapters_paths


def assign_rates_to_adapters(
        adapters: List[str],
        values_to_use: List[float]
) -> List[float]:
    assert len(adapters) >= len(values_to_use)
    random.shuffle(values_to_use)
    adapters_data: List[float] = []

    index = 0
    while len(adapters_data) < len(adapters):
        adapters_data.append(values_to_use[index])
        index += 1
        if index >= len(values_to_use):
            index = 0

    random.shuffle(adapters_data)
    return adapters_data


def distribute_aggregated_rate_to_adapters(
        adapters: List[str],
        aggregated_rate: float
) -> Tuple[List[str], List[float]]:
    adapters_rates: List[float] = [aggregated_rate / len(adapters)] * len(adapters)
    return adapters, adapters_rates


def create_adapter_prompts(
    total_time: int,
    adapters: List[str],
    adapters_rates: List[float],
    dataset_name: str,
    dataset_path: str,
    sharegpt_output_len: int,
    tokenizer
) -> List[List[Tuple[str, int, int]]]:
    # determine total prompts
    adapters_prompts_size: List[int] = []
    for rate in adapters_rates:
        adapters_prompts_size.append(max(1, round(total_time * rate)) * 3)  # *2 in case we need more due to random arrival intervals
    print("Adapter prompts.", adapters_prompts_size)

    # retrieve prompts
    total_prompts_size: int = sum(adapters_prompts_size)
    if dataset_name == "sharegpt":
        total_prompts: List[Tuple[str, int, int]] = sample_sharegpt_requests(
            dataset_path=dataset_path,
            num_requests=total_prompts_size,
            tokenizer=tokenizer,
            fixed_output_len=sharegpt_output_len,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    # if required, duplicate results until obtaining desired number of requests
    initial_length: int = len(total_prompts)
    while len(total_prompts) < total_prompts_size:
        index_to_duplicate: int = random.randint(0, initial_length - 1)  # always extract from initial set
        total_prompts.append(copy.deepcopy(total_prompts[index_to_duplicate]))
    random.shuffle(total_prompts)
    total_inputs_size: int = sum([input_tokens for _, input_tokens, _ in total_prompts])
    total_outputs_size: int = sum([output_tokens for _, _, output_tokens in total_prompts])
    print("Prompts retrieved:", len(total_prompts), ". Total input tokens:", total_inputs_size, ". Total output tokens:", total_outputs_size)

    # distribute prompts over adapters
    adapters_prompts: List[List[Tuple[str, int, int]]] = []
    global_index = 0
    for adapter_prompt_size in adapters_prompts_size:
        index = global_index
        grouped_prompts = []
        while index < len(total_prompts) and index < (global_index + adapter_prompt_size):
            grouped_prompts.append(total_prompts[index])
            index += 1
        global_index = index
        adapters_prompts.append(grouped_prompts)
    assert total_prompts_size == sum([len(adapter_prompts) for adapter_prompts in adapters_prompts])
    assert all([adapters_prompts_size[index] == len(adapters_prompts[index]) for index in range(len(adapters))])

    return adapters_prompts


def get_available_gpus():
    try:
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'])
        uuids = [x.strip() for x in output.decode().strip().split('\n')]
        ids = [str(index) for index in range(len(uuids))]
        return ids, uuids
    except Exception as e:
        print('Error fetching GPU UUIDs:', e)
        raise e


def get_base_url(host: str, port: int, server_id: int):
    return f"http://{host}:{port + server_id}"


def get_api_url(host: str, port: int, endpoint: str, server_id: int):
    return f"http://{host}:{port + server_id}{endpoint}"


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    backend = args.backend
    model_id = args.model
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model

    servers_processes = []
    open_server_processes = []
    concurrent_metrics_checkers = []
    try:
        # define adapters
        adapters: List[str] = [f'adapter_{index}' for index in range(args.num_adapters)]
        random.shuffle(adapters)

        # distribute ranks between adapters
        adapters_ranks, adapters_paths = assign_ranks_to_adapters(adapters, args.adapter_ranks)
        values, counts = np.unique(adapters_ranks, return_counts=True)
        print(f"Adapter ranks. Values: {values}. Counts: {counts}")

        # distribute rates between adapters
        if args.adapter_rates is None and args.aggregated_rate is None:
            raise ValueError('No adapter rate was determined, either use --adapter-rates or --aggregated-rate')
        if args.adapter_rates is not None and args.aggregated_rate is not None:
            raise ValueError('Both distinct rates and aggregated was determined, only use one option')
        if args.adapter_rates is not None:
            rates_to_use: List[float] = args.adapter_rates
            adapters_rates = assign_rates_to_adapters(adapters, rates_to_use)

        else:
            aggregated_rate: float = args.aggregated_rate
            adapters_rates = distribute_aggregated_rate_to_adapters(adapters, aggregated_rate)
        values, counts = np.unique(adapters_rates, return_counts=True)
        print(f"Adapter rates. Values: {values}. Counts: {counts}")

        # create adapter prompts
        total_time: int = args.total_time
        tokenizer = get_tokenizer(tokenizer_id, trust_remote_code=args.trust_remote_code)
        adapters_prompts: List[List[Tuple[str, int, int]]] = create_adapter_prompts(
            total_time=total_time,
            adapters=adapters,
            adapters_rates=adapters_rates,
            dataset_name=args.dataset_name,
            dataset_path=args.dataset_path,
            sharegpt_output_len=args.sharegpt_output_len,
            tokenizer=tokenizer
        )

        # retrieve overall input and output mean length
        count: int = 0
        mean_input_length: float = 0
        mean_output_length: float = 0
        for prompts in adapters_prompts:
            for _, prompt_len, output_len in prompts:
                mean_input_length += prompt_len
                mean_output_length += output_len
                count += 1
        mean_input_length /= count
        mean_output_length /= count
        print('Found mean input/output length', mean_input_length, mean_output_length)

        # define servers (check GPU availability)
        available_gpus, available_gpus_ids = get_available_gpus()
        if len(available_gpus) < args.num_servers:
            raise ValueError('Available GPUs are below set number of servers')
        servers: List[str] = available_gpus[:args.num_servers]
        print('Available GPUs.', 'Defined ids:', available_gpus, '. Full ids:', available_gpus_ids)
        print('Defined servers', servers)

        # retrieve placement from algorithm
        init_time: float = time.perf_counter()
        placement_algorithm: PlacementAlgorithmInterface = get_placement_algorithm_subclass(args.placement_algorithm)()
        try:
            servers_adapter_slots, adapters_servers = placement_algorithm.define_placement(
                os.path.basename(model_id),
                servers=servers,
                adapters=adapters,
                adapters_ranks=adapters_ranks,
                adapters_rates=adapters_rates,
                mean_input_length=mean_input_length,
                mean_output_length=mean_output_length
            )
            values, counts = np.unique(adapters_servers, return_counts=True)
            print(f'Output placement. Adapter slots by server: {servers_adapter_slots}. Server by adapter: -> Values: {values} Counts: {counts}')
        finally:
            print('Elapsed time during placement estimation:', time.perf_counter() - init_time)

        # check if any server is not being used, and act accordingly
        used_servers: Set[str] = set(values)
        if len(servers) > len(used_servers):
            new_servers: List[str] = []
            new_servers_adapter_slots: List[int] = []
            for index, server in enumerate(servers):
                if server in used_servers:
                    new_servers.append(server)
                    new_servers_adapter_slots.append(servers_adapter_slots[index])
            servers = new_servers
            servers_adapter_slots = new_servers_adapter_slots
            print('There were unused servers. The new list of servers is as follows:', servers)

        # set adapters api urls
        adapters_api_urls: List[str] = []
        server_indexes: Dict[str, int] = {server_id: server_index for server_index, server_id in enumerate(servers)}
        for adapter_server in adapters_servers:
            adapters_api_urls.append(get_api_url(args.host, args.port, args.endpoint, server_indexes[adapter_server]))

        # launch servers
        servers_adapters: Dict[str, List[Tuple[str, str]]] = {server_id: [] for server_id in servers}
        servers_max_lora_rank: Dict[str, int] = {server_id: None for server_id in servers}
        for index_adapter, adapter_server in enumerate(adapters_servers):
            assert adapter_server in servers_adapters
            servers_adapters[adapter_server].append((adapters[index_adapter], adapters_paths[index_adapter]))
            if servers_max_lora_rank[adapter_server] is None or servers_max_lora_rank[adapter_server] < adapters_ranks[index_adapter]:
                servers_max_lora_rank[adapter_server] = adapters_ranks[index_adapter]
        for index_server, server_id in enumerate(servers):
            server = Server(
                server_args=args.server_args,
                output_path=args.result_dir,
                server_index=index_server,
                port=args.port + index_server,
                gpu_id=server_id,
                max_loras=servers_adapter_slots[index_server],
                max_cpu_loras=max(len(servers_adapters[server_id]), servers_adapter_slots[index_server]),  # in vLLM max_cpu_loras >= max_loras
                max_lora_rank=servers_max_lora_rank[server_id],
                lora_modules=servers_adapters[server_id]
            )
            open_server_process = server.run()
            servers_processes.append(server)
            open_server_processes.append(open_server_process)

        # check servers started
        init_time = time.time()
        servers_to_start = {get_base_url(args.host, args.port, index_server) + "/metrics" for index_server in range(len(servers))}
        while len(servers_to_start) > 0 and time.time() - init_time < args.server_init_time:
            started_servers: List[str] = []
            for ping_url in servers_to_start:
                try:
                    if requests.get(ping_url).status_code == 200:
                        started_servers.append(ping_url)
                    else:
                        time.sleep(1)
                except Exception as e:
                    time.sleep(1)
            time.sleep(5)
            for started_server in started_servers:
                servers_to_start.remove(started_server)
        if len(servers_to_start) > 0:
            raise Exception('At least one server did not start on time')
        print('Servers started')

        # assure correct server configuration with adapters
        for index_server, server_id in enumerate(servers):
            available_adapters, available_adapters_paths = get_available_adapters_on_server(get_base_url(args.host, args.port, index_server) + '/v1/models', model_id)
            assert len(available_adapters) == len(servers_adapters[server_id])
            assert len(available_adapters_paths) == len(servers_adapters[server_id])
            for index_available_adapter in range(len(available_adapters)):  # a little too much constrained, also assures same order which does not make a lot of sense
                assert available_adapters[index_available_adapter] == servers_adapters[server_id][index_available_adapter][0]
                assert available_adapters_paths[index_available_adapter] == servers_adapters[server_id][index_available_adapter][1]

        # launch metric checkers
        if not args.disable_log_stats:
            for server_index in range(len(servers)):
                concurrent_metrics_checker = ConcurrentMetricsChecker(
                    args.result_dir,
                    get_base_url(args.host, args.port, server_index) + '/metrics',
                    title=f'server_{server_index}',
                )
                concurrent_metrics_checker.start()
                concurrent_metrics_checkers.append(concurrent_metrics_checker)
            print('Concurrent checkers started')

        # run benchmark
        benchmark_result, all_arrivals = asyncio.run(
            benchmark(
                backend=backend,
                model_id=model_id,
                tokenizer=tokenizer,
                total_time=total_time,
                adapters=adapters,
                adapters_rates=adapters_rates,
                adapters_prompts=adapters_prompts,
                adapters_api_urls=adapters_api_urls
            ))

        # save config and results to json
        if args.save_result:
            result_json = {}

            # Setup
            current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
            result_json["date"] = current_dt
            result_json["backend"] = backend
            result_json["model_id"] = model_id
            result_json["tokenizer_id"] = tokenizer_id
            result_json["total_prompts_sent"] = len(all_arrivals)
            total_prompts_sent_by_adapter: Dict[str, int] = {}
            for arrival_time, input_tokens, output_tokens, adapter, api_url in all_arrivals:
                if adapter not in total_prompts_sent_by_adapter:
                    total_prompts_sent_by_adapter[adapter] = 0
                total_prompts_sent_by_adapter[adapter] += 1
            result_json["total_prompts_sent_by_adapter"] = total_prompts_sent_by_adapter

            # Metadata
            if args.metadata:
                for item in args.metadata:
                    if "=" in item:
                        kvstring = item.split("=")
                        result_json[kvstring[0].strip()] = kvstring[1].strip()
                    else:
                        raise ValueError(
                            "Invalid metadata format. Please use KEY=VALUE format."
                        )

            # Adapters
            with open(os.path.join(args.result_dir, "adapters.csv"), mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["adapter_id", "adapter_path", "adapter_rank", "adapter_rate", "server", "server_api_url"])
                for index in range(len(adapters)):
                    writer.writerow([adapters[index], adapters_paths[index], adapters_ranks[index], adapters_rates[index], adapters_servers[index], adapters_api_urls[index]])

            # Arrivals
            with open(os.path.join(args.result_dir, "arrivals.csv"), mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["arrival_time", "input_tokens", "output_tokens", "adapter", "api_url"])
                writer.writerows(all_arrivals)

            # Traffic
            result_json["adapter_rates"] = adapters_rates

            # Merge with benchmark result
            result_json_final = {**result_json, **benchmark_result}

            # Save to file
            base_model_id = model_id.split("/")[-1]
            file_name = f"{backend}-{base_model_id}-{current_dt}.json"  #noqa
            if args.result_dir:
                file_name = os.path.join(args.result_dir, file_name)
            with open(file_name, "w") as outfile:
                json.dump(result_json_final, outfile)

    finally:
        if not args.disable_log_stats and len(concurrent_metrics_checkers) > 0:
            time.sleep(15)  # for correctly monitoring of metrics
            for concurrent_metrics_checker in concurrent_metrics_checkers:
                try:
                    concurrent_metrics_checker.terminate()
                    concurrent_metrics_checker.join()
                except Exception as e:
                    print(e)
            print('Concurrent checkers terminated')

        if len(servers_processes) > 0:
            for index, server in enumerate(servers_processes):
                try:
                    server.terminate(open_server_processes[index])
                except Exception as e:
                    print(e)
            print('Servers terminated')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput.")
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/completions",
        help="API endpoint.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to the ShareGPT dataset, will be deprecated in the "
        "next release.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="sharegpt",
        choices=["sharegpt"],
        help="Name of the dataset to benchmark on.",
    )
    parser.add_argument("--dataset-path",
                        type=str,
                        default=None,
                        help="Path to the dataset.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help=
        "Name or path of the tokenizer, if not using the default tokenizer.",
    )
    parser.add_argument(
        "--sharegpt-output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output length "
        "from the ShareGPT dataset.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from huggingface",
    )
    parser.add_argument(
        "--save-result",
        action="store_true",
        help="Specify to save benchmark results to a json file",
    )
    parser.add_argument(
        "--metadata",
        metavar="KEY=VALUE",
        nargs="*",
        help="Key-value pairs (e.g, --metadata version=0.3.3 tp=1) "
        "for metadata of this run to be saved in the result JSON file "
        "for record keeping purposes.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        help="Specify directory to save benchmark json results."
        "If not specified, results are saved in the current directory.",
    )
    parser.add_argument(
        "--server-args",
        type=str,
        default="",
        help="Args to send to the servers when launching",
    )
    parser.add_argument(
        "--server-init-time",
        type=int,
        default=300,
        help="Timeout for server initialization",
    )
    parser.add_argument('--disable-log-stats',
                        action='store_true',
                        help='disable logging statistics'
                        )
    parser.add_argument(
        "--total-time",
        type=int,
        required=True,
        help="Total time in seconds.",
    )
    parser.add_argument(
        "--num-servers",
        type=int,
        required=True,
        help="Number of servers to use.",
    )
    parser.add_argument(
        "--num-adapters",
        type=int,
        required=True,
        help="Number of adapters to serve.",
    )
    parser.add_argument(
        "--adapter-ranks",
        type=str,
        required=True,
        help="Adapter rank paths to use.",
    )
    parser.add_argument(
        "--adapter-rates",
        type=str,
        required=False,
        default=None,
        help="Adapter rates to use.",
    )
    parser.add_argument(
        "--aggregated-rate",
        type=float,
        required=False,
        default=None,
        help="Use an aggregated rate to split between all adapters.",
    )
    parser.add_argument(
        "--placement-algorithm",
        type=str,
        required=True,
        help="Placement algorithm to use.",
    )

    args = parser.parse_args()
    args.adapter_ranks = [item for item in args.adapter_ranks.split(' ')]
    if args.adapter_rates is not None:
        args.adapter_rates = [float(item) for item in args.adapter_rates.split(' ')]
    check_placement_algorithm_subclass(args.placement_algorithm)
    main(args)
