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
from typing import AsyncGenerator, List, Optional, Tuple, Dict

import numpy as np
from backend_request_func import (ASYNC_REQUEST_FUNCS, RequestFuncInput,
                                  RequestFuncOutput)
from transformers import PreTrainedTokenizerBase

from concurrent_metrics_checker import ConcurrentMetricsChecker
from concurrent_metrics_checker_extended import ConcurrentMetricsCheckerExtended


@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_output: int
    request_throughput: float
    input_throughput: float
    output_throughput: float
    mean_ttft_ms: float
    mean_ttfts_ms_by_lora: Dict[str, float]
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
        total_input=total_input,
        total_output=sum(actual_output_lens),
        request_throughput=completed / dur_s,
        input_throughput=total_input / dur_s,
        output_throughput=sum(actual_output_lens) / dur_s,
        mean_ttft_ms=np.mean(ttfts or 0) *
        1000,  # ttfts is empty if streaming is not supported by backend
        mean_ttfts_ms_by_lora=mean_ttfts_by_lora,
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

    def __init__(self, server_args: str, output_path: str):
        super(Server, self).__init__()
        self.server_args = server_args
        self.output_path = output_path
        self.server_out = None
        self.server_err = None

    def run(self) -> Popen:
        try:
            self.server_out = open(os.path.join(self.output_path, 'server_out.log'), 'w')
            self.server_err = open(os.path.join(self.output_path, 'server_err.log'), 'w')
            command = f'python3 -m vllm.entrypoints.openai.api_server {self.server_args}'
            open_subprocess = subprocess.Popen(
                shlex.split(command),
                shell=False,
                cwd='/',
                stdout=self.server_out,
                stderr=self.server_err
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
        arrivals.append([time.perf_counter(), prompt_len, output_len, adapter])
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
    api_url: str,
    model_id: str,
    tokenizer: PreTrainedTokenizerBase,
    total_time: int,
    adapters: List[str],
    adapters_rates: List[float],
    adapters_prompts: List[List[Tuple[str, int, int]]],
    lora_pre_loading: bool,
):
    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS.get(backend)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    if not lora_pre_loading:
        print("Starting initial single prompt test run for base model")
        test_prompt, test_prompt_len, test_output_len = adapters_prompts[0][0]
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
        else:
            print("Initial test run completed. Starting main benchmark run...")
    else:
        print("Starting initial prompt test run for every adapter")
        test_prompt, test_prompt_len, test_output_len = adapters_prompts[0][0]
        for adapter in adapters:
            test_input = RequestFuncInput(
                model=adapter,
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
        print("Initial test run completed. Starting main benchmark run...")

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
                api_url=api_url,
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
            inputs += _input
            inputs_adapters += [adapters[index]] * len(inputs)
            outputs += output
            #arrivals_by_adapter.append(arrivals)
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
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "input_throughput": metrics.input_throughput,
        "output_throughput": metrics.output_throughput,
        "mean_ttft_ms": metrics.mean_ttft_ms,
        "mean_ttfts_ms_by_lora": metrics.mean_ttfts_ms_by_lora,
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
) -> Tuple[List[str], Dict[str, str]]:
    response = requests.get(models_url).json()
    available_loras_paths: Dict[str, str] = {model["id"]: model["root"] for model in response["data"]}
    available_loras = list(available_loras_paths.keys())
    if main_model_id not in available_loras:
        raise ValueError("Benchmarking with a model that is not available in the server")
    available_loras.remove(main_model_id)
    if len(available_loras) == 0:
        raise ValueError("No available LoRAs in the server")
    return available_loras, available_loras_paths


def assign_rates_to_adapters(
        adapters: List[str],
        rates_to_use: List[float]
) -> Tuple[List[str], List[float]]:
    assert len(adapters) >= len(rates_to_use)
    adapters_rates: List[float] = []

    index = 0
    while len(adapters_rates) < len(adapters):
        adapters_rates.append(rates_to_use[index])
        index += 1
        if index >= len(rates_to_use):
            index = 0
    aux_shuffled_list = list(zip(adapters, adapters_rates))
    random.shuffle(aux_shuffled_list)
    adapters, adapters_rates = zip(*aux_shuffled_list)

    values, counts = np.unique(adapters_rates, return_counts=True)
    print(f"Adapter rates. Values: {values}. Counts: {counts}")
    return adapters, adapters_rates


def distribute_aggregated_rate_to_adapters(
        adapters: List[str],
        aggregated_rate: float
) -> Tuple[List[str], List[float]]:
    adapters_rates: List[float] = [aggregated_rate / len(adapters)] * len(adapters)
    values, counts = np.unique(adapters_rates, return_counts=True)
    print(f"Adapter rates. Values: {values}. Counts: {counts}")
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
    # if required duplicate results until obtaining desired number of requests
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
    print("Prompts distributed")

    return adapters_prompts


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    backend = args.backend
    model_id = args.model
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model

    if args.base_url is not None:
        api_url = f"{args.base_url}{args.endpoint}"
        metrics_api_url = f"{args.base_url}/metrics/"
        models_api_url = f"{args.base_url}/models/"
    else:
        api_url = f"http://{args.host}:{args.port}{args.endpoint}"
        metrics_api_url = f"http://{args.host}:{args.port}/metrics/"
        models_api_url = f"http://{args.host}:{args.port}/v1/models/"

    server = None
    open_server_process = None
    concurrent_metrics_checker = None
    try:
        if args.launch_server:
            server = Server(args.server_args, args.result_dir)
            open_server_process = server.run()

            init_time = time.time()
            server_started = False
            while not server_started and time.time() - init_time < args.server_init_time:
                try:
                    if requests.get(metrics_api_url).status_code == 200:
                        server_started = True
                    else:
                        time.sleep(5)
                except Exception as e:
                    time.sleep(5)
            if not server_started:
                raise Exception("Server did not start on time")
            print("Server started")

        # get available adapters
        adapters, adapters_paths = get_available_adapters_on_server(models_api_url, model_id)
        print("Number of adapters:", len(adapters))

        # distribute rates between adapters
        if args.adapter_rates is None and args.aggregated_rate is None:
            raise ValueError('No adapter rate was determined, either use --adapter-rates or --aggregated-rate')
        if args.adapter_rates is not None and args.aggregated_rate is not None:
            raise ValueError('Both distinct rates and aggregated was determined, only use one option')
        if args.adapter_rates is not None:
            rates_to_use: List[float] = args.adapter_rates
            adapters, adapters_rates = assign_rates_to_adapters(adapters, rates_to_use)
        else:
            aggregated_rate: float = args.aggregated_rate
            adapters, adapters_rates = distribute_aggregated_rate_to_adapters(adapters, aggregated_rate)

        # create adapter prompts
        total_time: int = args.total_time
        from vllm.transformers_utils.tokenizer import get_tokenizer
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

        if not args.disable_log_stats:
            if args.use_extended_metric_checker:
                concurrent_metrics_checker = ConcurrentMetricsCheckerExtended(
                    args.result_dir,
                    metrics_api_url,
                    adapters
                )
            else:
                concurrent_metrics_checker = ConcurrentMetricsChecker(
                    args.result_dir,
                    metrics_api_url,
                    adapters
                )
            concurrent_metrics_checker.start()
            print("Concurrent checker started")

        benchmark_result, all_arrivals = asyncio.run(
            benchmark(
                backend=backend,
                api_url=api_url,
                model_id=model_id,
                tokenizer=tokenizer,
                total_time=total_time,
                adapters=adapters,
                adapters_rates=adapters_rates,
                adapters_prompts=adapters_prompts,
                lora_pre_loading=args.lora_pre_loading
            ))

        # Save config and results to json
        if args.save_result:
            result_json = {}

            # Setup
            current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
            result_json["date"] = current_dt
            result_json["backend"] = backend
            result_json["model_id"] = model_id
            result_json["tokenizer_id"] = tokenizer_id
            result_json["total_prompts_sent"] = len(all_arrivals)

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
                writer.writerow(["adapter_id", "adapter_path"])
                for index in range(len(adapters)):
                    writer.writerow([adapters[index], adapters_paths[adapters[index]]])

            # Arrivals
            with open(os.path.join(args.result_dir, "arrivals.csv"), mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["arrival_time", "input_tokens", "output_tokens", "adapter"])
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

            '''# Save arrivals
            for index, arrivals in enumerate(arrivals_by_adapter):
                np.save(os.path.join(args.result_dir, f'arrival_by_adapter_{index}'), arrivals)'''
    finally:
        try:
            if not args.disable_log_stats and concurrent_metrics_checker is not None:
                time.sleep(15)  # for correctly monitoring of metrics
                concurrent_metrics_checker.terminate()
                concurrent_metrics_checker.join()
                print('Concurrent checker terminated')
        except Exception as e:
            print(e)
        try:
            if args.launch_server and server and open_server_process:
                server.terminate(open_server_process)
                print('Server terminated')
        except Exception as e:
            print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput.")
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server or API base url if not using http host and port.",
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
        "--launch-server",
        action="store_true",
        help="Launch server in addition to benchmark",
    )
    parser.add_argument(
        "--server-args",
        type=str,
        default="",
        help="Args to send to the server when launching. Only useful when passing --launch-server as well",
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
        "--use-extended-metric-checker",
        default=False,
        action="store_true",
        help="Use extended concurrent metric checker to extract runtime metric values",
    )
    parser.add_argument(
        '--lora-pre-loading',
        action='store_true',
        help='Pre load all LoRA adapters in the test previous to the real benchmark. It is not needed, vLLM does it by default',
        default=False
    )

    args = parser.parse_args()
    if args.adapter_rates is not None:
        args.adapter_rates = [float(item) for item in args.adapter_rates.split(' ')]
    main(args)
