import shlex
import argparse
import asyncio
import json
import os
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
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase

from vllm.transformers_utils.tokenizer import get_tokenizer
from concurrent_metrics_checker import ConcurrentMetricsChecker


# TODO delete users logic


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
    mean_ttfts_ms_by_user: Dict[str, float]
    median_ttft_ms: float
    p99_ttft_ms: float
    mean_tpot_ms: float
    median_tpot_ms: float
    p99_tpot_ms: float
    mean_itl_ms: float
    median_itl_ms: float
    p99_itl_ms: float


def generate_synthetic_dataset(
    words_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    chosen_input_length: int,
    chosen_output_length: int
) -> List[Tuple[str, int, int]]:
    # retrieve all available words
    with open(words_path) as f:
        words = f.read().splitlines()

    # filter out weird words
    words = [word for word in words if word.isalpha() and word.islower()]
    print('All available words:', len(words))

    # generate prompt
    # We do not generate more than one, because we want all requests to have the same input tokens
    # With the tokenizer, different words have different number of tokens
    prompt = ' '.join(random.choice(words) for _ in range(chosen_input_length))
    prompt = prompt.capitalize() + '.'
    prompt_token_ids = tokenizer(prompt).input_ids
    prompt_len = len(prompt_token_ids)
    print(f'Generated prompt with {prompt_len} input length:', prompt)

    # generate prompts
    filtered_dataset: List[Tuple[str, int, int]] = [(prompt, prompt_len, chosen_output_length)] * num_requests

    return filtered_dataset


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


def sample_sonnet_requests(
    dataset_path: str,
    num_requests: int,
    input_len: int,
    output_len: int,
    prefix_len: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, str, int, int]]:
    assert (
        input_len > prefix_len
    ), "'args.sonnet-input-len' must be greater than 'args.prefix-input-len'."

    # Load the dataset.
    with open(dataset_path) as f:
        poem_lines = f.readlines()

    # Tokenize the poem lines.
    poem_token_ids = tokenizer(poem_lines).input_ids
    average_poem_len = sum(
        len(token_ids) for token_ids in poem_token_ids) / len(poem_token_ids)

    # Base prefix for all requests.
    base_prompt = "Pick as many lines as you can from these poem lines:\n"
    base_message = [{
        "role": "user",
        "content": base_prompt,
    }]
    base_prompt_formatted = tokenizer.apply_chat_template(
        base_message, add_generation_prompt=True, tokenize=False)
    base_prompt_offset = len(tokenizer(base_prompt_formatted).input_ids)

    assert (
        input_len > base_prompt_offset
    ), f"Please set 'args.sonnet-input-len' higher than {base_prompt_offset}."
    num_input_lines = round(
        (input_len - base_prompt_offset) / average_poem_len)

    # First approximately `prefix_len` number of tokens in the
    # prompt are fixed poem lines.
    assert (
        prefix_len > base_prompt_offset
    ), f"Please set 'args.sonnet-prefix-len' higher than {base_prompt_offset}."

    num_prefix_lines = round(
        (prefix_len - base_prompt_offset) / average_poem_len)
    prefix_lines = poem_lines[:num_prefix_lines]

    # Sample the rest of lines per request.
    sampled_requests: List[Tuple[str, int, int]] = []
    for _ in range(num_requests):
        sampled_lines = "".join(
            prefix_lines +
            random.sample(poem_lines, num_input_lines - num_prefix_lines))

        prompt = f"{base_prompt}{sampled_lines}"
        message = [
            {
                "role": "user",
                "content": prompt,
            },
        ]
        prompt_formatted = tokenizer.apply_chat_template(
            message, add_generation_prompt=True, tokenize=False)
        prompt_len = len(tokenizer(prompt_formatted).input_ids)
        sampled_requests.append(
            (prompt, prompt_formatted, prompt_len, output_len))

    return sampled_requests


async def get_request(
    input_requests: List[Tuple[str, int, int]],
    input_requests_loras: List[str],
    input_requests_users: List[str],
    request_rate: float,
) -> AsyncGenerator[Tuple[str, int, int, str, str], None]:
    input_requests = iter(input_requests)
    for index, (prompt, prompt_len, output_len) in enumerate(input_requests):
        yield prompt, prompt_len, output_len, input_requests_loras[index], input_requests_users[index]

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue
        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


def calculate_metrics(
    input_requests: List[Tuple[str, int, int]],
    input_requests_loras: List[str],
    input_requests_users: List[str],
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
    ttfts_by_user: Dict[str, List[float]] = {}
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
            request_user: str = input_requests_users[i]
            if request_user in ttfts_by_user:
                ttfts_by_user[request_user] += [outputs[i].ttft]
            else:
                ttfts_by_user[request_user] = [outputs[i].ttft]
            completed += 1
        else:
            actual_output_lens.append(0)
            ttfts.append(dur_s)
            request_lora: str = input_requests_loras[i]
            if request_lora in ttfts_by_lora:
                ttfts_by_lora[request_lora] += [dur_s]
            else:
                ttfts_by_lora[request_lora] = [dur_s]
            request_user: str = input_requests_users[i]
            if request_user in ttfts_by_user:
                ttfts_by_user[request_user] += [dur_s]
            else:
                ttfts_by_user[request_user] = [dur_s]

    mean_ttfts_by_lora: Dict[str, float] = {}
    for user, values in ttfts_by_lora.items():
        mean_ttfts_by_lora[user] = np.mean(values or 0) * 1000
    mean_ttfts_by_user: Dict[str, float] = {}
    for user, values in ttfts_by_user.items():
        mean_ttfts_by_user[user] = np.mean(values or 0) * 1000

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
        mean_ttfts_ms_by_user=mean_ttfts_by_user,
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


async def benchmark(
    backend: str,
    api_url: str,
    model_id: str,
    tokenizer: PreTrainedTokenizerBase,
    input_requests: List[Tuple[str, int, int]],
    input_requests_loras: List[str],
    input_requests_users: List[str],
    request_rate: float,
    disable_tqdm: bool,
    lora_pre_loading: bool,
):
    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS.get(backend)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    if not lora_pre_loading:
        print("Starting initial single prompt test run for base model")
        test_prompt, test_prompt_len, test_output_len = input_requests[0]
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
        test_prompt, test_prompt_len, test_output_len = input_requests[0]
        loras = set(input_requests_loras)
        for lora in loras:
            test_input = RequestFuncInput(
                model=lora,
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

    print(f"Traffic request rate: {request_rate}")

    pbar = None if disable_tqdm else tqdm(total=len(input_requests))

    benchmark_start_time = time.perf_counter()
    tasks = []
    async for request in get_request(input_requests, input_requests_loras, input_requests_users, request_rate):
        prompt, prompt_len, output_len, lora, user = request
        request_func_input = RequestFuncInput(
            model=lora,
            prompt=prompt,
            api_url=api_url,
            prompt_len=prompt_len,
            output_len=output_len,
        )
        tasks.append(
            asyncio.create_task(
                request_func(request_func_input=request_func_input,
                             pbar=pbar)))

    outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)

    if not disable_tqdm:
        pbar.close()

    benchmark_finish_time = time.perf_counter()
    benchmark_duration = benchmark_finish_time - benchmark_start_time

    metrics, actual_output_lens = calculate_metrics(
        input_requests=input_requests,
        input_requests_loras=input_requests_loras,
        input_requests_users=input_requests_users,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
    )

    print("{s:{c}^{n}}".format(s=' Serving Benchmark Result ', n=50, c='='))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):",
                                    benchmark_duration))
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

    result = {
        "start_time": benchmark_start_time,
        "end_time": benchmark_finish_time,
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "input_throughput": metrics.input_throughput,
        "output_throughput": metrics.output_throughput,
        "mean_ttft_ms": metrics.mean_ttft_ms,
        "mean_ttfts_ms_by_lora": metrics.mean_ttfts_ms_by_lora,
        "mean_ttfts_ms_by_user": metrics.mean_ttfts_ms_by_user,
        "median_ttft_ms": metrics.median_ttft_ms,
        "p99_ttft_ms": metrics.p99_ttft_ms,
        "mean_tpot_ms": metrics.mean_tpot_ms,
        "median_tpot_ms": metrics.median_tpot_ms,
        "p99_tpot_ms": metrics.p99_tpot_ms,
        "mean_itl_ms": metrics.mean_itl_ms,
        "median_itl_ms": metrics.median_itl_ms,
        "p99_itl_ms": metrics.p99_itl_ms,
    }
    return result


def main(args: argparse.Namespace):
    def __get_available_loras(
            models_url: str,
            main_model_id: str
    ) -> List[str]:
        response = requests.get(models_url).json()
        print('Models', response)
        available_loras = [model["id"] for model in response["data"]]
        if main_model_id not in available_loras:
            raise ValueError("Benchmarking with a model that is not available in the server")
        available_loras.remove(main_model_id)
        if len(available_loras) == 0:
            raise ValueError("No available LoRAs in the server")
        return available_loras

    def __assign_users_loras(
            input_requests: List[Tuple[str, int, int]],
            user_lora_request_relation: str,
            available_loras: List[str],
    ) -> Tuple[List[str], List[str]]:
        input_requests_users: List[str] = []
        input_requests_loras: List[str] = []

        if user_lora_request_relation is None or user_lora_request_relation in ['default', 'balance']:
            index = 0
            while len(input_requests_users) < len(input_requests):
                input_requests_users.append(str(index))
                input_requests_loras.append(available_loras[index])
                index += 1
                if index >= len(available_loras):
                    index = 0
            aux_shuffled_list = list(zip(input_requests_users, input_requests_loras))
            random.shuffle(aux_shuffled_list)
            input_requests_users, input_requests_loras = zip(*aux_shuffled_list)
        elif user_lora_request_relation == 'imbalance':
            index = 0
            while len(input_requests_users) < len(input_requests):
                input_requests_users.append(str(index))
                input_requests_loras.append(available_loras[index])
                if index % 2 == 0 and len(input_requests_users) < len(input_requests):
                    input_requests_users.append(str(index))
                    input_requests_loras.append(available_loras[index])
                index += 1
                if index >= len(available_loras):
                    index = 0
            aux_shuffled_list = list(zip(input_requests_users, input_requests_loras))
            random.shuffle(aux_shuffled_list)
            input_requests_users, input_requests_loras = zip(*aux_shuffled_list)
        else:
            raise ValueError(f"User assignation {user_lora_request_relation} not implemented")

        values, counts = np.unique(input_requests_users, return_counts=True)
        print(f"Requests users. Values: {values}. Counts: {counts}")
        values, counts = np.unique(input_requests_loras, return_counts=True)
        print(f"Requests loras. Values: {values}. Counts: {counts}")
        return input_requests_users, input_requests_loras

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

    tokenizer = get_tokenizer(tokenizer_id,
                              trust_remote_code=args.trust_remote_code)

    if args.use_synthetic_dataset is not None:
        if args.synthetic_input_length is None or args.synthetic_output_length is None:
            raise ValueError('When using a synthetic dataset, the input and output length should be define')
        input_requests = generate_synthetic_dataset(
            words_path=args.use_synthetic_dataset,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            chosen_input_length=args.synthetic_input_length,
            chosen_output_length=args.synthetic_output_length
        )

    else:
        if args.dataset is not None:
            warnings.warn(
                "The '--dataset' argument will be deprecated in the next "
                "release. Please use '--dataset-name' and "
                "'--dataset-path' in the future runs.",
                stacklevel=2)
            input_requests = sample_sharegpt_requests(
                dataset_path=args.dataset,
                num_requests=args.num_prompts,
                tokenizer=tokenizer,
                fixed_output_len=args.sharegpt_output_len,
            )

        elif args.dataset_name == "sharegpt":
            input_requests = sample_sharegpt_requests(
                dataset_path=args.dataset_path,
                num_requests=args.num_prompts,
                tokenizer=tokenizer,
                fixed_output_len=args.sharegpt_output_len,
            )

        elif args.dataset_name == "sonnet":
            # Do not format the prompt, pass to message directly
            if args.backend == "openai-chat":
                input_requests = sample_sonnet_requests(
                    dataset_path=args.dataset_path,
                    num_requests=args.num_prompts,
                    input_len=args.sonnet_input_len,
                    output_len=args.sonnet_output_len,
                    prefix_len=args.sonnet_prefix_len,
                    tokenizer=tokenizer,
                )
                input_requests = [(prompt, prompt_len, output_len)
                                  for prompt, prompt_formatted, prompt_len,
                                  output_len in input_requests]
            else:
                assert (
                    tokenizer.chat_template or tokenizer.default_chat_template
                ), "Tokenizer/model must have chat template for sonnet dataset."
                input_requests = sample_sonnet_requests(
                    dataset_path=args.dataset_path,
                    num_requests=args.num_prompts,
                    input_len=args.sonnet_input_len,
                    output_len=args.sonnet_output_len,
                    prefix_len=args.sonnet_prefix_len,
                    tokenizer=tokenizer,
                )
                input_requests = [(prompt_formatted, prompt_len, output_len)
                                  for prompt, prompt_formatted, prompt_len,
                                  output_len in input_requests]

        else:
            raise ValueError(f"Unknown dataset: {args.dataset_name}")

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

        request_rate = args.request_rate
        if args.disable_loras_users or args.restrict_loras is not None and args.restrict_loras == 0:
            input_requests_users: List[str] = ["0"] * len(input_requests)
            all_available_loras: List[str] = []
            input_requests_loras: List[str] = [model_id] * len(input_requests)
        else:
            all_available_loras: List[str] = __get_available_loras(models_api_url, model_id)
            if args.restrict_loras is None:
                available_loras = all_available_loras
            else:
                if len(all_available_loras) < args.restrict_loras:
                    raise ValueError('Less available LoRAs than the ones that need to be restricted')
                available_loras = all_available_loras[:args.restrict_loras]
            input_requests_users, input_requests_loras = __assign_users_loras(
                input_requests,
                args.user_lora_request_relation,
                available_loras
            )
            if args.request_rate_by_lora is not None:
                request_rate = args.request_rate_by_lora * len(available_loras)

        if not args.disable_log_stats:
            concurrent_metrics_checker = ConcurrentMetricsChecker(
                args.result_dir,
                metrics_api_url,
                all_available_loras
            )
            concurrent_metrics_checker.start()

        benchmark_result = asyncio.run(
            benchmark(
                backend=backend,
                api_url=api_url,
                model_id=model_id,
                tokenizer=tokenizer,
                input_requests=input_requests,
                input_requests_loras=input_requests_loras,
                input_requests_users=input_requests_users,
                request_rate=request_rate,
                disable_tqdm=args.disable_tqdm,
                lora_pre_loading=args.lora_pre_loading,
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
            result_json["num_prompts"] = args.num_prompts

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

            # Traffic
            result_json["request_rate"] = (
                args.request_rate if request_rate < float("inf") else "inf")

            # Merge with benchmark result
            result_json_final = {**result_json, **benchmark_result}

            # Save to file
            base_model_id = model_id.split("/")[-1]
            file_name = f"{backend}-{args.request_rate}qps-{base_model_id}-{current_dt}.json"  #noqa
            if args.result_dir:
                file_name = os.path.join(args.result_dir, file_name)
            with open(file_name, "w") as outfile:
                json.dump(result_json_final, outfile)
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
        choices=["sharegpt", "sonnet"],
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
        "--num-prompts",
        type=int,
        default=1000,
        help="Number of prompts to process.",
    )
    parser.add_argument(
        "--sharegpt-output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output length "
        "from the ShareGPT dataset.")
    parser.add_argument(
        "--sonnet-input-len",
        type=int,
        default=550,
        help=
        "Number of input tokens per request, used only for sonnet dataset.",
    )
    parser.add_argument(
        "--sonnet-output-len",
        type=int,
        default=150,
        help=
        "Number of output tokens per request, used only for sonnet dataset.",
    )
    parser.add_argument(
        "--sonnet-prefix-len",
        type=int,
        default=200,
        help=
        "Number of prefix tokens per request, used only for sonnet dataset.",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process to synthesize "
        "the request arrival times.",
    )
    parser.add_argument(
        "--request-rate-by-lora",
        type=float,
        default=None,
        help="Number of requests per second by LoRA. If this is inf, "
             "then all the requests are sent at time 0. "
             "Otherwise, we use Poisson process to synthesize "
             "the request arrival times.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from huggingface",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Specify to disable tqdm progress bar.",
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
    parser.add_argument('--disable-loras-users',
                        action='store_true',
                        help='Only send not LORA requests'
                        )
    parser.add_argument(
        "--user-lora-request-relation",
        type=str,
        default=None,
        help="Relation of lora<->request<->user",
    )
    parser.add_argument(
        "--restrict-loras",
        type=int,
        default=None,
        help="Do not send to all available loras.",
    )
    parser.add_argument('--lora-pre-loading',
                        action='store_true',
                        help='Pre load all LoRA adapters in the test previous to the real benchmark. It is not needed, vLLM does it by default',
                        default=False
                        )
    parser.add_argument('--use-synthetic-dataset',
                        type=str,
                        default=None,
                        help='Use a synthetic dataset',
                        )
    parser.add_argument(
        "--synthetic-input-length",
        type=int,
        default=None,
        help="Define synthetic dataset input length.",
    )
    parser.add_argument(
        "--synthetic-output-length",
        type=int,
        default=None,
        help="Define synthetic dataset output length.",
    )

    args = parser.parse_args()
    main(args)
