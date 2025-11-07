import time
import random
from interface import PlacementAlgorithmInterface
from factory import get_subclass
import numpy as np
from typing import Dict, List

from benchmarks.lora.predict_digital_twin import predict_digital_twin
from benchmarks.lora.deployment.slurm.launcher_simulator_from_scratch import get_gpu_memory_availability

ML_MODEL_PATH_THROUGHPUT: Dict[str, str] = {
    'llama-3.1-8b-instruct': '/home/ferran/Documents/repositories/vLLMAdapterServingScaling/benchmarks/lora/definitive_results/finding_maximum/digital_twin_simplification/training/llama-3.1-8b-instruct/reg/rf/',
    'qwen-2.5-7b-instruct': '/home/ferran/Documents/repositories/vLLMAdapterServingScaling/benchmarks/lora/definitive_results/finding_maximum/digital_twin_simplification/training/qwen-2.5-7b-instruct/reg/knn/',
}
ML_MODEL_PATH_STARVATION: Dict[str, str] = {
    'llama-3.1-8b-instruct': '/home/ferran/Documents/repositories/vLLMAdapterServingScaling/benchmarks/lora/definitive_results/finding_maximum/digital_twin_simplification/training/llama-3.1-8b-instruct/class/rf/',
    'qwen-2.5-7b-instruct': '/home/ferran/Documents/repositories/vLLMAdapterServingScaling/benchmarks/lora/definitive_results/finding_maximum/digital_twin_simplification/training/qwen-2.5-7b-instruct/class/rf/',
}

def assign_ranks_to_adapters(
        adapters: List[str],
        values_to_use: List[int]
) -> List[int]:
    assert len(adapters) >= len(values_to_use)
    random.shuffle(values_to_use)

    # distribute
    adapters_ranks: List[int] = []
    index = 0
    while len(adapters_ranks) < len(adapters):
        adapters_ranks.append(values_to_use[index])
        index += 1
        if index >= len(values_to_use):
            index = 0

    # shuffle
    random.shuffle(adapters_ranks)

    return adapters_ranks


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


# high rates -> [2.4, 1.2, 0.6, 0.3, 0.15]
# low rates -> [0.075, 0.0375, 0.01875, 0.009375, 0.0046875]
# mixed rates -> [2.4, 1.2, 0.6, 0.3, 0.15, 0.075, 0.0375, 0.01875, 0.009375, 0.0046875]
# mixed rates new -> [2.4, 0.0046875, 1.2, 0.009375, 0.6, 0.01875]
# mixed rates new 2 -> [0.6, 0.3, 0.15, 0.075, 0.0375]

# high sizes -> [32]
# low sizes -> [8]
# mixed sizes -> [8, 16, 32]


ADAPTERS = 128
SERVERS = 1
MODEL = 'qwen-2.5-7b-instruct'
# METHOD = 'proposal-starvation-2'
# METHOD = 'baseline-4-with-proposal'
METHOD = 'baseline-2'

adapters = [f'a{index}' for index in range(0, ADAPTERS)]
servers = [f's{index}' for index in range(0, SERVERS)]
ranks_to_use = [32]
rates_to_use = [0.6, 0.3, 0.15, 0.075, 0.0375]
adapters_ranks = assign_ranks_to_adapters(adapters, ranks_to_use)
adapters_rates = assign_rates_to_adapters(adapters, rates_to_use)

# TEST PROPOSAL PLACEMENT
init_time: float = time.perf_counter()
placement_algorithm: PlacementAlgorithmInterface = get_subclass(METHOD)()
if 'proposal-starvation' in METHOD:
    servers_adapter_slots, adapters_servers = placement_algorithm.define_placement(
        model=MODEL,
        servers=servers,
        adapters=adapters,
        adapters_ranks=adapters_ranks,
        adapters_rates=adapters_rates,
        mean_input_length=221.6654239766082,
        mean_output_length=196.51532651072125,
        ml_model_path_throughput=ML_MODEL_PATH_THROUGHPUT,
        ml_model_path_starvation=ML_MODEL_PATH_STARVATION,
    )
elif 'with-proposal' in METHOD:
    servers_adapter_slots, adapters_servers = placement_algorithm.define_placement(
        model=MODEL,
        servers=servers,
        adapters=adapters,
        adapters_ranks=adapters_ranks,
        adapters_rates=adapters_rates,
        mean_input_length=221.6654239766082,
        mean_output_length=196.51532651072125,
        ml_model_path_starvation=ML_MODEL_PATH_STARVATION,
    )
else:
    servers_adapter_slots, adapters_servers = placement_algorithm.define_placement(
        model=MODEL,
        servers=servers,
        adapters=adapters,
        adapters_ranks=adapters_ranks,
        adapters_rates=adapters_rates,
        mean_input_length=221.6654239766082,
        mean_output_length=196.51532651072125,
    )
print('Elapsed time during placement estimation:', time.perf_counter() - init_time)
values, counts = np.unique(adapters_servers, return_counts=True)
print(f'Output placement. Adapter slots by server: {servers_adapter_slots}. Server by adapter: -> Values: {values} Counts: {counts}')

gpu_memory_availability = get_gpu_memory_availability('/home/ferran/Documents/repositories/vLLMAdapterServingScaling/benchmarks/lora/definitive_results/finding_maximum/check_available_gpu_memory/')
max_gpu_memory_availability = {}
for model, values_model in gpu_memory_availability.items():
    max_gpu_memory_availability[model] = {}
    for slots, values_slots in values_model.items():
        for rank in values_slots.keys():
            if rank not in max_gpu_memory_availability[model]:
                max_gpu_memory_availability[model][rank] = slots
            else:
                max_gpu_memory_availability[model][rank] = max(slots, max_gpu_memory_availability[model][rank])
del gpu_memory_availability
print(max_gpu_memory_availability)

for index_server, server in enumerate(servers):
    adapter_slots = servers_adapter_slots[index_server]
    served_adapters = 0
    rates = []
    sizes = []
    for index_adapter, adapter_server in enumerate(adapters_servers):
        if adapter_server == server:
            served_adapters += 1
            rates.append(adapters_rates[index_adapter])
            sizes.append(adapters_ranks[index_adapter])
    if served_adapters > 0:
        rates = np.asarray(rates)
        sizes = np.asarray(sizes)
        input_data = {
            'sum_rate': float(np.sum(rates)),
            'std_rate': float(np.std(rates)),
            'max_size': float(np.max(sizes)),
            'mean_size': float(np.mean(sizes)),
            'std_size': float(np.std(sizes)),
            'adapter_slots': adapter_slots,
            'served_adapters': served_adapters,
        }
        starvation = bool(predict_digital_twin(
            regression=False,
            model_path=ML_MODEL_PATH_STARVATION[MODEL],
            x_features=input_data,
            y_features_to_predict=['starvation']
        )['starvation'][0])

        assert input_data['max_size'] in max_gpu_memory_availability[MODEL]

        memory_error = False if adapter_slots <= max_gpu_memory_availability[MODEL][input_data['max_size']] else True

        print(f'Server {server}. Starvation: {starvation}. Memory error: {memory_error}')
    else:
        print(f'Server {server}: Empty')


# TEST ML directly
'''input_data = {
    'sum_rate': float(np.sum(np.asarray(adapters_rates))),
    'std_rate': float(np.std(np.asarray(adapters_rates))),
    'max_size': float(np.max(np.asarray(adapters_ranks))),
    'mean_size': float(np.mean(np.asarray(adapters_ranks))),
    'std_size': float(np.std(np.asarray(adapters_ranks))),
    'adapter_slots': 8,
    'served_adapters': ADAPTERS,
}

output_data = predict_digital_twin(
    model_path=ML_MODEL_PATH['llama-3.1-8b-instruct'],
    x_features=input_data,
    y_features_to_predict=['total_throughput']
)

print(output_data)'''

'''# TEST B3 PLACEMENT
init_time: float = time.perf_counter()
placement_algorithm: PlacementAlgorithmInterface = get_subclass('baseline-3')()
servers_adapter_slots, adapters_servers = placement_algorithm.define_placement(
    model='llama-3.1-8b-instruct',
    servers=['s0', 's1', 's2', 's3'],
    adapters=adapters,
    adapters_ranks=adapters_ranks,
    adapters_rates=adapters_rates,
    mean_input_length=221.6654239766082,
    mean_output_length=196.51532651072125,
)
print('Elapsed time during placement estimation:', time.perf_counter() - init_time)
values, counts = np.unique(adapters_servers, return_counts=True)
print(f'Output placement. Adapter slots by server: {servers_adapter_slots}. Server by adapter: -> Values: {values} Counts: {counts}')
print(adapters_servers)'''