COEFFICIENTS = {
    'avg_batch_size': 0.004373,
    'avg_waiting_queue': 0.004615,
    'avg_waiting_queue_x_gpu/cpu_diff': -0.004391,
}

SCALING_PARAMS = {
    'avg_batch_size': {'min': 0.232143, 'max': 1081.371212},
    'avg_waiting_queue': {'min': 0.000000, 'max': 20309.088235},
    'avg_waiting_queue_x_gpu/cpu_diff': {'min': 0.000000, 'max': 16371.311594},
}

BASELINE_LATENCY = (0.00019961895773199724+0.0001932263105461435)/2  # Placeholder for baseline latency, if needed

def min_max_scale(value, scaling_param_dict):
    """Apply min-max scaling to a value."""
    min_val = scaling_param_dict['min']
    max_val = scaling_param_dict['max']
    if max_val == min_val:
        return 0.0
    return (value - min_val) / (max_val - min_val)


def scheduler_overhead_predictor(
    batch_size: int,
    waiting_requests: int,
    cpu_loras: int,
    gpu_loras: int
) -> float:
    """
    Predict scheduling time in milliseconds.
    
    Args:
        batch_size: Number of requests in batch
        waiting_requests: Number of waiting requests in queue
        cpu_loras: Number of LoRAs on CPU
        gpu_loras: Number of LoRAs on GPU
        loras_by_batch: Mean number of LoRAs per batch
    
    Returns:
        Predicted scheduling time in milliseconds
    """
    gpu_div_cpu_diff = gpu_loras/cpu_loras
    avg_waiting_queue_x_gpu_div_cpu_diff = waiting_requests * gpu_div_cpu_diff
    
    # Scale the features
    batch_size_scaled = min_max_scale(
        batch_size,
        SCALING_PARAMS['avg_batch_size']
    )
    
    waiting_queue_scaled = min_max_scale(
        waiting_requests,
        SCALING_PARAMS['avg_waiting_queue']
    )
    avg_waiting_queue_x_gpu_div_cpu_diff_scaled = min_max_scale(
        avg_waiting_queue_x_gpu_div_cpu_diff,
        SCALING_PARAMS['avg_waiting_queue_x_gpu/cpu_diff']
    )
    
    return max(COEFFICIENTS['avg_batch_size'] * batch_size_scaled + \
           COEFFICIENTS['avg_waiting_queue'] * waiting_queue_scaled + \
           COEFFICIENTS['avg_waiting_queue_x_gpu/cpu_diff'] * avg_waiting_queue_x_gpu_div_cpu_diff_scaled - BASELINE_LATENCY,0)*1000  # Convert to milliseconds
