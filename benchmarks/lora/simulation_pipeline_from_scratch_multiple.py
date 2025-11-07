import argparse
import os
import json
from benchmarks.lora.simulation_pipeline_from_scratch import simulation_pipeline_from_scratch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--paths',
        type=str,
        required=True
    )
    args = parser.parse_args()

    paths = [item for item in args.paths.split(' ')]
    for path in paths:
        print('------------>Running', path)

        # load arguments
        with open(os.path.join(path, 'arguments.json'), 'r') as file:
            arguments = json.load(file)
        print('With arguments', json.dumps(arguments, indent=4))

        # run
        simulation_output = simulation_pipeline_from_scratch(
            **arguments
        )

        if simulation_output is not None:
            with open(os.path.join(path, 'simulation_results.json'), 'w') as file:
                json.dump(simulation_output, file, indent=4)
