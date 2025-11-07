import argparse
import os

from huggingface_hub import snapshot_download


def main(args):
    output = args.output
    token = args.token

    if args.download_llama_2_7b:
        snapshot_download(
            repo_id='meta-llama/Llama-2-7b-hf',
            local_dir=os.path.join(output, 'llama/llama-2-7b'),
            local_dir_use_symlinks=False,
            token=token
        )

        snapshot_download(
            repo_id='yard1/llama-2-7b-sql-lora-test',
            local_dir=os.path.join(output, 'llama/llama-2-7b/lora/yard1_sql-lora-test'),
            local_dir_use_symlinks=False
        )

    if args.download_llama_2_13b:
        snapshot_download(
            repo_id='meta-llama/Llama-2-13b-hf',
            local_dir=os.path.join(output, 'llama/llama-2-13b'),
            local_dir_use_symlinks=False,
            token=token
        )

    if args.download_llama_3_1_8b_instruct:
        snapshot_download(
            repo_id='meta-llama/Llama-3.1-8B-Instruct',
            local_dir=os.path.join(output, 'llama/llama-3.1-8b-instruct'),
            local_dir_use_symlinks=False,
            token=token
        )

        snapshot_download(
            repo_id='Wengwengwhale/llama-3.1-8B-Instruct-Finance-lora-adapter',
            local_dir=os.path.join(output, 'llama/llama-3.1-8b-instruct/lora/lora-finance'),
            local_dir_use_symlinks=False
        )

    if args.download_mistral:
        snapshot_download(
            repo_id='mistralai/Mistral-7B-v0.1',
            local_dir=os.path.join(output, 'mistral/mistral-7b'),
            local_dir_use_symlinks=False,
            token=token
        )

    if args.download_mixtral:
        snapshot_download(
            repo_id='mistralai/Mixtral-8x7B-v0.1',
            local_dir=os.path.join(output, 'mixtral/mixtral-8x7b'),
            local_dir_use_symlinks=False,
            token=token
        )

    if args.download_qwen_2_5_7b:
        snapshot_download(
            repo_id='Qwen/Qwen2.5-7B',
            local_dir=os.path.join(output, 'qwen/qwen-2.5-7b'),
            local_dir_use_symlinks=False,
            token=token
        )

    if args.download_qwen_2_5_7b_instruct:
        snapshot_download(
            repo_id='Qwen/Qwen2.5-7B-Instruct',
            local_dir=os.path.join(output, 'qwen/qwen-2.5-7b-instruct'),
            local_dir_use_symlinks=False,
            token=token
        )

        snapshot_download(
            repo_id='zjudai/flowertune-medical-lora-qwen2.5-7b-instruct',
            local_dir=os.path.join(output, 'qwen/qwen-2.5-7b-instruct/lora/lora-medical'),
            local_dir_use_symlinks=False
        )

    if args.download_gpt_2_xl:
        snapshot_download(
            repo_id='openai-community/gpt2-xl',
            local_dir=os.path.join(output, 'gpt/gpt-2-xl'),
            local_dir_use_symlinks=False,
            token=token
        )

        snapshot_download(
            repo_id='MHGanainy/gpt2-xl-lora-multi',
            local_dir=os.path.join(output, 'gpt/gpt-2-xl/lora/lora-multi'),
            local_dir_use_symlinks=False
        )

    if args.download_bloomz_7b:
        snapshot_download(
            repo_id='bigscience/bloomz-7b1',
            local_dir=os.path.join(output, 'bloomz/bloomz-7b'),
            local_dir_use_symlinks=False,
            token=token
        )

        snapshot_download(
            repo_id='MBZUAI/bactrian-x-bloom-7b1-lora',
            local_dir=os.path.join(output, 'bloomz/bloomz-7b/lora/lora-bactrian'),
            local_dir_use_symlinks=False
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download required models from hugging face')
    parser.add_argument("--output", type=str, help="Path to output directory", required=True)
    parser.add_argument("--token", type=str, help="Auth token", required=True)
    parser.add_argument('--download-llama-2-7b', action='store_true', help='Download Llama-2-7b')
    parser.add_argument('--download-llama-2-13b', action='store_true', help='Download Llama-2-13b')
    parser.add_argument('--download-llama-3-1-8b-instruct', action='store_true', help='Download Llama-3.1-8B-Instruct')
    parser.add_argument('--download-mistral', action='store_true', help='Download mistral')
    parser.add_argument('--download-mixtral', action='store_true', help='Download mixtral')
    parser.add_argument('--download-qwen-2-5-7b', action='store_true', help='Download Qwen2.5-7b')
    parser.add_argument('--download-qwen-2-5-7b-instruct', action='store_true', help='Download Qwen2.5-7b-Instruct')
    parser.add_argument('--download-gpt-2-xl', action='store_true', help='Download GPT2-XL')
    parser.add_argument('--download-bloomz-7b', action='store_true', help='Download Bloomz-7b1')
    args = parser.parse_args()
    main(args)
