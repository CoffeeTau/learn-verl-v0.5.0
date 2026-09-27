# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Preprocess locally downloaded GSM8K JSONL files for verl training."""

import argparse
import os
import re

import datasets


DATA_SOURCE = "openai/gsm8k"
EXPECTED_COLUMNS = {"data_source", "prompt", "ability", "reward_model", "extra_info"}
EXPECTED_ROWS = {"train": 7473, "test": 1319}
INSTRUCTION = 'Let\'s think step by step and output the final answer after "####".'


def extract_solution(answer: str) -> str:
    match = re.search(r"#### (\-?[0-9\.\,]+)", answer)
    if match is None:
        raise ValueError(f"Unable to extract a GSM8K answer from: {answer!r}")
    return match.group(1).replace(",", "")


def make_map_fn(split: str):
    def process_fn(example, index):
        question = example["question"]
        answer = example["answer"]

        return {
            # Keep the original data-source name because verl uses it to select
            # the built-in GSM8K reward function.
            "data_source": DATA_SOURCE,
            "prompt": [{"role": "user", "content": f"{question} {INSTRUCTION}"}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": extract_solution(answer)},
            "extra_info": {
                "split": split,
                "index": index,
                "answer": answer,
                "question": question,
            },
        }

    return process_fn


def validate_split(split: str, dataset: datasets.Dataset) -> None:
    actual_columns = set(dataset.column_names)
    if actual_columns != EXPECTED_COLUMNS:
        raise ValueError(
            f"Unexpected columns for {split}: expected {sorted(EXPECTED_COLUMNS)}, got {sorted(actual_columns)}"
        )

    expected_rows = EXPECTED_ROWS[split]
    if len(dataset) != expected_rows:
        raise ValueError(f"Unexpected row count for {split}: expected {expected_rows}, got {len(dataset)}")

    first = dataset[0]
    if first["data_source"] != DATA_SOURCE:
        raise ValueError(f"Unexpected data_source for {split}: {first['data_source']!r}")
    if not first["reward_model"]["ground_truth"]:
        raise ValueError(f"Empty ground truth in the first {split} example")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw_dir",
        default="data/gsm8k-raw",
        help="Directory containing train.jsonl and test.jsonl",
    )
    parser.add_argument(
        "--local_dir",
        default="data/gsm8k",
        help="Directory in which to write train.parquet and test.parquet",
    )
    args = parser.parse_args()

    raw_dir = os.path.abspath(os.path.expanduser(args.raw_dir))
    local_dir = os.path.abspath(os.path.expanduser(args.local_dir))
    data_files = {split: os.path.join(raw_dir, f"{split}.jsonl") for split in ("train", "test")}

    missing_files = [path for path in data_files.values() if not os.path.isfile(path)]
    if missing_files:
        missing = "\n".join(f"  - {path}" for path in missing_files)
        raise FileNotFoundError(f"Missing GSM8K source files:\n{missing}")

    print(f"Reading raw GSM8K data from: {raw_dir}")
    raw_datasets = datasets.load_dataset("json", data_files=data_files)
    os.makedirs(local_dir, exist_ok=True)

    for split in ("train", "test"):
        source = raw_datasets[split]
        processed = source.map(
            make_map_fn(split),
            with_indices=True,
            remove_columns=source.column_names,
            desc=f"Processing {split}",
        )
        validate_split(split, processed)

        output_path = os.path.join(local_dir, f"{split}.parquet")
        processed.to_parquet(output_path)
        print(f"Wrote {len(processed)} {split} examples to: {output_path}")

    print("GSM8K preprocessing and validation completed successfully.")


if __name__ == "__main__":
    main()
