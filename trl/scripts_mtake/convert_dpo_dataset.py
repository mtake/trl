import argparse
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Convert DPO dataset to SFT dataset")

    parser.add_argument("input_file", help="Path to the input file")
    parser.add_argument(
        "--output", "-o",
        # default="output.jsonl",
        help="Path to the output file"
    )
    # parser.add_argument(
    #     "--verbose", "-v",
    #     action="store_true",
    #     help="Enable verbose output"
    # )

    args = parser.parse_args()

    # if args.verbose:
    #     print("Verbose mode enabled")

    input_file = args.input_file
    output_file = args.output


    # 1. Load your DPO dataset
    if input_file.endswith((".json",".jsonl")):
        if output_file is None:
            output_file = f"{input_file[:input_file.rfind(".")]}-sft.jsonl"
        dpo_dataset = load_dataset("json", data_files=input_file, split="train")
    else:
        if output_file is None:
            output_file = f"{input_file[input_file.rfind("/")+1:]}-sft.jsonl"
        dpo_dataset = load_dataset(input_file, split="train")

    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")


    # 2. Map function to extract only the 'chosen' response
    # def convert_to_sft(example):
    #     return {
    #         "messages": [
    #             {"role": "user", "content": example["prompt"]},
    #             {"role": "assistant", "content": example["chosen"]}
    #         ]
    #     }
    def convert_to_sft(example):
        return {
            "messages": example["chosen"]
        }


    # 3. Apply the transformation
    sft_dataset = dpo_dataset.map(convert_to_sft, remove_columns=dpo_dataset.column_names)
    sft_dataset.to_json(output_file)

    # sft_dataset is now ready for SFTTrainer


if __name__ == "__main__":
    main()
