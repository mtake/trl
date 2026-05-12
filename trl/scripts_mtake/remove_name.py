import argparse
import json

def remove_name(x):
    if isinstance(x, dict):
        return { k: remove_name(v) for k, v in x.items() if not (k == "name" and v == "zrag_retriever") }
    elif isinstance(x, list):
        return [ remove_name(v) for v in x ]
    else:
        return x

def main():
    parser = argparse.ArgumentParser(description="Remove name in JSONL file")

    parser.add_argument("input_file", help="Path to the input file")
    parser.add_argument(
        "--output", "-o",
        default="output.jsonl",
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

    with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:

        for line in fin:
            obj = json.loads(line)
            obj = remove_name(obj)
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
