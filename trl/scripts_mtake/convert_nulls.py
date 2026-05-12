import argparse
import json

def remove_nulls(x):
    if isinstance(x, dict):
        return { k: remove_nulls(v) for k, v in x.items() if v is not None }
    elif isinstance(x, list):
        return [ remove_nulls(v) for v in x ]
    else:
        return x

def replace_nulls(x):
    if isinstance(x, dict):
        return { k: replace_nulls(v) for k, v in x.items() }
    elif isinstance(x, list):
        return [ replace_nulls(v) for v in x ]
    elif x is None:
        return ""
    else:
        return x

def main():
    parser = argparse.ArgumentParser(description="Remove or replace nulls in JSONL file")

    parser.add_argument("input_file", help="Path to the input file")
    parser.add_argument(
        "--output", "-o",
        default="output.jsonl",
        help="Path to the output file"
    )
    parser.add_argument(
        "--empty", "-e",
        action="store_true",
        help="Replace nulls with empty strings rather than remove"
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
    empty = args.empty

    with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:

        for line in fin:
            obj = json.loads(line)
            if empty:
                obj = replace_nulls(obj)
            else:
                obj = remove_nulls(obj)
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
