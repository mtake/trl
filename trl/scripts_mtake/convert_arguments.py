import argparse
import json

def convert_arguments(obj):
    """
    再帰的に走査し、
    function.arguments が JSON文字列なら dict/list に変換する
    """

    if isinstance(obj, dict):
        new_obj = {}

        for key, value in obj.items():
            # function.arguments の場合
            if key == "arguments" and isinstance(value, str):
                try:
                    parsed = json.loads(value)

                    # JSONとして解釈できた場合は再帰処理
                    new_obj[key] = convert_arguments(parsed)
                    continue

                except json.JSONDecodeError:
                    pass

            new_obj[key] = convert_arguments(value)

        return new_obj

    elif isinstance(obj, list):
        return [convert_arguments(x) for x in obj]

    else:
        return obj


def main():
    parser = argparse.ArgumentParser(description="Convert arguments from str to JSON")

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
            data = json.loads(line)
            converted = convert_arguments(data)
            fout.write(json.dumps(converted, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
