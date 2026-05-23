import argparse
import json
import re

def main():
    parser = argparse.ArgumentParser(description="Convert sft dataset to retriever data")

    parser.add_argument("input_file", help="Path to the input file")
    parser.add_argument(
        "--output", "-o",
        default="output.jsonl",
        help="Path to the output file"
    )
    parser.add_argument(
        "--parse", "-p",
        action="store_true",
        help="Parse raw documents to a list of documents. Otherwise, keep raw documents."
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
    parse = args.parse

    # rtrvr = {}  # NOTE as single dictionary

    with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:

        for line in fin:
            obj = json.loads(line)
            messages = obj["messages"]
            query = None
            raw_documents = None
            for message in reversed(messages):
                role = message["role"]
                if not raw_documents:
                    if role == "tool":
                        raw_documents = message["content"]
                    continue
                if role == "user":
                    query = message["content"]
                    break

            if query and raw_documents:
                # print(f"XXX raw_documents = XXX {raw_documents} XXX")

                if parse:
                    documents = []
                    # Parse raw documents to a list of documents
                    document_split = re.split(r'\[Rank \d+\] ', raw_documents)
                    if not document_split[0].strip():
                        document_split = document_split[1:]
                    # print(f"XXX document_split = XXX {document_split} XXX")
                    for i, document in enumerate(document_split):
                        start_title = document.find("Title: ")
                        start_url = document.find("URL: ")
                        start_score = document.find("Score: ")
                        start_content = document.find("Content: ")
                        title = document[start_title+len("Title: "):start_url].strip()
                        url = document[start_url+len("URL: "):start_score].strip()
                        score = document[start_score+len("Score: "):start_content].strip()
                        content = document[start_content+len("Content: "):]
                        if content.endswith("\n\n"):
                            content = content[:-len("\n\n")]
                        # if content.endswith("\n#\n"):
                        #     content = content[:-len("\n#\n")]
                        document_dict = {
                            "rank": i+1,
                            "title": title,
                            "url": url,
                            "score": float(score),
                            "content": content,
                        }
                        # print(f"XXX document_dict = XXX {document_dict} XXX")
                        documents.append(document_dict)
                else:
                    documents = raw_documents

                # rtrvr[query] = documents  # NOTE as single dictionary
                fout.write(json.dumps({"query": query, "documents": documents}, ensure_ascii=False) + "\n")  # NOTE as multiple dictionaries

        # json.dump(rtrvr, fout, ensure_ascii=False)  # NOTE as single dictionary

if __name__ == "__main__":
    main()
