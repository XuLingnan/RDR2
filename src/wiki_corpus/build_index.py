import os
import json
import argparse

def construct_file_index(jsonl_dir, output_index_file):
    title_index = {}

    for filename in os.listdir(jsonl_dir):
        # if filename.endswith(".jsonl"):
        if '.jsonl' in filename:
            filepath = os.path.join(jsonl_dir, filename)
            
            with open(filepath, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f):
                    data = json.loads(line.strip())
                    title = data["title"]
                    
                    if title not in title_index:
                        title_index[title] = {"filename": filename, "line_num": line_num}

    with open(output_index_file, "w", encoding="utf-8") as f:
        json.dump(title_index, f, ensure_ascii=False, indent=4)


def get_nodes_from_title(title, title_index, path_to_document_trees):
    if title in title_index:
        jsonl_filename = title_index[title]["filename"]
        line_num = title_index[title]["line_num"]
        jsonl_filepath = os.path.join(path_to_document_trees, jsonl_filename)
        
        with open(jsonl_filepath, "r", encoding="utf-8") as f:
            for current_line_num, line in enumerate(f):
                if current_line_num == line_num:
                    print(f"=====json.loads=====")
                    print("jsonl_filepath: " + jsonl_filepath)
                    return json.loads(line.strip())
    else:
        print("=====None=====")
        return None



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="File Processing")
    
    parser.add_argument('--task', required=True, choices=['build_index', 'get_nodes'], help="task to perform: 'build_index' or 'get_nodes'")

    # build_index
    parser.add_argument('--jsonl_dir', help='input jsonl files directory (dst dir)')
    parser.add_argument('--output_index_file', help='output index file in json format')

    # get_nodes
    parser.add_argument('--title', help='page title to search for nodes')
    parser.add_argument('--title_index_file', help='index file path')
    parser.add_argument('--path_to_document_trees', help='dst dir')

    args = parser.parse_args()

    if args.task == 'build_index':
        if not all([args.jsonl_dir, args.output_index_file]):
            print("error: 'build_index' task requires --jsonl_dir and --output_index_file parameters.")
            parser.print_help()
            exit(1)
        construct_file_index(args.jsonl_dir, args.output_index_file)
        print(f"index built: {args.output_index_file}")

    elif args.task == 'get_nodes':
        if not all([args.title, args.title_index_file, args.path_to_document_trees]):
            print("error: 'get_nodes' task requires --title, --title_index_file, and --path_to_document_trees parameters.")
            parser.print_help()
            exit(1)
            
        try:
            with open(args.title_index_file, "r", encoding="utf-8") as f:
                title_index = json.load(f)
        except FileNotFoundError:
            print(f"error: index file not found at '{args.title_index_file}'")
            exit(1)
        
        nodes = get_nodes_from_title(args.title, title_index, args.path_to_document_trees)
        
        if nodes:
            print(json.dumps(nodes, ensure_ascii=False, indent=4))
        else:
            print(f"No nodes found for title: {args.title}")