import sys
import json
import argparse
from collections import Counter


def count_words(file_path):
    word_counts = Counter()
    
    try:
        with open(file_path, 'r') as file:
            for line in file:
                words = line.strip().lower().split()
                word_counts.update(words)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except IOError:
        print(f"Error: Unable to read file '{file_path}'.")
        sys.exit(1)
    
    return dict(word_counts)


def main():
    parser = argparse.ArgumentParser(description="Count word frequency in a file.")
    parser.add_argument("input_file", help="Path to the input file")
    parser.add_argument("-o", "--output", help="Path to the output file (optional)")
    args = parser.parse_args()

    frequency_dict = count_words(args.input_file)
    
    if args.output:
        sorted_frequency = sorted(frequency_dict.items(), key=lambda x: x[1], reverse=True)
        with open(args.output, 'w') as outfile:
            json.dump(dict(sorted_frequency), outfile, indent=4)
        print(f"Word frequency written to {args.output}")
    else:
        print("Word frequency:")
        for word, count in frequency_dict.items():
            print(f"{word}: {count}")

if __name__ == "__main__":
    main()
