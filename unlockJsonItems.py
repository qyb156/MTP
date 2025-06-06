#!/usr/bin/env python

import os
import json
import time
import glob
import argparse
import portalocker


def unlock_all_processing_items(json_file_path):
    max_retries = 5
    retry_delay = 0.5
    unlocked_count = 0

    print(f"Attempting to unlock file: {json_file_path}")

    for attempt in range(max_retries):
        try:
            with portalocker.Lock(json_file_path, 'r+', timeout=10, encoding='utf-8') as file:
                file.seek(0)
                content = file.read()

                if content.strip():
                    data = json.loads(content)
                else:
                    print(f"File {json_file_path} is empty or improperly formatted")
                    return 0

                modified = False
                for item in data:
                    if item.get("stage") == "processing":
                        item["stage"] = "-1"
                        unlocked_count += 1
                        modified = True

                if modified:
                    file.seek(0)
                    file.truncate()
                    json.dump(data, file, ensure_ascii=False, indent=4)
                    print(f"Successfully unlocked {unlocked_count} items in 'processing' state")
                else:
                    print(f"No items in 'processing' state found in the file")

                return unlocked_count

        except portalocker.LockException:
            print(
                f"Attempt {attempt+1}/{max_retries}: Unable to acquire file lock, waiting {retry_delay} seconds to retry...")
            time.sleep(retry_delay)
        except json.JSONDecodeError:
            print(f"Error: File {json_file_path} is not valid JSON format")
            return 0
        except Exception as e:
            print(f"Attempt {attempt+1}/{max_retries}: Error processing file: {str(e)}")
            if attempt == max_retries - 1:
                print(f"Maximum retry attempts reached, abandoning file {json_file_path}")
                return 0
            time.sleep(retry_delay)

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Unlock items with 'processing' status in JSON files")
    parser.add_argument('--result_dir', type=str, required=True, help="Results directory path")
    args = parser.parse_args()

    if not os.path.exists(args.result_dir):
        print(f"Error: Directory '{args.result_dir}' does not exist")
        return

    json_pattern = os.path.join(args.result_dir, "*_PAP_SCANING.json")
    json_files = glob.glob(json_pattern)

    if not json_files:
        print(f"No *_PAP_SCANING.json files found in directory '{args.result_dir}'")
        return

    total_unlocked = 0

    for json_file in json_files:
        print(f"\nProcessing file: {json_file}")
        unlocked = unlock_all_processing_items(json_file)
        total_unlocked += unlocked

    print(f"\nSummary: {total_unlocked} items unlocked in total")

    print("\nOperation complete, sleeping for 1 second...")
    time.sleep(1)


if __name__ == "__main__":
    main()
