import os
import json
import glob
from AgentVictimLLM_Defense import attacker
from agentReward import myreward


def process_json_files(folder_path, victim_model="o1-mini"):
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist")
        return

    json_files = glob.glob(os.path.join(folder_path, "*.jsonl"))

    if not json_files:
        print(f"Warning: No JSON files found in '{folder_path}'")
        return

    print(f"Found {len(json_files)} JSON files")

    for json_file in json_files:

        attack_succeed = 0
        file_name = os.path.basename(json_file)

        with open(json_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())

                    if "turns" in data and isinstance(data["turns"], list) and data["turns"]:
                        for turn in data["turns"]:
                            response = attacker.inference(turn, victim_model)
                            _, jud = myreward.judge(turn, response)
                            print("-" * 50)

                            print(
                                f"Judge score (1 means success, 0 means failure, -1 means exception): {jud}")
                            if int(jud) == 1:
                                attack_succeed += 1
                            print(
                                f"Target: {turn}, Output: {response}, Current successful attacks: {attack_succeed}")

                except json.JSONDecodeError:
                    print(f"Warning: Line {line_num} is not valid JSONL format, skipped")
                except Exception as e:
                    print(f"Error processing line {line_num}: {str(e)}")
        print("--Output Results----" * 4)
        print(
            f"\nProcessed file: {file_name}, Victim model: {victim_model}, Attack success rate: {round(attack_succeed/440, 2)}")


if __name__ == "__main__":
    victim_model = "o1-mini"
    folder_path = "sorry_bench_for_reasoning_models"

    process_json_files(folder_path, victim_model)
