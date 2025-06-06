from wxpusher import WxPusher
from serverchan_sdk import sc_send
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from grpoV13 import send_message_to_notify
from agentReward import myreward
from grpoV13 import lock_json_item
import os
import json
import glob
import datetime
from agentVictimLLM import attacker


def find_successful_prompts(target_bad_q, base_folder="RESULTS_harmbench"):
    base_path = os.path.abspath(base_folder)
    print(f"Search path: {base_path}")

    json_files = []

    pattern1 = os.path.join(base_path, "**", "*_SCANING.json")
    files1 = glob.glob(pattern1, recursive=True)
    json_files.extend(files1)

    pattern2 = os.path.join(base_path, "**", "*_PAP_*.json")
    files2 = glob.glob(pattern2, recursive=True)
    json_files.extend([f for f in files2 if f not in json_files])

    if not json_files:
        for root, dirs, files in os.walk(base_path):
            for file in files:
                if file.endswith(".json") and ("victimModel" in file or "PAP" in file):
                    full_path = os.path.join(root, file)
                    if full_path not in json_files:
                        json_files.append(full_path)

    if not json_files:
        print(f"No matching JSON files found in {base_path}")
        return []

    successful_prompts = []

    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for item in data:
                if item.get('bad_q') == target_bad_q and item.get('stage') != "-1":
                    adv_prompt = item.get('adv_prompt')
                    if adv_prompt:
                        successful_prompts.append(adv_prompt)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    return successful_prompts


filename_victim_models_attackResults = "./RESULTS_harmbench/victim_model_claude-2_qwen0.5b/GRPO_victimModel_claude-2_PAP_SCANING.json"


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)

    response = WxPusher.send_message('This is a test message'+str(datetime.datetime.now()),
                                     uids=['UID_lJP5Xvj6dTAMpSfIY2zcXi5Kmep3'],
                                     token='AT_P1hbf0HMRLCfNWgttIa7kbIuNVzQoq1d')
    print(response)
