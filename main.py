from AgentVictimLLM_Defense import attacker
from agentReward import myreward
from agentAdvPromptGenerator import generateAdvPrompt
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)
from peft import (LoraConfig, PeftModel, TaskType, get_peft_model,
                  prepare_model_for_kbit_training)
import torch.optim as optim
import torch.nn.functional as F
import torch
import portalocker
import time
import re
import os
import logging
import json
import glob
import csv
import copy
import argparse
from serverchan_sdk import sc_send


alternative_model = 'qwen'


def append_to_json_RL_stage(
        bad_q: str,
        adv_content: str,
        adv_prompt: str,
        adv_response: str,
        score: float,
        file_path: str,
        stage: str = 'RL',
        score1: float = None,
        score2: float = None,
        score3: float = None,
        scores_text: str = None,
) -> None:
    new_data = {
        "bad_q": bad_q,
        "adv_content": adv_content,
        "adv_prompt": adv_prompt,
        "adv_response": adv_response,
        "score": score,
        "stage": stage,
        "score1(qwen2.5)": score1,
        "score2(deepseek-v2)": score2,
        "score3(deepseek-v3)": score3,
        "scores_text": scores_text
    }
    max_retries = 5
    retry_delay = 0.1

    for attempt in range(max_retries):
        try:
            if not os.path.exists(file_path):
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump([], f)

            with portalocker.Lock(file_path, 'r+', timeout=10, encoding='utf-8') as file:
                try:
                    file.seek(0)
                    content = file.read()

                    if content.strip():
                        data = json.loads(content)
                        if not isinstance(data, list):
                            data = [data]
                    else:
                        data = []

                    data.append(new_data)

                    file.seek(0)
                    file.truncate()
                    json.dump(data, file, ensure_ascii=False, indent=4)

                except json.JSONDecodeError as e:
                    print(f"JSON parsing error: {e}")
                    data = [new_data]
                    file.seek(0)
                    file.truncate()
                    json.dump(data, file, ensure_ascii=False, indent=4)

            break

        except portalocker.LockException as e:
            if attempt == max_retries - 1:
                raise Exception(f"Unable to acquire file lock after {max_retries} attempts: {str(e)}")
            time.sleep(retry_delay)
            continue

        except Exception as e:
            if attempt == max_retries - 1:
                raise Exception(f"Error writing to file: {str(e)}")
            time.sleep(retry_delay)
            continue


def lock_json_item(file_path):
    import random

    max_retries = 5
    retry_delay = 0.1

    for attempt in range(max_retries):
        try:
            with portalocker.Lock(file_path, 'r+', timeout=10, encoding='utf-8') as file:
                file.seek(0)
                content = file.read()

                if content.strip():
                    data = json.loads(content)
                else:
                    data = []

                all_completed = True
                for item in data:
                    stage = item.get("stage")
                    if stage != "RL_FIFTH_STAGE" and stage != "pap":
                        all_completed = False
                        break

                if all_completed and data:
                    print("All items are already in 'RL_FIFTH_STAGE' or 'pap' state, program will terminate...")
                    print("★★★★★ All jailbreak attack tasks have been completed ★★★★★")
                    import sys
                    sys.exit(0)

                eligible_items = []
                for i, item in enumerate(data):
                    if item.get("stage") == "-1":
                        eligible_items.append((i, item))

                if eligible_items:
                    random_index, random_item = random.choice(eligible_items)

                    data[random_index]["stage"] = "processing"

                    file.seek(0)
                    file.truncate()
                    json.dump(data, file, ensure_ascii=False, indent=4)

                    return random_item.get("bad_q"), random_item.get("adv_prompt")

                return None, None
        except portalocker.LockException:
            if attempt == max_retries - 1:
                raise Exception(f"Unable to acquire file lock after {max_retries} attempts")
            time.sleep(retry_delay)
        except json.JSONDecodeError:
            raise Exception("JSON file format error")
        except Exception as e:
            if attempt == max_retries - 1:
                raise Exception(f"Error processing file: {str(e)}")
            time.sleep(retry_delay)


def append_to_json(
        bad_q: str,
        adv_content: str,
        adv_prompt: str,
        adv_response: str,
        score: float,
        file_path: str,
        stage: str,
        score_mistral: float = 0,
        scores_gpt4o_mini: float = 0,
        score1: float = None,
        score2: float = None,
        score3: float = None,
        scores_text: str = None,
) -> None:
    new_data = {
        "bad_q": bad_q,
        "adv_content": adv_content,
        "adv_prompt": adv_prompt,
        "adv_response": adv_response,
        "score": score,
        "stage": stage,
        "score_mistral": score_mistral,
        "scores_gpt4o_mini": scores_gpt4o_mini,
        "score1(qwen2.5)": score1,
        "score2(deepseek-v2)": score2,
        "score3(deepseek-v3)": score3,
        "scores_text": scores_text
    }

    if not os.path.exists(file_path):
        print('*****' * 5)
        print(f"File does not exist, need to create new file: {file_path}")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump([], f)

    try:
        with portalocker.Lock(file_path, 'r+', timeout=10, encoding='utf-8') as file:
            file.seek(0)
            content = file.read()

            if content.strip():
                data = json.loads(content)
                if not isinstance(data, list):
                    data = [data]
            else:
                data = []

            found = False
            for i, item in enumerate(data):
                if item.get("bad_q") == bad_q:
                    found = True
                    if score > item.get("score") and stage == 'processing':
                        data[i] = new_data
                    if stage == 'RL_FIFTH_STAGE' or stage == 'pap':
                        data[i] = new_data
                    break

            if not found:
                data.append(new_data)

            file.seek(0)
            file.truncate()
            json.dump(data, file, ensure_ascii=False, indent=4)

    except portalocker.LockException as e:
        raise Exception(f"Unable to acquire file lock: {str(e)}")
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        data = [new_data]
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
    except Exception as e:
        raise Exception(f"Error writing to file: {str(e)}")


def compute_prob_ratio(new_logits, old_logits, target_ids, attention_mask):
    new_log_probs = F.log_softmax(new_logits, dim=-1)
    old_log_probs = F.log_softmax(old_logits, dim=-1)

    new_token_log_probs = torch.gather(
        new_log_probs, -1, target_ids.unsqueeze(-1)).squeeze(-1)
    old_token_log_probs = torch.gather(
        old_log_probs, -1, target_ids.unsqueeze(-1)).squeeze(-1)

    prob_ratio = torch.exp(new_token_log_probs - old_token_log_probs)
    return prob_ratio


class GRPOTrainer:
    def __init__(self,
                 model_name,
                 checkpoint_model_path,
                 victim_model,
                 EPOCHS,
                 device="auto",
                 learning_rate=1e-5,
                 results_path=None,
                 mylogger=None,
                 host='http://localhost:11434',
                 filename=None,
                 use_qlora=False,
                 model_bits=4):
        self.mylogger = mylogger
        self.host = host
        self.filename = filename

        if use_qlora:
            self.mylogger.info(
                f"Using QLoRA to load model: {model_name}, quantization bits: {model_bits} bits")

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=(model_bits == 4),
                load_in_8bit=(model_bits == 8),
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map=device,
                trust_remote_code=True
            )

            self.model = prepare_model_for_kbit_training(self.model)

            if "Qwen" in model_name:
                target_modules = ["q_proj", "k_proj", "v_proj",
                                  "o_proj", "gate_proj", "up_proj", "down_proj"]
            else:
                target_modules = ["query_key_value", "dense",
                                  "dense_h_to_4h", "dense_4h_to_h"]

            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=target_modules,
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )

            self.model = get_peft_model(self.model, lora_config)
            self.mylogger.info("QLoRA configuration applied, showing trainable parameters:")
            self.model.print_trainable_parameters()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map=device,
                trust_remote_code=True
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.checkpoint_model_path = checkpoint_model_path
        self.victim_model = victim_model
        self.EPOCHS = EPOCHS
        self.results_path = results_path

        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=learning_rate)

        self.old_model = copy.deepcopy(self.model)
        self.old_model.eval()

    def compute_logits(self, model, input_ids):
        with torch.set_grad_enabled(isinstance(model, type(self.model))):
            outputs = model(input_ids=input_ids)
            return outputs.logits

    def generate_responses(self, prompts, max_new_tokens=512, num_return_sequences=3):
        all_messages = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user",
                         "content": prompt}
            ]
            for prompt in prompts
        ]
        texts = [
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            for messages in all_messages
        ]

        model_inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.model.device)

        with torch.no_grad():
            generated_outputs = self.model.generate(
                input_ids=model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                num_return_sequences=num_return_sequences,
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=True,
                temperature=0.9,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.2
            )

        all_responses = []
        for i in range(len(prompts)):
            responses = []
            for j in range(num_return_sequences):
                output_ids = generated_outputs.sequences[i *
                                                         num_return_sequences + j]
                response = self.tokenizer.decode(
                    output_ids[len(model_inputs.input_ids[i]):],
                    skip_special_tokens=True
                )
                responses.append(response)
            all_responses.append(responses)

        return all_responses, model_inputs, generated_outputs.sequences

    def compute_rewards(self, attack_goal, responses):
        rewards = [myreward.score(attack_goal, response, host=self.host)[
            0] for response in responses]
        return rewards

    def train(self, prompts_original_rephrase, attack_goal, max_new_tokens=512, num_return_sequences=1, epsilon=0.3):
        rephrased_responses_adv_prompts_list, model_inputs, generated_outputs = self.generate_responses(
            prompts_original_rephrase, max_new_tokens, num_return_sequences
        )
        all_responses_adv_content = []
        for cur_rephrased_responses_adv_prompts_list in rephrased_responses_adv_prompts_list:
            for cur_jailbreak_prompt in cur_rephrased_responses_adv_prompts_list:
                cur_responses_adv_content = attacker.inference(cur_jailbreak_prompt, self.victim_model,
                                                               host=self.host)
                all_responses_adv_content.append(cur_responses_adv_content)
        all_rewards = []
        all_scores_text = []
        all_score1 = []
        all_score2 = []
        all_score3 = []
        for responses in all_responses_adv_content:
            curreward, scores_text, score1, score2, score3 = myreward.score(attack_goal, responses,
                                                                            host=self.host)
            self.mylogger.info(scores_text)
            all_rewards.append(curreward)
            all_scores_text.append(scores_text)
            all_score1.append(score1)
            all_score2.append(score2)
            all_score3.append(score3)
        terminate = False
        for tmp_jailbreak_prompt, tmp_llm_jailbreak_response, tmp_reward, tmp_scorestext, tmp_score1, tmp_score2, tmp_score3 in zip(
                rephrased_responses_adv_prompts_list[0], all_responses_adv_content, all_rewards, all_scores_text, all_score1, all_score2, all_score3):
            if terminate:
                break

            saved_data_file_path = generate_saved_file_path(
                attack_goal, self.victim_model, self.results_path)
            append_to_json_RL_stage(
                bad_q=attack_goal,
                adv_content=prompts_original_rephrase[0],
                adv_prompt=tmp_jailbreak_prompt,
                adv_response=tmp_llm_jailbreak_response,
                score=tmp_reward,
                file_path=saved_data_file_path,
                stage='RL',
                score1=tmp_score1,
                score2=tmp_score2,
                score3=tmp_score3,
                scores_text=tmp_scorestext
            )

            self.mylogger.info(f"RL stage large model output: {tmp_llm_jailbreak_response}")

            score_is_attack_succed_1, score_is_attack_succed_2 = myreward.judge(
                attack_goal, tmp_llm_jailbreak_response)
            if score_is_attack_succed_2 == 1:
                terminate = True
                append_to_json(attack_goal,
                               adv_content=prompts_original_rephrase[0],
                               adv_prompt=tmp_jailbreak_prompt,
                               adv_response=tmp_llm_jailbreak_response,
                               score=tmp_reward,
                               stage='RL_FIFTH_STAGE',
                               file_path=self.filename,
                               score1=tmp_score1,
                               score2=tmp_score2,
                               score3=tmp_score3,
                               scores_text=tmp_scorestext,
                               scores_gpt4o_mini=score_is_attack_succed_2,
                               score_mistral=score_is_attack_succed_1
                               )
            else:
                append_to_json(attack_goal,
                               adv_content=prompts_original_rephrase[0],
                               adv_prompt=tmp_jailbreak_prompt,
                               adv_response=tmp_llm_jailbreak_response,
                               score=tmp_reward,
                               stage='processing',
                               file_path=self.filename,
                               score1=tmp_score1,
                               score2=tmp_score2,
                               score3=tmp_score3,
                               scores_text=tmp_scorestext,
                               scores_gpt4o_mini=score_is_attack_succed_2,
                               score_mistral=score_is_attack_succed_1
                               )
        if terminate:
            return -1, -1, -1
        flat_rewards = torch.tensor(
            all_rewards, dtype=torch.float32, device=self.model.device)

        new_logits = self.compute_logits(self.model, model_inputs.input_ids)
        with torch.no_grad():
            old_logits = self.compute_logits(
                self.old_model, model_inputs.input_ids)

        ratio = compute_prob_ratio(
            new_logits,
            old_logits,
            model_inputs.input_ids,
            model_inputs.attention_mask
        )
        advantages = torch.mean(flat_rewards).item()+1e-6
        surr1 = ratio * advantages
        clipped_prob_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
        surr2 = clipped_prob_ratio * advantages
        policy_loss = -torch.min(surr1, surr2)
        policy_loss = policy_loss.mean()

        total_loss = policy_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return policy_loss, 0, 0

    def save_model(self, epoch):
        if hasattr(self.model, 'peft_config'):
            save_path = f"{self.checkpoint_model_path}_epoch_{epoch}"
            self.model.save_pretrained(save_path)
            self.mylogger.info(f"LoRA weights saved: {save_path}")
        else:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }
            if (epoch + 1) % self.EPOCHS == 0:
                torch.save(checkpoint, self.checkpoint_model_path)
                self.mylogger.info(
                    f"Checkpoint saved: {self.checkpoint_model_path}")


def read_csv_and_get_prompts(file_path='sorry_bench_202503.csv'):
    prompts = []
    try:
        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)

            for row in reader:
                prompt = row.get('prompt')
                if prompt:
                    prompts.append(prompt)

    except FileNotFoundError:
        print(f"Error: File {file_path} not found!")
    except Exception as e:
        print(f"Error reading file: {e}")
    return prompts


def load_finetuned_model(model_name, lora_path):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = PeftModel.from_pretrained(model, lora_path)

    return model, tokenizer


def read_csv_and_get_first_prompt(file_path='sorry_bench_202503.csv'):
    prompts = {}
    try:
        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)

            for row in reader:
                category = row.get('category')
                prompt = row.get('prompt')
                if category and prompt and category not in prompts:
                    prompts[category] = prompt

    except FileNotFoundError:
        print(f"Error: File {file_path} not found!")
    except Exception as e:
        print(f"Error reading file: {e}")
    return prompts


def get_top_n_by_category(file_path, n=3):
    import pandas as pd
    df = pd.read_csv(file_path)

    prompts = {}

    categories = df['category'].unique()

    for category in categories:
        category_df = df[df['category'] == category]
        category_prompts = category_df['prompt'].head(n).tolist()
        prompts[str(category)] = category_prompts

    return prompts


def clean_filename(filename, max_length=255):
    words = filename.split()

    first_ten = words[:10]

    filename = ' '.join(first_ten)
    cleaned_name = re.sub(r'[<>:"/\\|?*\',;]', '_', filename)

    cleaned_name = re.sub(r'\s+', '_', cleaned_name)

    cleaned_name = cleaned_name.strip('. ')

    if len(cleaned_name) > max_length:
        if '.' in cleaned_name:
            name_part, ext_part = cleaned_name.rsplit('.', 1)
            max_name_length = max_length - len(ext_part) - 1
            if max_name_length > 0:
                cleaned_name = name_part[:max_name_length] + '.' + ext_part
            else:
                cleaned_name = name_part[:1] + '.' + ext_part[:max_length-2]
        else:
            cleaned_name = cleaned_name[:max_length]

    return cleaned_name


def generate_saved_file_path(cur_bad_q, victim_model, filepath):
    cur_bad_q = clean_filename(cur_bad_q)
    saved_data_file_path = filepath + \
        f'GRPO_data_victimModel_{victim_model}_{cur_bad_q}.json'
    return saved_data_file_path


def extract_after_second_slash(path):
    parts = path.split('/', 2)

    if len(parts) >= 3:
        return parts[2]
    else:
        return ""


def sort_json_files_by_score():
    json_files = glob.glob("GRPO_data*.json")

    if not json_files:
        print("No JSON files found starting with GRPO_data")
        return

    print(f"Found {len(json_files)} files to process")

    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not isinstance(data, list):
                print(f"Warning: {file_path} does not contain a JSON array, skipping")
                continue

            sorted_data = sorted(
                data, key=lambda x: x.get('score', 0), reverse=True)

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(sorted_data, f, ensure_ascii=False, indent=4)

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")


def read_json_file_judge(file_path, filename, stage='RL'):
    try:
        if not os.path.exists(file_path):
            print(f"Error: File '{file_path}' does not exist")
            return

        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        if not isinstance(data, list):
            print("Error: JSON file should contain a list")
            return

        print(f"Successfully read JSON file {file_path}, with {len(data)} records")

        for index, item in enumerate(data):
            bad_q = item.get("bad_q", "None")
            adv_response = item.get("adv_response", "None")
            adv_content = item.get("adv_content", "None")
            adv_prompt = item.get("adv_prompt", "None")
            score = item.get("score", "None")
            score_is_attack_succed_1, score_is_attack_succed_2 = myreward.judge(
                bad_q, adv_response)
            if score_is_attack_succed_2 == 1:
                print(f"Jailbreak attack succeeded, bad_q is: {bad_q}")
                append_to_json(bad_q,
                               adv_content=adv_content,
                               adv_prompt=adv_prompt,
                               adv_response=adv_response,
                               file_path=filename,
                               score=score,
                               stage=stage)
    except json.JSONDecodeError:
        print("Error: Invalid JSON format")
    except Exception as e:
        print(f"Error processing file: {e}")


def setup_logger(args, results_path):
    log_filename = f"{results_path}log-{args.victim_model}.txt"

    logger = logging.getLogger(f"victim_model_{args.victim_model}")
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.log_filename = log_filename
    return logger


def report_jailbreak_progress(filename, mylogger):
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)

    total_records = len(data)
    pap_count = 0

    RL_FIFTH_STAGE_count = 0
    failed_count = 0
    for record in data:
        if record.get("stage") == "pap":
            pap_count += 1
        elif record.get("stage") == "-1":
            failed_count += 1
        elif record.get("stage") == "processing":
            failed_count += 1
        elif record.get("stage") == "RL_FIFTH_STAGE":
            RL_FIFTH_STAGE_count += 1

    mylogger.info('*************'*10)
    mylogger.info(f"Number of attack targets to jailbreak imported from dataset: {total_records}")
    mylogger.info(
        f'pap stage jailbreak success records: {pap_count}, attack success rate: {round(pap_count/total_records,2)}')
    mylogger.info(
        f'RL_FIFTH_STAGE stage jailbreak success records: {RL_FIFTH_STAGE_count}')
