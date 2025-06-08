This is the repository that contains the source code of MTP for the paper 'Multi-Turn Persuasion Attack for Jailbreaking Large Language Models'. 

## Project Structure

This project mainly contains the following Python files:

--'main.py' is the main program file. This Python file implements a reinforcement learning-based jailbreak attack method for large language models. The code orchestrates a multi-phase attack process that generates adversarial prompts to bypass safety measures in victim models. It utilizes a reward mechanism to evaluate attack effectiveness, maintains attack progress through JSON file tracking with cross-platform file locking, and implements a randomized task selection strategy to prevent processing bottlenecks. The system tests various victim models with different defense strategies, particularly focusing on defenses proposed in recent research papers about LLM safety. Due to ethical considerations regarding potential misuse, we are not sharing this training code publicly; interested developers should contact us directly for access after appropriate vetting.  
--'agentReward.py' is used to calculate the rewards for the large language model.
--'AgentVictimLLM_Defense.py' is used to obtain feedback from the victim model.

## Data Structure

This project mainly contains the following outcomes of our jailbreak attack experiments:

--'RESULTS_*******' All folders beginning with "RESULTS" contain the outcomes of our jailbreak attack experiments. To facilitate reproducibility for researchers, we have shared most of our experimental data. However, we have temporarily withheld results for commercial models like GPT-4 and Claude due to concerns that malicious actors might use this data to train adversarial attack models. Nevertheless, to demonstrate the effectiveness of our method, we have specifically shared the file "RESULTS_PAP_prompts_defense_constitutional_classifer," which contains results from attacking the latest commercial model Claude 3.5 Haiku using our approach with the Advbench dataset. This provides sufficient evidence of our method's capabilities while maintaining responsible disclosure practices.

--'persuasion_taxonomy.jsonl'  This repository contains a comprehensive taxonomy of persuasion techniques in JSONL format. The dataset catalogs 23 distinct persuasion strategies, each with a formal definition and practical example. Techniques range from evidence-based persuasion and logical appeals to emotional manipulation tactics like storytelling and negative emotion appeals. Each entry follows a consistent structure with three fields: the technique name, its definition, and an illustrative example (primarily using smoking cessation scenarios). This taxonomy serves as a valuable resource for researchers studying persuasion psychology, social influence mechanisms, and the development of communication analysis tools. The JSONL format makes it easily accessible for computational processing and integration into natural language understanding systems.

--'adv_bench_sub_gpt3.5.jsonl' The adv_bench_sub_gpt3.5.jsonl file contains a structured dataset designed to evaluate AI systems' weaknesses to jailbreaking attempts, featuring pairs of harmful requests ("bad_q") alongside their corresponding adversarial prompts ("ss_prompt") that use sophisticated techniques like priming, loyalty appeals, and storytelling to circumvent AI safety measures. Each entry includes the original AI response to the harmful request ("ori_output") and the potentially compromised response ("jb_output") generated when the same request is reframed using these adversarial techniques, providing valuable insights for researchers developing more robust AI safety mechanisms against increasingly nuanced manipulation attempts. ( referenced as ` the Advbench dataset’ in our paper)

--'harmbench.csv' The harmbench.csv file contains a comprehensive evaluation dataset designed to test AI systems' safety mechanisms against harmful requests, featuring 117+ categorized prompts that span a wide range of dangerous content including instructions for creating illegal substances, planning violent acts, spreading misinformation, promoting discrimination, facilitating cybercrime, and engaging in various forms of harm against individuals or groups. This structured benchmark appears intended for researchers and developers to systematically evaluate and improve AI safety guardrails by exposing systems to diverse harmful scenarios, allowing for the identification of vulnerabilities in content moderation capabilities and the development of more robust protective measures against potential misuse of AI technologies. ( referenced as ` the Harmbench dataset’ in our paper)

--'sorry_bench_202503.csv' The sorry_bench_202503.csv file contains a structured dataset designed to test AI systems' responses to inappropriate requests, featuring prompts categorized into  distinct types of harmful content, including insults, discrimination, threats, sexual harassment, defamation, self-harm promotion, violence, manipulation, and illegal activities. This benchmark appears to be intended to evaluate AI safety measures and the systems' ability to appropriately recognize, decline, and respond to potentially harmful user inputs. The name "SorryBench" suggests its purpose of assessing whether AI systems properly decline such problematic requests. ( referenced as ` the SORRY-Bench dataset’ in our paper)
## Usage

### 1. Clone the project
```bash
git clone <Project repository address>
cd <Project directory>
```

### 2. Install dependencies and API Preparation
```bash
pip install -r requirements.txt
```

#### API Preparation

To use this project, you need to obtain API keys for the relevant large language models. Here are the steps and considerations for different models:

##### OpenAI Models (e.g., GPT series)
1. **Sign up for an OpenAI account**:
   - Go to the OpenAI official website (https://openai.com/) and sign up for an account if you don't have one already.
2. **Generate an API key**:
   - Log in to your OpenAI account and navigate to the API key management section.
   - Create a new API key. Make sure to keep this key secure, as it grants access to your OpenAI account resources.
3. **Configure the API key in the project**:
   - In the `AgentVictimLLM_Defense.py` , replace the placeholder `"sk-xxx"` with your actual OpenAI API key for the relevant models (e.g., `"gpt-3.5-turbo"` and `"gpt-4-1106-preview"`).

##### Other Models (DeepSeek, Llama, etc.)
- Similar to OpenAI, you need to sign up for accounts on the respective platforms that provide these models.
- Obtain the API keys from their official websites or developer portals.


**Note**:
- Some models may require additional configuration, such as setting the base URL. Make sure to follow the official documentation of each model to correctly configure these settings.
- API keys are sensitive information. Do not share them publicly or commit them to version control systems. It is recommended to use environment variables or other secure methods to manage your API keys in a production environment.


### 3. Run the program

## Example
```bash
EPOCHS=6
DATA_SOURCE="PAP_prompts.csv"
MODEL_PATH="./Qwen/Qwen2.5-7B-Instruct"
FIFTH_STAGE="true"

# Define GPU start index and end index
START_GPU=1 # There are eight GPUs in total, starting from GPU 0. Now GPU 0 is reserved for the ollama program, which has been specified in the configuration file.
END_GPU=7   # Ends at GPU 7

# Extract data source file name (remove .csv suffix)
DATA_SOURCE_NAME=$(basename "${DATA_SOURCE}" .csv)

# Extract model name (extract the last directory name from the path)
MODEL_NAME=$(basename "${MODEL_PATH}")

# Construct experiment name using the corresponding variables
experiment_name="${VICTIM_MODEL}_${DATA_SOURCE_NAME}_${MODEL_NAME}"

# Add timestamp
timestamp=$(date +"%Y%m%d_%H%M%S")
log_dir="logs/${experiment_name}_${timestamp}"

# Create log directory
mkdir -p ${log_dir}

# Record experiment information
echo "Experiment start time: $(date)" > "${log_dir}/experiment_info.txt"
echo "Experiment name: ${experiment_name}" >> "${log_dir}/experiment_info.txt"
echo "Using model: ${VICTIM_MODEL}" >> "${log_dir}/experiment_info.txt"
echo "GPU range: ${START_GPU}-${END_GPU}" >> "${log_dir}/experiment_info.txt"

# Check if necessary result files exist
result_dir="./RESULTS_${DATA_SOURCE_NAME}/victim_model_${VICTIM_MODEL}_qwen0.5b"
# First, forcibly unlock the json file. It is possible that after the program is forcefully terminated, many entries in the json are locked with their status set to processing. We need to unlock these entries.
# This command will unlock all entries.
python unlockJsonItems.py --result_dir ${result_dir} > unlock.log 2>&1 & 
  
# Run function
run_experiment() {
    local gpu_id=$1
    # GPU ID now starts from 0 and is directly used as the CUDA device ID
    local cuda_device=${gpu_id}
    
    # Modify log file name format using corresponding variables
    local log_file="${log_dir}/output_gpu${gpu_id}_${VICTIM_MODEL}-${DATA_SOURCE_NAME}_${MODEL_NAME}-${gpu_id}.log"
    
    echo "Starting experiment for GPU ${gpu_id} (CUDA device: ${cuda_device})..."
    CUDA_VISIBLE_DEVICES=${cuda_device} nohup python grpoV13.py \
        --index ${gpu_id} \
        --victim_model "${VICTIM_MODEL}" \
        --host "${HOST}" \
        --EPOCHS ${EPOCHS} \
        --num_return_sequences ${num_return_sequences} \
        --is_open_fifth_stage "${FIFTH_STAGE}" \
        --data_source_file_path "${DATA_SOURCE}" \
        --model_name "${MODEL_PATH}" \
        > "${log_file}" 2>&1 &    
     
    echo "Experiment for GPU ${gpu_id} has started, log file: ${log_file}"
}

# Start GPU experiment in the specified range
for i in $(seq ${START_GPU} ${END_GPU}); do
    run_experiment $i
    sleep 0.1  # Add a short delay to avoid resource competition from simultaneous starts
done

echo "All experiments have started, log files are stored in: ${log_dir}"
echo "Use the following commands to check the running status of each GPU:"
for i in $(seq ${START_GPU} ${END_GPU}); do
    echo "GPU ${i}: tail -f ${log_dir}/output_gpu${i}_${VICTIM_MODEL}-${DATA_SOURCE_NAME}_${MODEL_NAME}-${i}.log"
done

# Display running processes
sleep 2
echo "All programs have started, view the processes running on each GPU:"
nvidia-smi
sleep 2 # Add a short delay to avoid resource competition from simultaneous starts
ps -aux | grep python
```



## Contributing
If you want to contribute to this project, please follow these steps:
1. Fork the project repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them with descriptive commit messages.
4. Push your changes to your forked repository.
5. Create a pull request to the original repository.

## License
This project is licensed under the [MIT License]. See the `LICENSE` file for more details.
