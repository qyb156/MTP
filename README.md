

# QueryAttack

This is the repository that contains the source code for RPJ. 

## Project Structure

This project mainly contains the following Python files:

--'main.py' is the main program file;  
--'agentReward.py' is used to calculate the rewards for the large language model;  
--'AgentVictimLLM_Defense.py' is used to obtain feedback from the victim model.

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
