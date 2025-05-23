

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
#!/bin/bash
# 初始化conda
eval "$(conda shell.bash hook)"
conda activate jailbreak
cd /***/jailbreak_grpo/

echo "关闭所有的python进程。。" 
pkill -u $(whoami) -f python
sleep 2 # 添加短暂延迟，避免同时启动造成资源竞争
ps -aux | grep python

# 通用参数
VICTIM_MODEL="claude-2"
HOST="http://localhost:11434"

num_return_sequences=6
EPOCHS=6
DATA_SOURCE="PAP_prompts.csv"
MODEL_PATH="./Qwen/Qwen2.5-7B-Instruct"
FIFTH_STAGE="true"

# 定义GPU起始索引和结束索引
START_GPU=1 # 共有八块GPU，从GPU 0开始。现在预留GPU 0给ollama程序使用，已经在配置文件中指定完毕。
END_GPU=7   # 到GPU 7结束

# 提取数据源文件名（去除.csv后缀）
DATA_SOURCE_NAME=$(basename "${DATA_SOURCE}" .csv)

# 提取模型名称（从路径中提取最后一个目录名）
MODEL_NAME=$(basename "${MODEL_PATH}")

# 构建实验名称，使用对应的变量
experiment_name="${VICTIM_MODEL}_${DATA_SOURCE_NAME}_${MODEL_NAME}"



# 添加时间戳
timestamp=$(date +"%Y%m%d_%H%M%S")
log_dir="logs/${experiment_name}_${timestamp}"

# 创建日志目录
mkdir -p ${log_dir}

# 记录实验信息
echo "实验开始时间: $(date)" > "${log_dir}/experiment_info.txt"
echo "实验名称: ${experiment_name}" >> "${log_dir}/experiment_info.txt"
echo "使用模型: ${VICTIM_MODEL}" >> "${log_dir}/experiment_info.txt"
echo "GPU范围: ${START_GPU}-${END_GPU}" >> "${log_dir}/experiment_info.txt"


# 检查必要的结果文件是否存在
result_dir="./RESULTS_${DATA_SOURCE_NAME}/victim_model_${VICTIM_MODEL}_qwen0.5b"
# 首先是强制解锁json文件，有可能程序被强制终止以后，在json中大量的条目被锁定了，状态被设置为 了processing。我们需要将这些条目解锁。
# 这个命令会将所有的条目都解锁。
python unlockJsonItems.py --result_dir ${result_dir} > unlock.log 2>&1 & 
  
# 运行函数
run_experiment() {
    local gpu_id=$1
    # GPU ID现在是从0开始，直接用作CUDA设备ID
    local cuda_device=${gpu_id}
    
    # 修改日志文件名格式，使用对应的变量
    local log_file="${log_dir}/output_gpu${gpu_id}_${VICTIM_MODEL}-${DATA_SOURCE_NAME}_${MODEL_NAME}-${gpu_id}.log"
    
    echo "启动 GPU ${gpu_id} 的实验 (CUDA设备: ${cuda_device})..."
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
     
    echo "GPU ${gpu_id} 实验已启动，日志文件: ${log_file}"
}

# 启动指定范围的GPU实验
for i in $(seq ${START_GPU} ${END_GPU}); do
    run_experiment $i
    sleep 0.1  # 添加短暂延迟，避免同时启动造成资源竞争
done

echo "所有实验已启动，日志文件保存在: ${log_dir}"
echo "使用以下命令查看各GPU运行状态："
for i in $(seq ${START_GPU} ${END_GPU}); do
    echo "GPU ${i}: tail -f ${log_dir}/output_gpu${i}_${VICTIM_MODEL}-${DATA_SOURCE_NAME}_${MODEL_NAME}-${i}.log"
done

# 显示正在运行的进程
sleep 2
echo "所有程序已启动，查看各GPU上运行的进程："
nvidia-smi
sleep 2 # 添加短暂延迟，避免同时启动造成资源竞争
ps -aux | grep python



## Contributing
If you want to contribute to this project, please follow these steps:
1. Fork the project repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them with descriptive commit messages.
4. Push your changes to your forked repository.
5. Create a pull request to the original repository.

## License
This project is licensed under the [MIT License]. See the `LICENSE` file for more details.
解释
