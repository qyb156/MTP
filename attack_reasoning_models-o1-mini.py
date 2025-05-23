
import os
import json
import glob
from AgentVictimLLM_Defense import attacker
from agentReward import myreward


def process_json_files(folder_path, victim_model="o1-mini"):
    """
    遍历指定文件夹中的所有JSON文件，提取turns字段内容

    参数:
        folder_path: JSON文件所在的文件夹路径
    """
    # 确保文件夹路径存在
    if not os.path.exists(folder_path):
        print(f"错误: 文件夹 '{folder_path}' 不存在")
        return

    # 获取文件夹中所有JSON文件的路径
    json_files = glob.glob(os.path.join(folder_path, "*.jsonl"))

    if not json_files:
        print(f"警告: 在 '{folder_path}' 中未找到JSON文件")
        return

    print(f"发现 {len(json_files)} 个JSON文件")

    # 遍历所有JSON文件
    for json_file in json_files:

        attack_succeed = 0
        file_name = os.path.basename(json_file)

        # 读取JSON文件内容
        with open(json_file, 'r', encoding='utf-8') as f:
            # 按行读取JSON文件
            for line_num, line in enumerate(f, 1):
                try:
                    # 解析当前行的JSON对象
                    data = json.loads(line.strip())

                    # 提取turns字段内容
                    if "turns" in data and isinstance(data["turns"], list) and data["turns"]:
                        for turn in data["turns"]:
                            # print(f"\n处理文件: {file_name}")
                            # # 输出每个turn内容
                            # print(f"{turn}")
                            response = attacker.inference(turn, victim_model)
                            _, jud = myreward.judge(turn, response)
                            print("-" * 50)

                            print(
                                f"judge的得分是(1表示是成功的，0表示是不成功的，-1表示是异常了)：{jud}")
                            if int(jud) == 1:
                                attack_succeed += 1
                            print(
                                f"目标是：{turn},输出是：{response}，当前攻击成功的次数是：{attack_succeed}")

                except json.JSONDecodeError:
                    print(f"警告: 第 {line_num} 行不是有效的JSONL格式，已跳过")
                except Exception as e:
                    print(f"处理第 {line_num} 行时出错: {str(e)}")
        print("--输出结果----" * 4)
        print(
            f"\n处理文件: {file_name},受害模型是:{victim_model},攻击成功率为：{round(attack_succeed/440, 2)}")


if __name__ == "__main__":
    # 添加变量受害模型
    victim_model = "o1-mini"
    # 设置要处理的文件夹路径
    folder_path = "sorry_bench_for_reasoning_models"

    # 处理文件夹中的所有JSON文件
    process_json_files(folder_path, victim_model)

    # process_json_files(folder_path, victim_model)

    # 在Linux中运行的命令如下：nohup python attack_reasoning_models-o1-mini.py > attack_reasoning_models-o1-mini.log 2>&1 &
    # 如果你需要停止程序，可以先找到进程 ID，然后使用 kill 命令：
    #     ps aux | grep attack_reasoning_models.py
    # kill - 9 [进程ID]
    # 例如：kill -9 12345
