#!/usr/bin/env python
# filepath: d:\SynologyDrive\project\jailbreak_grpo\approach_ours\unlockJsonItems.py

import os
import json
import time
import glob
import argparse
import portalocker


def unlock_all_processing_items(json_file_path):
    """
    将指定 JSON 文件中所有状态为 "processing" 的条目重置为 "-1"

    参数:
        json_file_path (str): JSON 文件的路径

    返回:
        int: 解锁的条目数量
    """
    max_retries = 5  # 最大重试次数
    retry_delay = 0.5  # 每次重试的延迟时间（秒）
    unlocked_count = 0

    print(f"尝试解锁文件: {json_file_path}")

    # 尝试多次获取文件锁，避免因暂时性问题而失败
    for attempt in range(max_retries):
        try:
            # 使用文件锁确保互斥访问，超时10秒
            with portalocker.Lock(json_file_path, 'r+', timeout=10, encoding='utf-8') as file:
                # 读取文件内容
                file.seek(0)  # 将文件指针移到开头
                content = file.read()  # 读取整个文件内容

                # 如果文件为空，初始化为空列表
                if content.strip():  # 检查文件内容是否非空
                    data = json.loads(content)  # 解析JSON内容
                else:
                    print(f"文件 {json_file_path} 为空或格式不正确")
                    return 0

                # 查找并重置所有 "processing" 状态的条目
                modified = False
                for item in data:
                    if item.get("stage") == "processing":
                        item["stage"] = "-1"  # 重置为待处理状态
                        unlocked_count += 1
                        modified = True

                # 如果有修改，写回文件
                if modified:
                    file.seek(0)  # 将文件指针移到开头
                    file.truncate()  # 清空文件内容
                    json.dump(data, file, ensure_ascii=False,
                              indent=4)  # 写入修改后的数据
                    print(f"已成功解锁 {unlocked_count} 个处于 'processing' 状态的条目")
                else:
                    print(f"文件中没有找到处于 'processing' 状态的条目")

                return unlocked_count

        except portalocker.LockException:
            print(
                f"尝试 {attempt+1}/{max_retries}: 无法获取文件锁，等待 {retry_delay} 秒后重试...")
            time.sleep(retry_delay)
        except json.JSONDecodeError:
            print(f"错误: 文件 {json_file_path} 不是有效的 JSON 格式")
            return 0
        except Exception as e:
            print(f"尝试 {attempt+1}/{max_retries}: 处理文件时发生错误: {str(e)}")
            if attempt == max_retries - 1:
                print(f"达到最大重试次数，放弃处理文件 {json_file_path}")
                return 0
            time.sleep(retry_delay)

    return 0  # 所有尝试都失败


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="解锁 JSON 文件中状态为 'processing' 的条目")
    parser.add_argument('--result_dir', type=str, required=True, help="结果目录路径")
    args = parser.parse_args()

    # 确保目录存在
    if not os.path.exists(args.result_dir):
        print(f"错误: 目录 '{args.result_dir}' 不存在")
        return

    # 查找所有符合条件的 JSON 文件
    json_pattern = os.path.join(args.result_dir, "*_PAP_SCANING.json")
    json_files = glob.glob(json_pattern)

    if not json_files:
        print(f"在目录 '{args.result_dir}' 中未找到 *_PAP_SCANING.json 文件")
        return

    total_unlocked = 0

    # 处理每个找到的文件
    for json_file in json_files:
        print(f"\n处理文件: {json_file}")
        unlocked = unlock_all_processing_items(json_file)
        total_unlocked += unlocked

    print(f"\n总结: 共解锁了 {total_unlocked} 个条目")

    # 执行完成后休眠一秒
    print("\n操作完成，休眠1秒...")
    time.sleep(1)


if __name__ == "__main__":
    main()
