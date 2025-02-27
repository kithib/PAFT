import subprocess
import os
import pandas as pd
from openpyxl import Workbook
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def run_command_with_gpu(prompt, gpu_id):
    # 定义要执行的命令
    # /apdcephfs_cq10/share_1567347/kitwei/LLaMA-Factory/model/llama3-8b/flora_v1/winogrande_qvproj_lora_sft
    # /apdcephfs_cq10/share_1567347/kitwei/LLaMA-Factory/model/llama3-8b/prompt/winogrande_sft_random_prompt_1_3epoch
    command = [
        'bash', '-c',  # 使用 Bash 执行命令
        f'export CUDA_VISIBLE_DEVICES={gpu_id} && '  # 设置环境变量
        'python3 run.py '
        f'--datasets winogrande_gen_458220-{gpu_id} '
        '--hf-type base '
        '--hf-path /home/chenxing/Meta-Llama-3-8B '
        '--peft-path /home/chenxing/model/winogrande_sft_ZOPO_prompt '
        '--num-gpus 1 '
        '--debug'
    ]
    try:
        # 执行命令
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        print(f"{prompt} success")
        # 打印标准输出和错误输出
        return result.stdout

    except subprocess.CalledProcessError as e:
        print(f"命令执行失败，返回码: {e.returncode}")
        print("错误输出:\n", e.stderr)
        return ''


def split_prompts(file_path, num_parts=8):
    # 读取 Excel 文件
    df = pd.read_excel(file_path)

    # 检查是否存在 'prompt' 列
    if 'text' not in df.columns:
        raise ValueError("Excel 文件中没有 'prompt' 列。")

    # 提取 'prompt' 列并转换为列表
    prompts = df['text'].tolist()

    # 计算每份的大小
    part_size = len(prompts) // num_parts
    remainder = len(prompts) % num_parts

    # 将 prompts 平均分为指定数量的部分
    parts = []
    start_index = 0

    for i in range(num_parts):
        # 计算当前部分的大小
        current_part_size = part_size + (1 if i < remainder else 0)
        end_index = start_index + current_part_size
        parts.append(prompts[start_index:end_index])
        start_index = end_index

    return parts

def replace_line_in_file(file_path, new_line_content, line_number):
    """
    替换指定文件中的指定行内容。

    :param file_path: 文件路径
    :param new_line_content: 新的行内容
    :param line_number: 要替换的行号（从 1 开始）
    """
    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 检查行号是否有效
    if line_number < 1 or line_number > len(lines):
        raise ValueError(f"行号 {line_number} 超出范围，文件总行数为 {len(lines)}。")

    # 修改指定行
    lines[line_number - 1] = new_line_content + '\n'  # 注意：列表索引从 0 开始

    # 将修改后的内容写回文件
    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(lines)

    print(f"第 {line_number} 行已成功修改。")

def run_prompt_part(gpu_id, prompt_part, ws):
    """运行每个提示部分并将结果写入 Excel。"""
    file_path = f"/home/chenxing/opencompass/configs/datasets/winogrande/winogrande_gen_458220-{gpu_id}.py"
    line_number = 18
    results = []
    for prompt in prompt_part:
        new_line_content = f"""                dict(role='HUMAN', prompt="{str(prompt)}")"""
        replace_line_in_file(file_path, new_line_content, line_number)
        result = run_command_with_gpu(prompt, gpu_id)
        results.append([prompt, result])
        print([prompt, result])

    return results


if __name__ == "__main__":
    prompt_path = "/home/chenxing/opencompass/prompt/robust_testdataset_win.xlsx"
    save_path = "/home/chenxing/opencompass/prompt/robust_testdataset_winogrande_sft_ZOPO_prompt_result_time.xlsx"
    # 给每个GPU分一块
    wb = Workbook()
    ws = wb.active
    ws.append(['prompt', 'acc'])
    num_parts = 8  # 可以根据需要更改分割的部分数量
    prompt_parts = split_prompts(prompt_path, num_parts)
    # 使用线程池并可视化进度
    with ThreadPoolExecutor(max_workers=num_parts) as executor:
        futures = {executor.submit(run_prompt_part, gpu_id, prompt_parts[gpu_id], ws): gpu_id for gpu_id in range(num_parts)}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            gpu_id = futures[future]
            try:
                results = future.result()
                for prompt, result in results:
                    ws.append([prompt, result])
            except Exception as e:
                print(f"GPU {gpu_id} 处理时发生错误: {e}")

    wb.save(save_path)

