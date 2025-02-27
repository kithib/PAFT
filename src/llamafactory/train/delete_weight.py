from transformers import AutoModelForCausalLM, AutoConfig

# 加载预训练模型
model = AutoModelForCausalLM.from_pretrained(
    "/apdcephfs_cq10/share_1567347/kitwei/LLaMA-Factory/saves/llama3-8b/delete_test/hellaswag_qvproj_test_sft", 
    device_map="auto", 
    trust_remote_code=True
)

# 获取模型的状态字典
state_dict = model.state_dict()

# 计算参数数量
total_params = sum(p.numel() for p in state_dict.values())

# 每个参数的大小（以字节为单位），假设是32位浮点数（4字节）
param_size = 4

# 计算总的显存需求（以字节为单位）
total_memory = total_params * param_size

# 将字节转换为GB
total_memory_in_gb = total_memory / (1024 ** 3)

print(f"模型所需的显存: {total_memory_in_gb:.2f} GB")

# 过滤掉与 LoRA 相关的参数
filtered_state_dict = {k: v for k, v in state_dict.items() if "lora" not in k.lower()}

# 计算参数数量
total_params = sum(p.numel() for p in filtered_state_dict.values())

# 每个参数的大小（以字节为单位），假设是32位浮点数（4字节）
param_size = 4

# 计算总的显存需求（以字节为单位）
total_memory = total_params * param_size

# 将字节转换为GB
total_memory_in_gb = total_memory / (1024 ** 3)

print(f"模型所需的显存（不包括 LoRA 参数）: {total_memory_in_gb:.2f} GB")