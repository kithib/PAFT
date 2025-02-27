from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 加载模型和处理器
model = LlavaForConditionalGeneration.from_pretrained("/apdcephfs_cq10/share_1567347/kitwei/LLava").to(device)
processor = AutoProcessor.from_pretrained("/apdcephfs_cq10/share_1567347/kitwei/LLava")

# 定义提示
prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"

# 读取本地图片
image_path = "/apdcephfs_cq10/share_1567347/kitwei/LLaMA-Factory/data/mllm_data/1.jpg"  # 替换为你的本地图片路径
image = Image.open(image_path)

# 处理输入
inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

# 生成输出
with torch.no_grad():  # 关闭梯度计算以节省内存
    generate_ids = model.generate(**inputs, max_new_tokens=500)

output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

print(output)
print(model)
for name, module in model.named_modules():
    print(name, module)