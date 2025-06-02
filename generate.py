import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 使用更彻底的方法禁用JIT
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
os.environ["PYTORCH_JIT"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 确保没有GPU被使用

# 在导入torch之前禁用JIT
import torch
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(False)
torch._C._jit_override_can_fuse_on_gpu(False)
torch._C._set_graph_executor_optimize(False)

# 完全禁用JIT编译
torch.jit._state.disable()

# 设置torch后端为CPU，避免CUDA相关错误
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

import mmfreelm
from transformers import AutoModelForCausalLM, AutoTokenizer

name = 'ridger/MMfreeLM-1.3B'
tokenizer = AutoTokenizer.from_pretrained(name)

# 设置pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 获取模型的最大长度
model_max_length = tokenizer.model_max_length
if model_max_length > 10000000000000000:  # 处理无效的大数值
    model_max_length = 512

print(f"Using model_max_length: {model_max_length}")

# 创建模型 - 使用float32避免混合精度问题
model = AutoModelForCausalLM.from_pretrained(
    name, 
    trust_remote_code=True,
    torch_dtype=torch.float32
)

input_prompt = "The future of AI is"

# 准备输入
inputs = tokenizer(
    input_prompt, 
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=model_max_length
)

# 强制使用CPU
device = torch.device("cpu")
print(f"Using device: {device}")
model = model.to(device)
model.eval()  # 设置为评估模式

# 准备输入数据
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)

try:
    # 使用最简化的生成参数
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=32,
        do_sample=True,  # 禁用采样
        num_beams=1,      # 使用贪婪搜索
        use_cache=False,   # 禁用缓存（可能触发Jiterator）
        temperature=0.6,  # 中性温度
        top_p=0.4
    )
    print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    
except Exception as e:
    print(f"Unexpected error during generation: {e}")
    print("Trying manual generation loop...")
    
    # 尝试手动生成循环
    generated = input_ids
    for _ in range(32):  # 最多生成32个token
        with torch.no_grad():
            outputs = model(
                input_ids=generated,
                attention_mask=torch.ones_like(generated) if attention_mask is None else attention_mask
            )
            
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        
        # 如果遇到结束符则停止
        if next_token.item() == tokenizer.eos_token_id:
            break
            
        generated = torch.cat([generated, next_token], dim=-1)
        print(tokenizer.decode(generated[0], skip_special_tokens=True))
    
    print("Final output:")
    print(tokenizer.decode(generated[0], skip_special_tokens=True))