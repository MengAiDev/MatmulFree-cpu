import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# More thorough method to disable JIT
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
os.environ["PYTORCH_JIT"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Ensure no GPU is used

# Disable JIT before importing torch
import torch
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(False)
torch._C._jit_override_can_fuse_on_gpu(False)
torch._C._set_graph_executor_optimize(False)

# Completely disable JIT compilation
torch.jit._state.disable()

# Set torch backend to CPU to avoid CUDA related errors
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

import mmfreelm
from transformers import AutoModelForCausalLM, AutoTokenizer

name = 'ridger/MMfreeLM-2.7B'
tokenizer = AutoTokenizer.from_pretrained(name)

# Set pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Get model's maximum length
model_max_length = tokenizer.model_max_length
if model_max_length > 10000000000000000:  # Handle invalid large values
    model_max_length = 512

print(f"Using model_max_length: {model_max_length}")

# Create model - use float32 to avoid mixed precision issues
model = AutoModelForCausalLM.from_pretrained(
    name, 
    trust_remote_code=True,
    torch_dtype=torch.float32,
)

input_prompt = "The future of AI is"

# Prepare input
inputs = tokenizer(
    input_prompt, 
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=model_max_length
)

# Force CPU usage
device = torch.device("cpu")
print(f"Using device: {device}")
model = model.to(device)
model.eval()  # Set to evaluation mode

# Prepare input data
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)

try:
    # Use minimal generation parameters
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=8,
        do_sample=True,    # Disable sampling
        num_beams=1,       # Use greedy search
        use_cache=False,   # Disable cache (might trigger Jiterator)
        temperature=0.6,   # Neutral temperature
        top_p=0.4
    )
    print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    
except Exception as e:
    print(f"Unexpected error during generation: {e}")
    print("Trying manual generation loop...")
    
    # Try manual generation loop
    generated = input_ids
    for _ in range(32):  # Generate up to 32 tokens
        with torch.no_grad():
            outputs = model(
                input_ids=generated,
                attention_mask=torch.ones_like(generated) if attention_mask is None else attention_mask
            )
            
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        
        # Stop if end token is encountered
        if next_token.item() == tokenizer.eos_token_id:
            break
            
        generated = torch.cat([generated, next_token], dim=-1)
        print(tokenizer.decode(generated[0], skip_special_tokens=True))
    
    print("Final output:")
    print(tokenizer.decode(generated[0], skip_special_tokens=True))