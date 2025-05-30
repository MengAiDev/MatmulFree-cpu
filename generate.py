import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import mmfreelm
from transformers import AutoModelForCausalLM, AutoTokenizer
#Change here to our open-sourced model
name = 'ridger/MMfreeLM-370M'
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True)
input_prompt = "In a shocking finding, scientist discovered a herd of unicorns living in a remote, "
input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids
try:
    outputs = model.generate(
        input_ids, 
        max_length=32,  
        do_sample=True, 
        top_p=0.4, 
        temperature=0.6
    )
    print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
except AttributeError as e:
    print(f"Generation failed: {e}. Please ensure model architecture inherits from GenerationMixin.")
except Exception as e:
    print(f"Unexpected error during generation: {e}")