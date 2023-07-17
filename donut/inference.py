from PIL import Image
import re
from transformers import DonutProcessor
from datasets import load_dataset
from optimum.onnxruntime import ORTModelForVision2Seq
import requests

model_name = "donut_docvqa_onnx"
processor = DonutProcessor.from_pretrained(model_name)
model = ORTModelForVision2Seq.from_pretrained(model_name)

# device = "cuda" if torch.cuda.is_available() else "cpu" 
device = "cpu"
model.to(device)

task_prompt = "<s_docvqa><s_question>{user_input}</s_question><s_answer>"
question = "When is the coffee break?"
prompt = task_prompt.replace("{user_input}", question)
decoder_input_ids = processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids

dataset = load_dataset("hf-internal-testing/example-documents", split="test")
image = dataset[0]["image"]
pixel_values = processor(image, return_tensors="pt").pixel_values

outputs = model.generate(pixel_values)
sequence = processor.batch_decode(outputs)[0]
sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
print(processor.token2json(sequence))

# {'text_sequence': 'ballroom foyer</s_answer>'}
