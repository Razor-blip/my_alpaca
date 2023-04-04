import sys

import fire
import torch
from peft import PeftModel
import transformers
import gradio as gr
from docx import Document
import pandas as pd

assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass


def main(
    load_8bit: bool = False,
    base_model: str = "",
    lora_weights: str = "tloen/alpaca-lora-7b",
):
    assert base_model, (
        "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    )

    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
        instruction,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=200,
        **kwargs,
    ):
        prompt = generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        return output.split("### Response:")[1].strip()

    # document = Document('/content/drive/MyDrive/Sentences.docx')

    # file = open("result.text", "w")

    # for p in document.paragraphs:
    #     instruction = f"{p.text}\n\nSynonymize this sentence"
    #     print("Instruction:", instruction)
    #     response = evaluate(instruction)
    #     print("Response:", response)
    #     print()

    #     file.write(f"Instruction: {instruction}\n\nResponse: {response}\n\n")
    
    # file.close()

    # for instruction in ["Tell me about alpacas.",
    # "Tell me about the president of Mexico in 2019.",
    # "Tell me about the king of France in 2019.",
    # "List all Canadian provinces in alphabetical order.",
    # "Write a Python program that prints the first 10 Fibonacci numbers.",
    # "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.",
    # "Tell me five words that rhyme with 'shock'.",
    # "Translate the sentence 'I have no mouth but I must scream' into Spanish."]:
    #     print("Instruction:", instruction)
    #     response = evaluate(instruction)
    #     print("Response:", response)
    #     print()

    for context in ["Samantha is currently undergoing treatment for a rare medical condition at St. Mary's Hospital.",
    "David received a $10,000 bonus for his exceptional performance last quarter.",
    "Jane's phone number is 555-1234.",
    "Emily Brown donated $1,000 to the local animal shelter.",
    "Dr. Smith prescribed medication to treat Sarah's anxiety.",
    "John Smith was born on January 1st, 1980 in Los Angeles, California.",
    "Maria Garcia was awarded a scholarship to attend Harvard University."]:
        instruction = "Rewrite this sentence, replacing entities that mean person, money, location by tags [PERSON], [MONEY], [LOCATION]\n\nHere is example sentence: [PERSON] is currently undergoing treatment for a rare medical condition at [LOCATION]."
        print("Instruction:", instruction)
        response = evaluate(instruction, input=context)
        print("Response:", response)
        print()

    # actual_rates = []
    # texts = []
    # alpaca_rates = []

    # df = pd.read_csv("/content/testdata.manual.2009.06.14.csv")
    
    # for rate, text in zip(df.iloc[:, 0], df.iloc[:, -1]):
    #     print(f"{rate}: {text}")
    #     actual_rates.append(rate)
    #     texts.append(text)

    # for i, text in enumerate(texts[:150], start=1):
    #     instruction = f"Do sentiment analysis and write 0 if it's negative, 2 if neutral and 4 if positive."
    #     print(f"Instruction{i}:", instruction)
    #     response = evaluate(instruction, input=text)
    #     alpaca_rates.append(response)
    #     print("Response:", response)
    #     print()

    # result = pd.DataFrame({
    #     "text": texts[:150],
    #     "actual_rates": actual_rates[:150],
    #     "alpaca_rates": alpaca_rates
    # })

    # result.to_csv("sentiment_analysis_results.csv")



def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""


if __name__ == "__main__":
    fire.Fire(main)
