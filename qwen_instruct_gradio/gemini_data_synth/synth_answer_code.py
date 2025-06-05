import json
import os
import asyncio

import magic
from aiolimiter import AsyncLimiter
from tqdm import tqdm

from api import bin_to_base64, call_gemini
from data_loader import DataLoader_MMEureka,DataLoader_MMEureka_CoT

rewrite_prompt = '''
You are a helpful AI assistant. You are provided with a complete problem-solving process. Please identify parts of the solution that would benefit from assistance using Python code. Rewrite those parts with Python code that is syntactically correct and directly executable. Ensure that:

1. The code includes necessary import statements for any dependencies.
2. Each step in the code contains comments that explain the purpose of the step.
3. Avoid redundancy between the code comments and the problem statement; ensure that the comments do not repeat descriptions already present in the problem statement.
4. Maintain the original format of the problem-solving process, preserving elements such as <think></think>, <answer></answer>, with the code primarily located within <think></think>, and <answer></answer> succinctly summarizing the process.
5. Use markdown syntax to encapsulate code blocks.
6. Ensure code blocks coexist with conventional reasoning parts, maintaining a balanced integration.


Here's the current problem-solving process:

<think>
[Incorporate Python code where needed, using markdown syntax to delineate code blocks. The solution should seamlessly integrate code with conventional text explanations.]
</think>

<answer>
[Provide a concise explanation of the solution, highlighting the outcomes achieved through both programmatic and conventional reasoning.]
</answer>

'''

rewrite_prompt='''
You are presented with a complete problem-solving process. Your task is to review the process and determine whether certain parts could be enhanced using Python code. This is not mandatory for every part; if conventional reasoning suffices, return the original process. However, if computational assistance, such as calculations or data processing, would improve the solution, integrate Python code accordingly. Please adhere to the following guidelines:

1. Ensure the code is syntactically correct and directly executable; include any required import statements for dependencies.
2. Provide comments within the code to clarify the purpose of each step.
3. Avoid redundancy between the code comments and the problem description; comments should not reiterate information already given in the text.
4. Preserve the original format of the problem-solving process with <think>xxx</think> and <answer>xxx</answer> tags, placing code mainly within <think>xxx</think>, while <answer>xxx</answer> should succinctly conclude the process.
5. Use markdown syntax to define code blocks.
6. Ensure a balanced integration where code blocks coexist with narrative reasoning, not replacing the entire process with code.


Example of a Completed Process:

<think>
[Determine the parts of the process where Python code can enhance efficiency. Use markdown to include Python code blocks alongside narrative explanations. Example:]
```
python
# Import necessary libraries
import numpy as np

# Calculate the sum of an array
array = np.array([1, 2, 3, 4])
sum_of_array = np.sum(array)
```
[Continue with conventional explanations where appropriate.]
</think>

<answer>
[Provide a concise summary of the results and insights obtained through both computational and narrative analysis.]
</answer>

'''
rewrite_prompt_v1='''
You are provided with a complete problem-solving process. Evaluate this process and determine whether specific parts could be enhanced using Python code. It is not mandatory to integrate code into every solution aspect; retain the original approach if conventional reasoning suffices. However, incorporate Python where computational tasks such as calculations or data processing would benefit. Please follow these guidelines:

Ensure the code is syntactically correct and directly executable, including necessary import statements for any dependencies.
Include comments within the code to explain the purpose of each step, avoiding repetition of the existing problem description.
Preserve the original problem-solving format using <think>xxx</think> and <answer>xxx</answer> tags. Code should primarily be located within <think>xxx</think>, and <answer>xxx</answer> should provide a concise summary.
Use markdown syntax to define code blocks, with the code block delimited by <code>xxx</code>.
Ensure code blocks coexist with narrative reasoning, maintaining an integrated solution approach.
For each code block, print only the final result using a print statement, with the output confined within <interpreter>xxx</interpreter> tags.


Example of a Completed Process:

<think>
[Identify sections where Python code can enhance efficiency. Implement code blocks using markdown syntax alongside narrative explanations. For example:]

<code>
```
# Import necessary library
import numpy as np

# Perform calculations: summing elements in an array
array = np.array([1, 2, 3, 4])
sum_of_array = np.sum(array)

# Print the final result
print("Sum of array:", sum_of_array)
```
</code>
<interpreter>
Sum of array: 10
</interpreter>

[Continue with conventional explanations where necessary.]

<code>
```
# Compute the maximum value in the array
max_value = np.max(array)

# Print the final result
print("Maximum value:", max_value)
```
</code>
<interpreter>
Maximum value: 4
</interpreter>
</think>

<answer>
[Summarize the main findings and insights obtained through both computational and narrative parts.]
</answer>

'''

from prompt_template import rewrite_prompt_v2,rewrite_prompt_v3

rewrite_prompt = rewrite_prompt_v2


def encode_image(image):
    mime = magic.from_file(image, mime=True)
    return {"inlineData": {"mimeType": mime, "data": bin_to_base64(image)}}

def build_message(item, generation_config):
    question = item["response"]
    # image = item["images"][0]

    if MODEL == "gemini-2.5-flash-preview-04-17":
        generation_config |= {"thinkingConfig": {"thinkingBudget": 0}}
    try:
        messages = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text":rewrite_prompt + question},
                    ],
                }
            ],
            "generation_config": {
                "temperature": 1.0,
                "topP": 0.95,
                "maxOutputTokens": 4096,
            }
            | generation_config,
        }
    except Exception as e:
        print(f" {item['id']} {e}")

    return messages


async def regen_answer_async(item, limiter, generation_config={}):
    messages = build_message(item, generation_config)
    async with limiter:
        # Run synchronous call_gemini in a thread to avoid blocking
        answer = await asyncio.to_thread(call_gemini, MODEL, messages)
    return answer


async def process_batch(batch, key, limiter, out_file):
    tasks = []
    for item in batch:
        res = {key: item[key], "err": None, "ans": None, "model": MODEL}
        task = asyncio.create_task(regen_answer_async(item, limiter))
        
        tasks.append((res, task))

    for res, task in tasks:
        try:
            res["ans"] = await task
        except Exception as e:
            res["err"] = str(e)
            with open(out_file.replace(".jsonl","_err.jsonl"), "a") as fp:
                json.dump(res, fp, ensure_ascii=False)
                fp.write("\n")
            continue
        with open(out_file, "a") as fp_out:
            json.dump(res, fp_out, ensure_ascii=False)
            fp_out.write("\n")

async def main(
    inp="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/hancong/code/pretrain/reasoning/data/Data_Tool/Filter_Process/acc_res_20k_debug.jsonl",
    out="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/hancong/code/pretrain/reasoning/data/MM-Eureka-Dataset/synth_gemini_cot_code/dataset_20k_cot_code.jsonl",
    key="id",
    rpm=10,
    model="gemini-2.5-flash-preview-04-17",
    total=100,
):
    global MODEL
    MODEL = model

    # with open(inp) as fp:
    #     total = sum(1 for _ in fp)
    # total = 10000
    done = set()

    if os.path.exists(out):
        with open(out) as fp:
            for line in tqdm(fp):
                item = json.loads(line)
                if item["err"] is None:
                    done.add(item[key])

    limiter = AsyncLimiter(rpm, 60)
    batch_size = rpm
    current_batch = []
    data_loader = DataLoader_MMEureka_CoT(inp)
    
    
    for line in tqdm(data_loader.data[:total], total=total,desc="Processing data gemini code"):
        # item = json.loads(line)
        item = line
        if item[key] not in done:
            current_batch.append(item)

            if len(current_batch) >= batch_size:
                await process_batch(current_batch, key, limiter, out)
                current_batch = []

    # Process remaining items
    if current_batch:
        await process_batch(current_batch, key, limiter, out)

if __name__ == "__main__":
    import fire

    fire.Fire(main)