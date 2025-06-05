import json
import os
import asyncio

import magic
from aiolimiter import AsyncLimiter
from tqdm import tqdm

from api import bin_to_base64, call_gemini
from data_loader import DataLoader_MMEureka

def encode_image(image):
    mime = magic.from_file(image, mime=True)
    return {"inlineData": {"mimeType": mime, "data": bin_to_base64(image)}}

def build_message(item, generation_config):
    question = item["question"]
    # image = item["images"][0]
    image = item["image_path"]

    if MODEL == "gemini-2.5-flash-preview-04-17":
        generation_config |= {"thinkingConfig": {"thinkingBudget": 0}}
    try:
        messages = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        encode_image(image),
                        {"text": question},
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
            with open(out_file.replace(".jsonl","_err.jsonl"), "w") as fp:
                json.dump(res, fp, ensure_ascii=False)
                fp.write("\n")
            continue
        # import pdb;pdb.set_trace()
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
    data_loader = DataLoader_MMEureka(inp)
    
    for line in tqdm(data_loader.data[:total], total=total,desc="Processing data gemini:"):
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