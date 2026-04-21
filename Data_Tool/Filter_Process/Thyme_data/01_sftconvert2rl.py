import pyarrow.parquet as pq
import os
import base64
import cv2
import numpy as np
import hashlib
import json
import re
from tqdm import tqdm
def convert_base642image(base64_str,image_path):
    img_data = base64.b64decode(base64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    #根据base64_str 计算md5
    md5_str = hashlib.md5(base64_str.encode('utf-8')).hexdigest()
    image_file = os.path.join(image_path,md5_str)+".jpg"
    cv2.imwrite(image_file, img_np)
    return image_file
system_prompt = """Solve the following problem step by step. You now have the ability to selectively write executable Python code to enhance your reasoning process. The Python code will be executed by an external sandbox, and the output (wrapped in `<interpreter>output_str</interpreter>`) can be returned to aid your reasoning and help you arrive at the final answer. The Python code should be complete scripts, including necessary imports. \nEach code snippet is wrapped with `<code>\n```python\ncode snippet\n```\n</code>`.\nThe last part of your response should be in the following format:\n<answer>\n\\boxed{{'The final answer goes here.'}}\n</answer>\n\n*user question:*\nAnswer the following Math Problem and put the answer in the format of \\boxed{{answer}}\n\n{query}\n\n\nRemember to place the final answer in the last part using the format: \n<answer>\n\\boxed{{'The final answer goes here.'}}\n</answer>"""
def parse_parquet(file_path: str,all_data_list, image_path):
    # df = pd.read_parquet(file_path, engine='fastparquet')
    table = pq.read_table(file_path,use_threads=False)
    df = table.to_pandas()
    images,questions, responses = df["image"],df['question'], df['response']
    basename = os.path.basename(file_path)
    basename = basename.split(".")[0]
    image_path = os.path.join(image_path,basename)
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    idx = 0
    for image,question,response in tqdm(zip(images,questions,responses),total=len(images)):
        if image.size == 0:
            continue
        idx += 1
        rl_data_sample = {}
        image_list = []
        # import pdb;pdb.set_trace()
        for image_str in image:
            img_file=convert_base642image(image_str,image_path)
            image_list.append(img_file)
        content_match = re.search(r"<answer>(.*?)</answer>", response)
        student_answer = content_match.group(1).strip() if content_match else response.strip()
        student_answer = student_answer.replace("</answer>", "").replace("<answer>", "").strip()
        if student_answer == "":
            continue
        
        path_str = "### User Image Path:**"
        question = question.split(path_str)[0].replace("<image>\n","")
        
        rl_data_sample["question"] = question
        rl_data_sample["answer"] = student_answer
        rl_data_sample["id"] = f"{basename}_{idx}"
        messages = []
        messages.append({"role":"system","content":system_prompt})
        contents = []
        contents.append({"type":"image","image":image_list[0]})
        contents.append({"type":"text","text":question})
        messages.append({"role":"user","content":contents})

        rl_data_sample["messages"] = json.dumps(messages)
        
        all_data_list.append(rl_data_sample)

if __name__=="__main__":
    data_list = ["computation-00000-of-00003.parquet","computation-00001-of-00003.parquet","computation-00002-of-00003.parquet",]
    computation_root = "/mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_4/nathanchan/projects/retool/data/Kwai-Keye/Thyme-SFT/data/"
    image_path = "/mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_4/nathanchan/projects/retool/data/Thyme/sft/images"
    all_data_list = []
    for data_file in data_list:
        print(f"processing {data_file}")
        data_file = os.path.join(computation_root,data_file)
        parse_parquet(data_file, all_data_list, image_path)
    output_path = "./thyme_sft_computation.jsonl"
    with open(output_path,"w") as f:
        for data_line in all_data_list:
            f.write(json.dumps(data_line)+"\n")
    print(f"processing")
    # df = parse_parquet(computation_parquet)