import pyarrow.parquet as pq
import pandas as pd
import os
import json
from tqdm import tqdm
system_prompt4_2509="Solve the following problem step by step. Your answer must be in latex format and wrapped in $...$. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, which means your output should start with <think> and end with </answer>. You now have the ability to selectively write executable Python code to enhance your reasoning process. For tasks involving complex numerical calculations, program flowcharts, iterative computations, etc., prioritize implementing them via Python scripts and output the results using $print()$. The Python code will be executed by an external sandbox, and the output (wrapped in `<interpreter>output_str</interpreter>`) can be returned to aid your reasoning and help you arrive at the final answer. The Python code should be complete scripts, including necessary imports. \nEach code snippet is wrapped with `<code>\n```python\ncode snippet\n```\n</code>`.\nFor example, <think> This is the reasoning process. <code> python code here </code> <interpreter> python interpreter result here </interpreter> This is the continuation of the reasoning process. </think> <answer> The final answer is  $\\boxed{answer here}$ </answer>. In the last part of the answer, the final exact answer is enclosed within $\\boxed{}$ with latex format."
img_path_root = "/mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_4/nathanchan/projects/retool/data/TIGER-Lab/ViRL39K"
def analyze_question(question,answer, qid, image):
    result ={}
    question = question.replace("<image>\n","")
    result["question"] = question
    result["answer"] = answer
    result["id"] = qid
    result["message"] = ""
    # import pdb;pdb.set_trace()
    contents = []
    #添加dict:{"role":"system","content":xxx}
    contents.append({"role":"system","content":system_prompt4_2509})

    content = []
    content.append({"type":"image","image":os.path.join(img_path_root,image[0])})    
    content.append({"type":"text","text":question})
    contents.append({"role":"user","content":content})

    result["message"] = json.dumps(contents)
    return result

def process_row(row):
    """处理单行数据的函数"""
    question = row['question']
    answer = row["answer"]
    qid = row["qid"]
    image_paths = row['image']
    
    # 您的处理逻辑
    result = analyze_question(question, answer, qid,image_paths)

    return result

def parse_virl_parquet(path,need_cols, save_jsonl):
    df_data= pd.read_parquet(path, columns=need_cols)

    results = df_data.apply(process_row, axis=1)

    # save jsonl 
    with open(save_jsonl, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    return results
    
def filter_claude(all_data_jsonl, claude_filter):
    with open(claude_filter, 'r') as f:
        claude_filter_ids = json.load(f)
    
    with open(all_data_jsonl, "r") as f:
        all_data = f.readlines()
    # import pdb;pdb.set_trace()
    res_data = []
    for data_line in all_data:
        data_dict = json.loads(data_line.strip())
        qid = data_dict["id"]
        if qid not in claude_filter_ids:
            continue
        res_data.append(data_line)
    # import pdb;pdb.set_trace()
    out_file = all_data_jsonl.replace(".jsonl","_requiredcode_claude.jsonl")
    with open(out_file, "w") as f:
        for line in res_data:
            f.write(line)

if __name__=="__main__":
    path_parquet = "/mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_4/nathanchan/projects/retool/data/TIGER-Lab/ViRL39K/39Krelease.parquet"
    need_cols = ['question', 'answer','qid','image']
    save_jsonl = "/mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_4/nathanchan/projects/retool/data/MM_retool_sft_rl_data/virlforrl/virl39k_data.jsonl"
    data_src = parse_virl_parquet(path_parquet,need_cols, save_jsonl)

    claude_filter = "/mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_4/nathanchan/projects/retool/data/MM_retool_sft_rl_data/RL_data_ruby/virl39k_code_required_ids_by_claude.json"
    filter_claude(save_jsonl, claude_filter)
    pass