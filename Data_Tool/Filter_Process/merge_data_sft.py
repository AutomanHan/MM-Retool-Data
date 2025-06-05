# 将经过gemini，将cot结果穿插Python代码后的数据，合并成训练数据
import json
from tqdm import tqdm
from data_loader import DataLoader_MMEureka, DataLoader_MMEureka_CoT
import random
import os
from utils import calculate_phash

class DataLoader:
    def __init__(self, data_path, data_path_cotcode):
        self.data_path = data_path
        self.data_path_cotcode = data_path_cotcode
        self.data = []
        self.data_cotcode = []
        self.data_dict = {}
        self.image_root = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/hancong/code/pretrain/reasoning/data/MM-Eureka-Dataset/MMPR/inspire/hdd/global_user/shaowenqi-shaowenqi/mengfanqing/OpenRLHF-InternVL/dataset/report_data"
        self._load_data()
    
    def _load_data(self):
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                # 解析message字段获取图片路径
                images = []

                # for msg in json.loads(item['message'].replace("'", '"')):
                if isinstance(item['conversations'], str):
                    # 如果是字符串，直接跳过
                    continue
                for content in item['conversations']:
                    if content["role"] == "user":
                        item["question"] = content["content"]
                for image_file in item["image_urls"]:
                    # 转换路径为web访问格式
                    image = os.path.join(self.image_root, image_file)
                    web_path = image
                    images.append(web_path)
                item['images'] = images
                self.data.append(item)
                self.data_dict[item["id"]] = item
        with open(self.data_path_cotcode, 'r', encoding='utf-8') as f:
            for line in f:
                item_code = json.loads(line)
                math_id = item_code["id"]
                
                if math_id in self.data_dict:
                    # 解析message字段获取图片路径
                    images = []
                    item = self.data_dict[math_id]
                    item["cot_code"] = item_code["ans"]
                    item["model"]= item_code["model"]

                    self.data_cotcode.append(item)
system_prompt = "Solve the following problem step by step. You now have the ability to selectively write executable Python code to enhance your reasoning process. The Python code will be executed by an external sandbox, and the output (wrapped in `<interpreter>output_str</interpreter>`) can be returned to aid your reasoning and help you arrive at the final answer. The Python code should be complete scripts, including necessary imports. \nEach code snippet is wrapped with `<code>\n```python\ncode snippet\n```\n</code>`.\nThe last part of your response should be in the following format:\n<answer>\n\\boxed{{'The final answer goes here.'}}\n</answer>\n\n*user question:*\nAnswer the following Math Problem and put the answer in the format of \\boxed{{answer}}\n\n{query}\n\n\nRemember to place the final answer in the last part using the format: \n<answer>\n\\boxed{{'The final answer goes here.'}}\n</answer>"
system_prompt2 = "Solve the following problem step by step. Your answer must be in latex format and wrapped in $...$. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> Since $1+1=2$, so the answer is $2$. </think><answer> $2$ </answer>, which means your output should start with <think> and end with </answer>. You now have the ability to selectively write executable Python code to enhance your reasoning process. The Python code will be executed by an external sandbox, and the output (wrapped in `<interpreter>output_str</interpreter>`) can be returned to aid your reasoning and help you arrive at the final answer. The Python code should be complete scripts, including necessary imports. \nEach code snippet is wrapped with `<code>\n```python\ncode snippet\n```\n</code>`.\nThe last part of your response should be in the following format:\n<answer>\n\\boxed{{'The final answer goes here.'}}\n</answer>\n\n*user question:*\nAnswer the following Math Problem and put the answer in the format of \\boxed{{answer}}\n\n{query}\n\n\nRemember to place the final answer in the last part using the format: \n<answer>\n\\boxed{{'The final answer goes here.'}}\n</answer>"
system_prompt3 = "Solve the following problem step by step. Your answer must be in latex format and wrapped in $...$. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> Since $1+1=2$, so the answer is $2$. </think><answer> $2$ </answer>, which means your output should start with <think> and end with </answer>. You now have the ability to selectively write executable Python code to enhance your reasoning process. The Python code will be executed by an external sandbox, and the output (wrapped in `<interpreter>output_str</interpreter>`) can be returned to aid your reasoning and help you arrive at the final answer. The Python code should be complete scripts, including necessary imports. \nEach code snippet is wrapped with `<code>\n```python\ncode snippet\n```\n</code>`.\nThe last part of your response should be in the following format:\n<answer>\n\\boxed{{'The final answer goes here.'}}\n</answer>\n\n*user question:*\nAnswer the following Math Problem and put the answer in the format of \\boxed{{answer}}\n\n{query}\n\n\nRemember to place the final answer in the last part using the format: \n<answer>\n\\boxed{{'The final answer goes here.'}}\n</answer>. \n Example Implementation:\n\n<think>\nFirst, we need to calculate the statistical properties of the dataset.\n\n<code>\n```\nimport numpy as np\n\n# Calculate core statistics for numerical analysis\nvalues = np.array([12, 15, 18, 22])\nmean = np.mean(values)\nstd_dev = np.std(values)\nprint(f\"Statistics| Mean:{mean:.2f}, SD:{std_dev:.2f}\")\n```\n</code>\n<interpreter>\nStatistics| Mean:16.75, SD:3.59\n</interpreter>\n\nThe results suggest we should...\n</think>\n\n<answer>\nThe analysis indicates moderate variability (SD=3.59) around the mean of 16.75.\n</answer>"


keep_dataset=["geoqa_plus","K12"]

def merge_data_sft(data_meger_infos, output_data_json):
    """
    将经过gemini，将cot结果穿插Python代码后的数据，合并成训练数据
    """
    #dict_keys(['id', 'conversations', 'images', 'answer_format', 'mode', 'ds_name', 'extra', 'tag', 'answer_token', 'image_phash'])
    all_new_datas = []
    
    for item in tqdm(data_meger_infos, desc="Merging data"):
        need_skip = True
        for path_i in keep_dataset:
            if path_i in item["images"][0]:
                need_skip = False
                break 
        if need_skip:
            continue
        item_new = {}
        # 处理每个item
        item_new["id"]= "mm_eureka_"+item["id"]
        
        question_old = item["question"]
        
        conversations = []
        question = "<image> " + question_old.split("Question:")[-1]
        
        answer = item["cot_code"]
        conversations.append({"from": "system", "content": system_prompt})
        conversations.append({"from": "user", "content": question})
        conversations.append({"from": "assistant", "content": answer})
        item_new["conversations"] = conversations

        item_new["images"] = item["images"]
        item_new["answer_format"] = item["model"] + "_cot_code"

        item_new["mode"] = "mllm"
        item_new["ds_name"] = "MM-Eureka-Dataset"
        item_new["extra"] = {}

        item_new["tag"] = {}
        item_new["answer_token"] = len(item["cot_code"].split())
        item_new["image_phash"] = calculate_phash(item["images"][0])
        all_new_datas.append(item_new)

    with open(output_data_json, 'w', encoding='utf-8') as f:
        json.dump(all_new_datas, f, indent=4, ensure_ascii=False)

if __name__=="__main__":
    # 初始化数据加载器
    data_jsonl="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/hancong/code/pretrain/reasoning/data/MM-Eureka-Dataset/dataset.jsonl"
    data_cotcode_jsonl = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/hancong/code/pretrain/reasoning/data/MM-Eureka-Dataset/synth_gemini_cot_code/dataset_20k_cot_code.jsonl"

    output_path = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/hancong/code/pretrain/reasoning/data/Data_Mine/MM_ReTool/SFT_CoT_Data/data_sft_cot.json"

    # v2 prompt
    data_cotcode_jsonl="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/hancong/code/pretrain/reasoning/data/MM-Eureka-Dataset/synth_gemini_cot_code/dataset_55k_cot_code_promptv2.jsonl"
    output_path = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/hancong/code/pretrain/reasoning/data/Data_Mine/MM_ReTool/SFT_CoT_Data/data_sft_cot_promptv2_15k.json"

    # v3 prompt
    data_cotcode_jsonl="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/hancong/code/pretrain/reasoning/data/MM-Eureka-Dataset/synth_gemini_cot_code/dataset_55k_cot_code_promptv3.jsonl"
    output_path = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/hancong/code/pretrain/reasoning/data/Data_Mine/MM_ReTool/SFT_CoT_Data/data_sft_cot_promptv3_15k.json"
    output_path = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/hancong/code/pretrain/reasoning/data/Data_Mine/MM_ReTool/SFT_CoT_Data/data_sft_cot_promptv3_15k_k12_geoqaplus.json"

    data_loader = DataLoader(data_jsonl, data_cotcode_jsonl)

    data_meger_infos = data_loader.data_cotcode
    merge_data_sft(data_meger_infos, output_path)

