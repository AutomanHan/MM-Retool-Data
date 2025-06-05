import json
from tqdm import tqdm


system_prompt = "Solve the following problem step by step. You now have the ability to selectively write executable Python code to enhance your reasoning process. The Python code will be executed by an external sandbox, and the output (wrapped in `<interpreter>output_str</interpreter>`) can be returned to aid your reasoning and help you arrive at the final answer. The Python code should be complete scripts, including necessary imports. \nEach code snippet is wrapped with `<code>\n```python\ncode snippet\n```\n</code>`.\nThe last part of your response should be in the following format:\n<answer>\n\\boxed{{'The final answer goes here.'}}\n</answer>\n\n*user question:*\nAnswer the following Math Problem and put the answer in the format of \\boxed{{answer}}\n\n{query}\n\n\nRemember to place the final answer in the last part using the format: \n<answer>\n\\boxed{{'The final answer goes here.'}}\n</answer>"
system_prompt2 = "Solve the following problem step by step. Your answer must be in latex format and wrapped in $...$. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> Since $1+1=2$, so the answer is $2$. </think><answer> $2$ </answer>, which means your output should start with <think> and end with </answer>. You now have the ability to selectively write executable Python code to enhance your reasoning process. The Python code will be executed by an external sandbox, and the output (wrapped in `<interpreter>output_str</interpreter>`) can be returned to aid your reasoning and help you arrive at the final answer. The Python code should be complete scripts, including necessary imports. \nEach code snippet is wrapped with `<code>\n```python\ncode snippet\n```\n</code>`.\nThe last part of your response should be in the following format:\n<answer>\n\\boxed{{'The final answer goes here.'}}\n</answer>\n\n*user question:*\nAnswer the following Math Problem and put the answer in the format of \\boxed{{answer}}\n\n{query}\n\n\nRemember to place the final answer in the last part using the format: \n<answer>\n\\boxed{{'The final answer goes here.'}}\n</answer>"
system_prompt3 = "Solve the following problem step by step. Your answer must be in latex format and wrapped in $...$. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> Since $1+1=2$, so the answer is $2$. </think><answer> $2$ </answer>, which means your output should start with <think> and end with </answer>. You now have the ability to selectively write executable Python code to enhance your reasoning process. The Python code will be executed by an external sandbox, and the output (wrapped in `<interpreter>output_str</interpreter>`) can be returned to aid your reasoning and help you arrive at the final answer. The Python code should be complete scripts, including necessary imports. \nEach code snippet is wrapped with `<code>\n```python\ncode snippet\n```\n</code>`.\nThe last part of your response should be in the following format:\n<answer>\n\\boxed{{'The final answer goes here.'}}\n</answer>\n\n*user question:*\nAnswer the following Math Problem and put the answer in the format of \\boxed{{answer}}\n\n{query}\n\n\nRemember to place the final answer in the last part using the format: \n<answer>\n\\boxed{{'The final answer goes here.'}}\n</answer>. \n Example Implementation:\n\n<think>\nFirst, we need to calculate the statistical properties of the dataset.\n\n<code>\n```\nimport numpy as np\n\n# Calculate core statistics for numerical analysis\nvalues = np.array([12, 15, 18, 22])\nmean = np.mean(values)\nstd_dev = np.std(values)\nprint(f\"Statistics| Mean:{mean:.2f}, SD:{std_dev:.2f}\")\n```\n</code>\n<interpreter>\nStatistics| Mean:16.75, SD:3.59\n</interpreter>\n\nThe results suggest we should...\n</think>\n\n<answer>\nThe analysis indicates moderate variability (SD=3.59) around the mean of 16.75.\n</answer>"
def convert2retool_systemprompt(src_jsonl, repromt_jsonl):
    with open(src_jsonl, "r") as f:
        lines = f.readlines()
    new_data = []
    for line in tqdm(lines):
        data = json.loads(line)
        messages = json.loads(data["message"])
        for i, message in enumerate(messages):
            if message["role"] == "system":
                message["content"] = system_prompt3
        new_messages = json.dumps(messages, ensure_ascii=False)
        data["message"] = new_messages
        new_data.append(data)
        # import pdb;pdb.set_trace()
    with open(repromt_jsonl, "w",encoding="utf-8") as f:
        for data in new_data:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


if __name__=="__main__":
    src_jsonl = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv/qiuhaibo/workspace/weights/huggingface.co/datasets/FanqingM/MM-Eureka-Dataset/dataset_k12_filtered_for_qwen_instruct.jsonl" # 8099条
    repromt_jsonl = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/hancong/code/pretrain/reasoning/data/Data_Mine/MM_ReTool/RL_mmeureka_data_qhb/dataset_k12_filtered_retoolprompt_for_qwen_instruct.jsonl"
    repromt_jsonl = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/hancong/code/pretrain/reasoning/data/Data_Mine/MM_ReTool/RL_mmeureka_data_qhb/dataset_k12_filtered_retoolprompt3_for_qwen_instruct.jsonl"
    convert2retool_systemprompt(src_jsonl, repromt_jsonl)
    pass