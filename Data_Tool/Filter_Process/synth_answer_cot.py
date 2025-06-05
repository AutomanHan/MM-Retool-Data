import json
import os
from tqdm import tqdm
from math_verify import ExprExtractionConfig, LatexExtractionConfig, StringExtractionConfig, parse, verify
from reward_func_qwen_instruct import accuracy_reward_func, format_reward_func

def gemini_flash_result(synth_gemini2_5flash_jsonl):
    response_dict = {}
    with open(synth_gemini2_5flash_jsonl, 'r', encoding='utf-8') as f:
        
        all_infos_lines = f.readlines()
        all_infos = [ json.loads(line) for line in all_infos_lines if line.strip() ]
        for item in tqdm(all_infos, desc="Processing data"):
            if "ans" in item:
                response_answer = item["ans"]
                math_id = item["id"]
                
                response_dict[math_id] = response_answer
    return response_dict
            
def extract_answer():
    pass
def verify_answer(response_content, gold_answer):
    answer_parsed = parse(
        response_content,
        extraction_config=[StringExtractionConfig(), LatexExtractionConfig(), ExprExtractionConfig()],
    )
    try:
        reward = float(verify(answer_parsed, gold_answer))
    except Exception:
        pass

    return reward

def compare_gt_response(gt_answer, response_answer):
    format_res = []
    acc_res = []
    acc_dict=[]
    for mathid, response in tqdm(response_answer.items(), desc="Comparing data"):
        verify_format, verify_acc = 0.0, 0.0
        if mathid in gt_answer:
            gt_answer_value = gt_answer[mathid]
            verify_format = format_reward_func(response)
            verify_acc,answer_response = accuracy_reward_func(response, gt_answer_value)
        else:
            print(f"Math ID not found in GT: {mathid}")
        if verify_format > 0.0:
            format_res.append(verify_format)
        if verify_acc > 0.0:
            acc_res.append(verify_acc)
            acc_dict_tmp = {
                "id": mathid,
                "gt_answer": gt_answer_value,
                "response": response,
                "verify_format": verify_format,
                "verify_acc": verify_acc,
            }
            acc_dict.append(acc_dict_tmp)

    print(f"total number of mathid: {len(response_answer)}, \
          format_res: {len(format_res)}, \
            acc_res: {len(acc_res)}")
    with open("acc_res_55k_debug.jsonl", "w") as f:
        # 每个dict以json形式 写入jsonl
        for item in acc_dict:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
def gt_value(input_jsonl):
    gt_dict = {}
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processing data"):
            item = json.loads(line)
            if "answer" in item:
                gt_dict[item["id"]] = item["answer"]
    return gt_dict


if __name__=="__main__":
    input_file= "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/hancong/code/pretrain/reasoning/data/MM-Eureka-Dataset/dataset.jsonl"
    synth_gemini2_5flash_jsonl = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/hancong/code/pretrain/reasoning/data/MM-Eureka-Dataset/synth_gemini_cot_code/dataset_cot.jsonl"

    gt_dict = gt_value(input_file)
    response_answer = gemini_flash_result(synth_gemini2_5flash_jsonl)

    compare_gt_response(gt_dict, response_answer)

    pass


