import os
from tqdm import tqdm
import json
import pandas as pd
import re
def parse_label_tsv(tsv_file):
    # 读取TSV文件
    df = pd.read_csv(tsv_file, sep='\t')

    # 提取所需的三个字段
    # result = df[['sample_index', 'problem_index', 'index']]
    result_dict_list = [
        {
            'sample_index': row['sample_index'],
            'problem_index': row['problem_index'], 
            'index': row['index']
        }
        for _, row in df.iterrows()
    ]
    result_dict = {}
    for tmp in result_dict_list:
        sample_index = tmp['sample_index']
        result_dict[sample_index] = tmp
    return result_dict_list, result_dict

def parse_label_tsv_mathvision(tsv_file):
    # 读取TSV文件
    df = pd.read_csv(tsv_file, sep='\t')

    # 提取所需的三个字段
    # result = df[['sample_index', 'problem_index', 'index']]
    result_dict_list = [
        {
            'index': row['index'],
        }
        for _, row in df.iterrows()
    ]
    result_dict = {}
    for tmp in result_dict_list:
        sample_index = tmp['index']
        result_dict[sample_index] = tmp
    return result_dict_list, result_dict

def parse_predict_xlsx(xlsx_file):
    # 读取xlsx文件
    df = pd.read_excel(xlsx_file)

    # 方法1：提取为列表，每个元素是(sample_index, problem_index)元组
    # result_tuples = list(zip(df['sample_index'], df['problem_index']))

    # 方法2：提取为字典列表
    result_dicts = [
        {
            'sample_index': row['sample_index'],
            'problem_index': row['problem_index'],
            'prediction': row['prediction'],
            'score':row['score'],
            'log_extract':row['log_extract'],
        }
        for _, row in df.iterrows()
    ]
    pattern_interpreter = r"<interpreter>(.*?)</interpreter>"
    pattern_code_block = r"<code>(.*?)</code>"
    # matches_format = re.search(pattern_code_block, log, re.DOTALL)
    # if matches_format is None:
    use_code_count_mathverse = 0
    not_use_code_count_mathverse = 0
    for i in range(len(result_dicts)):
        predict = result_dicts[i]['prediction']
        matches_code = re.search(pattern_code_block, predict, re.DOTALL)
        matches_interpreter = re.search(pattern_interpreter, predict, re.DOTALL)
        if matches_code is not None and matches_interpreter is not None:
            result_dicts[i]['code'] = True
            use_code_count_mathverse += 1
        else:
            result_dicts[i]['code'] = False
            not_use_code_count_mathverse += 1
    print(f"matverse predict use_code_count: {use_code_count_mathverse}, not_use_code_count: {not_use_code_count_mathverse}")
    return result_dicts

def parse_predict_xlsx_mathvision(xlsx_file):
    # 读取xlsx文件
    df = pd.read_excel(xlsx_file)

    # 方法1：提取为列表，每个元素是(sample_index, problem_index)元组
    # result_tuples = list(zip(df['sample_index'], df['problem_index']))

    # 方法2：提取为字典列表
    result_dicts = [
        {
            'index': row['index'],
            'prediction': row['prediction'],
            'score':row['score'],
            # 'log_extract':row['log_extract'],
        }
        for _, row in df.iterrows()
    ]
    pattern_interpreter = r"<interpreter>(.*?)</interpreter>"
    pattern_code_block = r"<code>(.*?)</code>"
    # matches_format = re.search(pattern_code_block, log, re.DOTALL)
    # if matches_format is None:
    use_code_count_mathvision = 0
    not_use_code_count_mathvision = 0
    for i in range(len(result_dicts)):
        predict = result_dicts[i]['prediction']
        matches_code = re.search(pattern_code_block, predict, re.DOTALL)
        matches_interpreter = re.search(pattern_interpreter, predict, re.DOTALL)
        if matches_code is not None and matches_interpreter is not None:
            result_dicts[i]['code'] = True
            use_code_count_mathvision += 1
        else:
            not_use_code_count_mathvision += 1
            result_dicts[i]['code'] = False
    print(f"mathvision predict use_code_count: {use_code_count_mathvision}, not_use_code_count: {not_use_code_count_mathvision}")
    return result_dicts

def cal_statics(label_claude_dict, predict_list,dataset_name="mathverse"):
    use_code_count = 0
    not_use_code_count = 0
    index_name = "sample_index"
    if dataset_name == "mathvision":
        index_name = "index"
    # import pdb;pdb.set_trace()
    for predict in predict_list:
        sample_index = predict[index_name]
        if sample_index not in label_claude_dict:
            continue
        use_code = predict['code']
        
        if use_code:
            use_code_count += 1
        else:
            not_use_code_count += 1
    return use_code_count, not_use_code_count
if __name__=="__main__":
    label_claude_mathverse = "/mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_4/nathanchan/projects/retool/data/MM_retool_sft_rl_data/RL_data_ruby/code_yes_bmks/MathVerse_MINIVOnly_code_yes.tsv"
    label_claude_mathvision = "/mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_4/nathanchan/projects/retool/data/MM_retool_sft_rl_data/RL_data_ruby/code_yes_bmks/MathVision_code_yes.tsv"
    label_claude_mathverse_list, label_claude_mathverse_dict = parse_label_tsv(label_claude_mathverse)
    print(f"label_claude_mathverse_list: {len(label_claude_mathverse_list)}")
    predict_mathverse = "/mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_4/nathanchan/projects/retool/code/vlmeval_output/mimo7b_rl_onlinefilter_singlenode_retool_10_penalty0_5_virl6k_1005/ckpt/global_step350_hf/Qwen2.5-VL-7B-Instruct-Eureka-CKPT-ReTool/Qwen2.5-VL-7B-Instruct-Eureka-CKPT-ReTool_MathVerse_MINI_Vision_Only_qwen-plus_score.xlsx"
    predict_root = "/mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_4/nathanchan/projects/retool/code/vlmeval_output"
    ckpt_path = "/qwenvl_7b_sft30_onlinefilter_singlenode_retool_9_penalty0_5_virl6k_0914/Qwen2.5-VL-7B-Instruct-Eureka-CKPT-ReTool/Qwen2.5-VL-7B-Instruct-Eureka-CKPT-ReTool_MathVerse_MINI_Vision_Only_qwen-plus_score.xlsx"
    ckpt_path = "/qwenvl_7b_sft30_onlinefilter_singlenode_retool_9_penalty0_5_virl6k_claude_0914/Qwen2.5-VL-7B-Instruct-Eureka-CKPT-ReTool/Qwen2.5-VL-7B-Instruct-Eureka-CKPT-ReTool_MathVerse_MINI_Vision_Only_qwen-plus_score.xlsx"
    ckpt_path = "/vlmeval_results_mimovlrl/Qwen2.5-VL-7B-Instruct-Eureka-CKPT-ReTool/Qwen2.5-VL-7B-Instruct-Eureka-CKPT-ReTool_MathVerse_MINI_Vision_Only_qwen-plus_score.xlsx"
    ckpt_path = "/mimo7b_rl_onlinefilter_singlenode_retool_10_penalty0_5_virl6k_1005/ckpt/global_step550_hf/Qwen2.5-VL-7B-Instruct-Eureka-CKPT-ReTool/Qwen2.5-VL-7B-Instruct-Eureka-CKPT-ReTool_MathVerse_MINI_Vision_Only_qwen-plus_score.xlsx"
    ckpt_path = "/mimo7b_rl_onlinefilter_singlenode_retool_10_penalty0_5_virl6k_1005/ckpt/global_step350_hf/Qwen2.5-VL-7B-Instruct-Eureka-CKPT-ReTool/Qwen2.5-VL-7B-Instruct-Eureka-CKPT-ReTool_MathVerse_MINI_Vision_Only_qwen-plus_score.xlsx"
    predict_mathverse = predict_root + ckpt_path
    # predict_mathvision = "/mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_4/nathanchan/projects/retool/code/vlmeval_output/mimo7b_rl_onlinefilter_singlenode_retool_10_penalty0_5_virl6k_1005/ckpt/global_step350_hf/Qwen2.5-VL-7B-Instruct-Eureka-CKPT-ReTool/Qwen2.5-VL-7B-Instruct-Eureka-CKPT-ReTool_MathVision_qwen-plus_score.xlsx"
    predict_mathverse_list = parse_predict_xlsx(predict_mathverse)
    use_code_count, not_use_code_count = cal_statics(label_claude_mathverse_dict, predict_mathverse_list)
    print(f"use_code_count: {use_code_count}, not_use_code_count: {not_use_code_count}")

    label_claude_mathvision_list, label_claude_mathvision_dict = parse_label_tsv_mathvision(label_claude_mathvision)
    predict_mathvision = "/mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_4/nathanchan/projects/retool/code/vlmeval_output/qwenvl_7b_sft30_onlinefilter_singlenode_retool_9_penalty0_5_virl6k_0914/Qwen2.5-VL-7B-Instruct-Eureka-CKPT-ReTool/Qwen2.5-VL-7B-Instruct-Eureka-CKPT-ReTool_MathVision_qwen-plus.xlsx"
    predict_mathvision_list = parse_predict_xlsx_mathvision(predict_mathvision)
    print(f"label_claude_mathvision_list: {len(label_claude_mathvision_list)}")
    dataset_name = "mathvision"
    use_code_count_mathvision, not_use_code_count_mathvision = cal_statics(label_claude_mathvision_dict, predict_mathvision_list,dataset_name)
    print(f"use_code_count_mathvision: {use_code_count_mathvision}, not_use_code_count_mathvison: {not_use_code_count_mathvision}")



    pass