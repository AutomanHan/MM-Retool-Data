import json
import re
from executor import PythonExecutor,excute_codes
from tqdm import tqdm
def extract_rewards(input_str):
    pattern = r'([A-Za-z]+)\s*Reward:\s*([0-9.]+)'
    pattern = r'([A-Za-z\s]*(?:[Rr]eward)[A-Za-z\s]*):\s*([\d.]+)'
    pattern = r'([A-Za-z\s]*(?:[Rr]eward)[A-Za-z\s]*?)\s*:\s*([\d.]+)'
    matches = re.findall(pattern, input_str)
    # return {match[0]: float(match[1]) for match in matches}
    return {re.sub(r'\s+', ' ', match[0].strip()): float(match[1]) for match in matches}
    # clean key
    
def parse_code(log):
    executor = PythonExecutor()
    pattern_format_all =r"(.*?<code>.*?</code>\s*?<interpreter>(?:(?!<code>).*?)*?</interpreter>.*?)+"
    pattern_code_block = r"<code>(.*?)</code>"
    #获取代码部分
    matches_format = re.search(pattern_code_block, log, re.DOTALL)
    if matches_format is None:
        return None
    matches_res = re.findall(pattern_code_block, log, re.DOTALL)
    for i, match_tmp in enumerate(matches_res, 0):
        # 提取其中所有内容
        code_exec = match_tmp.split("```python")[-1].replace("```", "").strip()
        batch_results, no_code_idx=excute_codes([code_exec],executor)
        if len(batch_results) ==0:
            continue
        output,report = batch_results[-1]
        if "output_str" in output:
            import pdb;pdb.set_trace()
def parse_interpreter(log,errs_dict):
    errs_set = set()
    # errs_dict = {}
    pattern_interpreter = r"<interpreter>(.*?)</interpreter>"
    matches_format = re.search(pattern_interpreter,log, re.DOTALL)
    if matches_format is None:
        return None,None
    matches_res = re.findall(pattern_interpreter, log, re.DOTALL)
    for i, match_tmp in enumerate(matches_res, 0):
        # 使用正则表达式进行不区分大小写的查找
        if re.search(r"error", match_tmp, re.IGNORECASE):
            errs_set.add(match_tmp)
            if match_tmp not in errs_dict:
                errs_dict[match_tmp] = 0
            errs_dict[match_tmp] += 1
    return errs_set,errs_dict
            

def parse_log(log_file, process_line,reward_parse=False):
    with open(log_file, "r") as f:
        log = f.read()
    
    # log = log_raw[process_line * -1:]
    # import pdb; pdb.set_trace()
    split_str="===============================================================\n"
    log_list_raw = log.split(split_str)
    log_list = [log for log in log_list_raw if not (log.startswith("=================") or log.startswith("-------------------------") or log=="")]
    format_right = 0
    format_code_right = 0
    acc_rewards = []
    run_code = []
    errs_set = set()
    errs_dict = {}
    print(f"log_list: {len(log_list)}")
    for log in tqdm(log_list[:1000], desc="parse log"):
        parse_code(log)
        err_set, err_dict=parse_interpreter(log,errs_dict)
        if err_set:
            errs_set.update(err_set)
            # errs_dict.update(err_dict)
        lines = log.strip().split("\n")
        if reward_parse:
            for line in lines:
                result = extract_rewards(line)
                if result != {}:
                    # import pdb;pdb.set_trace()
                    if result["Format Reward"] == 0.5:
                        format_right += 1
                    if result["code reward"] == 0.5:
                        format_code_right += 1
                    if result["Accuracy Reward"] == 0.0 and result["code reward"] ==0:
                        acc_rewards.append(log)
                    if result["Accuracy Reward"] == 1.0 and result["code reward"] ==0:
                        run_code.append(log)
    errs_dict = sorted(errs_dict.items(), key=lambda x: x[1], reverse=True)
    print(f"err set:lenth: {len(errs_set)},{errs_set}")
    for key, value in errs_dict:
        print(f"{key}: {value}")
    with open("parse_interpreter.json","w") as f:
        # json文件中缩进可读
        json.dump(errs_dict, f, indent=4)
                
    print(f"total:{len(log_list)}, format right: {format_right}, code format right: {format_code_right}, acc and code error: {len(acc_rewards)}, acc right, code error: {len(run_code)}")                    
    # import pdb;pdb.set_trace()
    pass

if __name__=="__main__":
    log_file = "/Users/nathan/projects/code/retool/log/reward.tail"
    log_file="/Users/nathan/projects/code/retool/log/reward_sample_20w_18-22-46-25.txt"
    log_file="/Users/nathan/projects/code/retool/log/reward_sample_10w_20-15-57-39.txt"
    log_file = "/mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_4/nathanchan/projects/retool/code/mm_retool_output/mimo7b_sft_onlinefilter_singlenode_retool_9_penalty0_5_virl6k_0930/reward.log"
    process_line = 100000
    parse_log(log_file,process_line)
    pass