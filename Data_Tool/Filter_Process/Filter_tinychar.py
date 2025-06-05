import json
from tqdm import tqdm

def filter_tinychar(data_json, output_jsonl, filter_key=[]):
    """
    过滤数据集，保留指定的key, 默认PoT数据
    :param data_json: 输入的json文件路径
    :param output_json: 输出的json文件路径
    """
    filter_data = []
    with open(data_json, 'r', encoding='utf-8') as f:
        all_infos = json.load(f)
        for item in tqdm(all_infos, desc=f"Filtering data {filter_key}"):
            item_key = False
            for f_key in filter_key:
                if f_key in item["id"]:
                    filter_data.append(item)
                    item_key = True
                    break
            
            if not item_key:
                for conversation in item["conversations"]:
                    if conversation["from"] == "gpt":
                        if "<step>" in conversation["value"]:
                            filter_data.append(item)
                            print(item["id"])
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for item in filter_data:
            # f.write(json.dumps(item, ensure_ascii=False,indent=4) + '\n')
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


if __name__=="__main__":
    input_file = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/hancong/code/pretrain/reasoning/data/TinyChartData/train.json"
    output_file="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/hancong/code/pretrain/reasoning/data/Data_Mine/TInyChart/train.jsonl"
    filter_key=["chartqagptpot"]
    filter_tinychar(input_file, output_file, filter_key)
    pass