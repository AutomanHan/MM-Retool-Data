import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify,send_from_directory
import random
import os

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# 正确配置图片服务路由
@app.route('/dolphinfs/<path:subpath>')
def serve_dolphinfs(subpath):
    """正确映射/mnt/dolphinfs到web路径"""
    base_dir = "/mnt/dolphinfs"
    return send_from_directory(base_dir, subpath)

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
                    # import pdb;pdb.set_trace()
                    if content["role"] == "user":
                        item["question"] = content["content"].replace('<', '&lt;').replace('>', '&gt;')
                    # if content['type'] == 'image':
                    #     # 转换路径为web访问格式
                    #     web_path = content['image'].replace(
                    #         "/mnt/dolphinfs/", 
                    #         "/dolphinfs/"
                    #     )
                    #     images.append(web_path)
                for image_file in item["image_urls"]:
                    # 转换路径为web访问格式
                    image = os.path.join(self.image_root, image_file)
                    web_path = image.replace(
                        "/mnt/dolphinfs/", 
                        "/dolphinfs/"
                    )
                    images.append(web_path)
                    # images.append(content['image'])
                # images.append(content['image'])
                item['images'] = images
                item["image_path"] = images[0].replace("/dolphinfs/","/mnt/dolphinfs/")
                self.data.append(item)
                self.data_dict[item["id"]] = item
        with open(self.data_path_cotcode, 'r', encoding='utf-8') as f:
            for line in f:
                item_code = json.loads(line)
                math_id = item_code["id"]
                # import pdb;pdb.set_trace()
                if math_id in self.data_dict:
                    # 解析message字段获取图片路径
                    images = []
                    item = self.data_dict[math_id]
                    item["cot_code"] = item_code["ans"].replace('<', '&lt;').replace('>', '&gt;')
                    item["model"]= item_code["model"]

                    keep_dataset=["geoqa_plus","K12"]
                    need_skip = True
                    for path_i in keep_dataset:
                        if path_i in item["images"][0]:
                            need_skip = False
                            break 
                    if need_skip:
                        continue

                    self.data_cotcode.append(item)
    
    def get_page(self, page=0, per_page=100, random_sample=False):
        start = page * per_page
        end = start + per_page
        if random_sample:
            return random.sample(self.data, min(per_page, len(self.data)))
        return self.data[start:end]
    def get_page_cotcode(self, page=0, per_page=100, random_sample=False):
        start = page * per_page
        end = start + per_page
        if random_sample:
            return random.sample(self.data_cotcode, min(per_page, len(self.data_cotcode)))
        return self.data_cotcode[start:end]

# 初始化数据加载器
data_jsonl="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/hancong/code/pretrain/reasoning/data/MM-Eureka-Dataset/dataset.jsonl"
data_cotcode_jsonl = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/hancong/code/pretrain/reasoning/data/MM-Eureka-Dataset/synth_gemini_cot_code/dataset_20k_cot_code.jsonl"
data_cotcode_jsonl="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/hancong/code/pretrain/reasoning/data/MM-Eureka-Dataset/synth_gemini_cot_code/dataset_55k_cot_code_promptv3.jsonl"
data_loader = DataLoader(data_jsonl, data_cotcode_jsonl)

@app.route('/')
def index():
    tmplate_file="index_cotcode.html"
    return render_template(tmplate_file)

@app.route('/api/data')
def get_data():
    page = int(request.args.get('page', 0))
    per_page = int(request.args.get('per_page', 100))
    random_sample = request.args.get('random', 'false') == 'true'
    
    data = data_loader.get_page_cotcode(page, per_page, random_sample)
    return jsonify({
        'data': data,
        'total': len(data_loader.data_cotcode),
        'page': page,
        'per_page': per_page
    })

if __name__ == '__main__':
    ip_host = "10.138.131.43"
    port = 8419
    app.run(host=ip_host, port=port, debug=True)
