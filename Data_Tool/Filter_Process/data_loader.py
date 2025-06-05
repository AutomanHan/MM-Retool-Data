from pathlib import Path
import json
import random
import os
class DataLoader_MMEureka:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = []
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
                        item["question"] = content["content"]#.replace('<', '&lt;').replace('>', '&gt;')
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



class DataLoader_MMEureka_CoT:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = []
        self.image_root = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/hancong/code/pretrain/reasoning/data/MM-Eureka-Dataset/MMPR/inspire/hdd/global_user/shaowenqi-shaowenqi/mengfanqing/OpenRLHF-InternVL/dataset/report_data"
        self._load_data()
    
    def _load_data(self):
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                
                self.data.append(item)