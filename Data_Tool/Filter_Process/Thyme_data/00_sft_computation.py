import pandas as pd
import pyarrow.parquet as pq
import os

def parse_parquet(file_path: str,all_data_list, image_path):
    # df = pd.read_parquet(file_path, engine='fastparquet')
    table = pq.read_table(file_path,use_threads=False)
    df = table.to_pandas()
    # 数据有三列，分别是：image[list], question[str], response[str]
    #获取qeustion和response的列
    images, question, response = df["image"],df['question'], df['response']
    # messages,question = df['messages'], df['question']
    no_image = 0
    for image,question,response in zip(images,question,response):
        if image.size == 0:
            no_image +=1
    return len(images), no_image
    


if __name__=="__main__":
    data_list = ["computation-00000-of-00003.parquet","computation-00001-of-00003.parquet","computation-00002-of-00003.parquet",]
    computation_root = "/mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_4/nathanchan/projects/retool/data/Kwai-Keye/Thyme-SFT/data/"
    image_path = "/mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_4/nathanchan/projects/retool/data/Thyme/sft/images"
    total_num = 0
    no_image_num = 0
    all_data_list = []
    for data_file in data_list:
        print(f"processing {data_file}")
        data_file = os.path.join(computation_root,data_file)
        image_num, no_image=parse_parquet(data_file, all_data_list, image_path)
        total_num += image_num
        no_image_num += no_image
    print(f"total num:{total_num}, no image num:{no_image_num}")
