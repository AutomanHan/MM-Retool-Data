import os
import json
import ast

def process_jsonl_file(input_file, output_file):
    processed_count = 0
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                # 解析JSON行
                data = json.loads(line)
                
                # 1. 给question字段添加前缀
                if 'question' in data:
                    data['question'] = f"<image>\n{data['question']}"
                
                # 2. 处理message字段
                if 'message' in data and data['message']:
                    try:
                        # 解析message字段（它已经是字符串形式的JSON）
                        messages = ast.literal_eval(data['message'])
                        
                        # 遍历message中的每个消息
                        for msg in messages:
                            if msg.get('role') == 'user' and 'content' in msg:
                                content = msg['content']
                                
                                # 查找最后一个类型为text的content项
                                last_text_index = -1
                                for i, item in enumerate(content):
                                    if isinstance(item, dict) and item.get('type') == 'text':
                                        last_text_index = i
                                
                                # 给最后一个text项添加前缀
                                if last_text_index != -1:
                                    content[last_text_index]['text'] = f"<image>\n{content[last_text_index]['text']}"
                        
                        # 将处理后的messages重新转换为字符串
                        data['message'] = json.dumps(messages, ensure_ascii=False)
                        
                    except (SyntaxError, ValueError, TypeError) as e:
                        print(f"警告: 第{line_num}行的message字段解析失败: {e}")
                        # 如果解析失败，保持原样
                
                # 写入处理后的数据
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                processed_count += 1
                
                if line_num % 1000 == 0:
                    print(f"已处理 {line_num} 行数据...")
                    
            except json.JSONDecodeError as e:
                print(f"错误: 第{line_num}行JSON解析失败: {e}")
                continue
    
    print(f"处理完成! 共处理 {processed_count} 条记录")
    print(f"输出文件: {output_file}")

if __name__ == "__main__":
    data_path = "/mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_4/nathanchan/projects/retool/data/MM_retool_sft_rl_data/RL_data_ruby"
    data_path = "/mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_4/nathanchan/projects/retool/data/MM_retool_sft_rl_data/virlforrl/"
    file_name = "base_vl_38k_pr_required_codes_rltrain.jsonl"
    file_name = "virl39k_data_requiredcode_claude.jsonl"
    input_file = os.path.join(data_path, file_name)
    output_file = input_file.replace(".jsonl","_image.jsonl")
    
    # 使用安全版本进行处理
    process_jsonl_file(input_file, output_file)