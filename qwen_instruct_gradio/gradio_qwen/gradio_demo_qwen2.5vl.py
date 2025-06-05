import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
#接收模型路径和ip/port作为输入
import sys

if len(sys.argv) > 1:
    model_name_in = sys.argv[1]
    ip_host = sys.argv[2]
    ip_port = int(sys.argv[3])
else:
    model_name_in = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hldy-nlp/VLIT/hancong11/models/reasoning/open_ckpt/Qwen2.5-VL-7B-Instruct"
    ip_host = "10.245.115.207"
    ip_port = 8414
    
# 加载模型
model_name = "Qwen/Qwen2.5-VL-Chat"
# model_name="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/hancong/code/pretrain/reasoning/code/open_ckpt/qwen2_5vl/Qwen2.5-VL-7B-Instruct"
model_name="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hldy-nlp/VLIT/hancong11/models/reasoning/open_ckpt/Qwen2.5-VL-7B-Instruct"
model_name="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hldy-nlp/VLIT/hancong11/models/reasoning/mm_tool_ckpt/mm_tool_sft_qwen7b_lxh/mmeureka5.5k_exp1/checkpoint-100"
model_name=model_name_in
# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

def qwen2_vl_infer(image,img_path, text):
    # 构造输入
    image_message = None
    if image:
        image_message = image
    else:
        if img_path.startswith("/dolphinfs/"):
            img_path = "/mnt"+img_path
        image_message = f"{img_path}"
    query = text if text else ""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_message,
                },
                {"type": "text", "text": f"{query}"},
            ],
        }
    ]
    # # Qwen-VL 输入格式: [(图片,文本)]
    # inputs = tokenizer.from_list_format([
    #     {'image': image},
    #     {'text': query}
    # ])
    # 推理
    # with torch.no_grad():
    #     output = model.generate(**inputs, max_new_tokens=512)
    # response = tokenizer.decode(output[0], skip_special_tokens=True)
    # return response
    try:
        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=2560)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
    except Exception as e:
        print(f"Error during inference: {e}")
        return "Error during inference"
    print(output_text)
    return output_text[0]
    

if True:
    # Gradio 界面
    gr.Interface(
        fn=qwen2_vl_infer,
        inputs=[
            gr.Image(type="pil", label="图片输入"),
            gr.Textbox(label="图片路径"),
            gr.Textbox(label="文本输入")
        ],
        outputs=gr.Textbox(label="模型回答"),
        title="Qwen2.5-VL 多模态模型在线服务"
    ).launch(server_name=ip_host, server_port=ip_port)
else: 
    data_root = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/hancong/code/pretrain/reasoning/data/MM-Eureka-Dataset"
    image_path=data_root+ "/K12/inspire/hdd/global_user/shaowenqi-shaowenqi/mengfanqing/OpenRLHF-InternVL/dataset/report_data/K12/0E62F78B27CE2CD814DAC60E1D956636.png"
    text_prompt = "You should first thinks about the reasoning process in the mind and then provides the user with the answer. Your answer must be in latex format and wrapped in $...$.The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> Since $1+1=2$, so the answer is $2$. </think><answer> $2$ </answer>, which means your output should start with <think> and end with </answer>.\nQuestion:\nAs shown in the figure, points $$A$$, $$B$$, and $$C$$ are all on circle $$O$$. If $$∠AOB + ∠ACB = 84^{\\circ}$$, then the measure of $$∠ACB$$ is ___."
    image = None
    qwen2_vl_infer(image, image_path, text_prompt)