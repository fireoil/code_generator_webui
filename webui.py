"""
This script refers to the dialogue example of streamlit, the interactive generation code of chatglm2 and transformers.
We mainly modified part of the code logic to adapt to the generation of our model.
Please refer to these links below for more information:
    1. streamlit chat example: https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps
    2. chatglm2: https://github.com/THUDM/ChatGLM2-6B
    3. transformers: https://github.com/huggingface/transformers
"""

from dataclasses import asdict

# 使用streamlit来做界面
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging

from modelscope import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM

from interface import GenerationConfig, generate_interactive

logger = logging.get_logger(__name__)


def on_btn_click():
    del st.session_state.messages

# 模型路径,使用的是CodeFuse-DeepSeek代码生成模型
model_name_or_path = "../../../pretrained_models/CodeFuse-DeepSeek-33B-4bits"

# 缓存下面的分词器和模型
@st.cache_resource
def load_model():
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, 
                                              trust_remote_code=True, 
                                              use_fast=False,
                                              lagecy=False)
    # 使用左补齐
    tokenizer.padding_side = "left"
    # 定义pad和eos的token id
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<｜end▁of▁sentence｜>")
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("<｜end▁of▁sentence｜>")

    # 加载模型,使用gptq来加载
    model = AutoGPTQForCausalLM.from_quantized(model_name_or_path, 
                                                inject_fused_attention=False,
                                                inject_fused_mlp=False,
                                                use_safetensors=True,
                                                use_cuda_fp16=True,
                                                disable_exllama=False,
                                                device_map='auto'   # Support multi-gpus
                                              )
    return model, tokenizer


def prepare_generation_config():
    # 在侧边栏定义模型生成的参数
    with st.sidebar:
        max_length = st.slider("Max Length", min_value=32, max_value=16000, value=512)
        top_p = st.slider("Top P", 0.0, 1.0, 0.8, step=0.01)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, step=0.01)
        # 清楚历史消息,也就是st.session_state.messages.该变量是个记录对话过程的变量
        st.button("Clear Chat History", on_click=on_btn_click)

    # 根据变量,来设置generation_config
    generation_config = GenerationConfig(max_length=max_length, top_p=top_p, temperature=temperature)

    return generation_config


# 定义模板
# 格式为
# <s>human
# {user} # 这里的user是用户的问题,也就是prompt的主要内容
# <s>bot
# <robot> # 这里的robot是模型返回的结果
user_prompt = "<s>human\n{user}"
robot_prompt = "<s>bot\n{robot}"
# 最后一个回答的模板
cur_query_prompt = "<s>human\n{user}<s>bot\n"


def combine_history(prompt):
    '''
    组合历史数据.因为模型输入有个token长度限制,因此这里可以设置一个回溯多长数据的参数
    '''
    # 历史数据为st.session_state.messages,这个变量是streamlit维护的全局变量
    # messages是一个列表,逐个保存对话内容
    # 格式为{user} -> {robot} -> {user} -> {robot} -> ...
    messages = st.session_state.messages
    total_prompt = ""
    for message in messages:
        # 获取该message的内容
        cur_content = message["content"]
        if message["role"] == "user":
            # 替换{user}
            cur_prompt = user_prompt.replace("{user}", cur_content)
        elif message["role"] == "robot":
            # 替换{robot}
            cur_prompt = robot_prompt.replace("{robot}", cur_content)
        else:
            raise RuntimeError
        # 叠加字符串
        total_prompt += cur_prompt
    # 替换输入的问题
    total_prompt = total_prompt + cur_query_prompt.replace("{user}", prompt)
    # 调整格式,主要看最后一个字符是否为\n
    total_prompt = total_prompt if total_prompt.endswith('\n') else f'{total_prompt}\n'
    # 返回多轮对话
    return total_prompt


def main():
    # torch.cuda.empty_cache()
    print("load model begin.")
    # 首先加载分词器和模型
    model, tokenizer = load_model()
    print("load model end.")

    # 这个图片可以自定义
    user_avator = "imgs/user.png"
    robot_avator = "imgs/user_luoxiaohei.jpg"

    st.title("CodeFuse-DeepSeek-33B")

    # 通过侧边栏获取模型生成文本过程中需要的参数
    generation_config = prepare_generation_config()

    # Initialize chat history
    # 初始化对话列表
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message.get("avatar")):
            st.markdown(message["content"])

    # Accept user input
    # 用户输入,在主界面的最下方
    if prompt := st.chat_input("填写提示词"):
        # Display user message in chat message container
        # 展示输入文本, avatar是图标位置,用来显示对话头像
        with st.chat_message("user", avatar=user_avator):
            st.markdown(prompt)
        # 将所有的历史对话都加载里面
        real_prompt = combine_history(prompt)
        # Add user message to chat history
        # 将对话的角色信息\内容\头像,打包成字典,传入messages
        st.session_state.messages.append({"role": "user", "content": prompt, "avatar": user_avator})

        # 模型生成对话
        with st.chat_message("robot", avatar=robot_avator):
            # 初始化一个空组件
            message_placeholder = st.empty()
            # 流式生成token
            for cur_response in generate_interactive(
                model=model,
                tokenizer=tokenizer,
                prompt=real_prompt,
                # additional_eos_token_id=92542, # 103028,
                **asdict(generation_config),
            ):
                # Display robot response in chat message container
                # 显示流式文本生成返回结果
                message_placeholder.markdown(cur_response + "▌")
            # 显示最终生成结果
            message_placeholder.markdown(cur_response)
        # Add robot response to chat history
        # 记录模型生成结果
        st.session_state.messages.append({"role": "robot", "content": cur_response, "avatar": robot_avator})
        # 清理GPU缓存
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
