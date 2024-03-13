# <代码生成系统>项目开发文档

## 1 项目目标

实现代码生成功能

- 提供多个模型,并可切换
- 基于gradio提供web界面
- 基于lmdeploy提供web API

## 2 开发过程

首先基于basic_product_env:v1.0镜像创建一个开发容器,

该镜像在我的开发机器上,不用管.后面将打包的docker镜像发布即可.

### 2.1 创建开发环境

**更改conda和pip源**

**conda**

将以上配置文件写在`~/.condarc`中
`vim ~/.condarc`

```text
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.bfsu.edu.cn/anaconda/pkgs/main
  - https://mirrors.bfsu.edu.cn/anaconda/pkgs/r
  - https://mirrors.bfsu.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.bfsu.edu.cn/anaconda/cloud
  msys2: https://mirrors.bfsu.edu.cn/anaconda/cloud
  bioconda: https://mirrors.bfsu.edu.cn/anaconda/cloud
  menpo: https://mirrors.bfsu.edu.cn/anaconda/cloud
  pytorch: https://mirrors.bfsu.edu.cn/anaconda/cloud
  pytorch-lts: https://mirrors.bfsu.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.bfsu.edu.cn/anaconda/cloud
```

**pip**

```bash
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
```

**在base中安装**

```bash
conda install nb_conda_kernels -y
```

创建code_generator的conda env

```bash
conda create -n code_generator python=3.10 ipykernel ninja -y
```

这里的nb_conda_kernels和pykernel ninja是为了在jupyter lab环境中切换环境方便而安装的.

```bash
conda activate code_generator
```

安装pytorch和transformers环境

如果是web安装,由于是cuda11.8版本,因此

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --default-time=200000

pip install transformers --default-time=200000
```

安装langchain

```
pip install langchain --default-time=200000
```

安装auto-gptq, triton和optimum

```bash
# auto gptq源码安装
# cd AutoGPTQ
pip install -vvv -e . --default-time=200000

pip install triton --default-time=200000

pip install optimum --default-time=200000
```

安装gradio

```bash
pip install gradio --default-time=200000
```

安装modelscope

```bash
pip install modelscope --default-time=200000
```

安装vllm

```python
# Install vLLM with CUDA 11.8.
export VLLM_VERSION=0.3.3
export PYTHON_VERSION=310
pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu118-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux1_x86_64.whl **到这一步**

# 因为版本问题,需要更新
# Re-install PyTorch with CUDA 11.8.
pip uninstall torch -y
pip install torch --upgrade --index-url https://download.pytorch.org/whl/cu118

# Re-install xFormers with CUDA 11.8.
pip uninstall xformers -y
pip install --upgrade xformers --index-url https://download.pytorch.org/whl/cu118
```

安装transformers-stream-generator

```bash
pip install transformers-stream-generator --default-time=200000
```

安装bitsandbytes,

```bash
pip install bitsandbytes --default-time=200000
```

安装flash attention, 基于源码安装

```bash
# cd flash_attn, 16 is the number of thread
MAX_JOBS=16 python setup.py install
```

安装ipywidgets,如果有报警告`tqdmWarning: IProgress not found. Please update jupyter and ipywidgets`

```python
pip install ipywidgets
```

### 2.2 模型

使用`CodeFuse-DeepSeek-33B-4bits`

```python
import os
import torch
import time
from modelscope import AutoTokenizer, snapshot_download
from auto_gptq import AutoGPTQForCausalLM

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_model_tokenizer(model_path):
    """
    Load model and tokenizer based on the given model name or local path of downloaded model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, 
                                              trust_remote_code=True, 
                                              use_fast=False,
                                              lagecy=False)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<｜end▁of▁sentence｜>")
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("<｜end▁of▁sentence｜>")

    model = AutoGPTQForCausalLM.from_quantized(model_path, 
                                                inject_fused_attention=False,
                                                inject_fused_mlp=False,
                                                use_safetensors=True,
                                                use_cuda_fp16=True,
                                                disable_exllama=False,
                                                device_map='auto'   # Support multi-gpus
                                              )
    return model, tokenizer


def inference(model, tokenizer, prompt):
    """
    Uset the given model and tokenizer to generate an answer for the speicifed prompt.
    """
    st = time.time()
    prompt = prompt if prompt.endswith('\n') else f'{prompt}\n'
    inputs =  f"<s>human\n{prompt}<s>bot\n"

    input_ids = tokenizer.encode(inputs, 
                                  return_tensors="pt", 
                                  padding=True, 
                                  add_special_tokens=False).to("cuda")
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            top_p=0.95,
            temperature=0.1,
            do_sample=True,
            max_new_tokens=512,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id              
        )
    print(f'generated tokens num is {len(generated_ids[0][input_ids.size(1):])}')
    outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True) 
    print(f'generate text is {outputs[0][len(inputs): ]}')
    latency = time.time() - st
    print('latency is {} seconds'.format(latency))

    
model_dir = 'CodeFuse-DeepSeek-33B-4bits'

prompt = 'Please write a QuickSort program in Python'

model, tokenizer = load_model_tokenizer(model_dir)
inference(model, tokenizer, prompt)
```

结合`transformers_stream_generator`可以通过上面代码进行流式推理.

### 2.3 界面

界面使用streamlit进行开发,不带RAG界面

```python
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
```

## 3 开发总结和注意事项

- 拿来主义: 已有的成熟代码,有用就直接拿来使用,甚至要理解过程和原理,还是逐行代码分析这个套路
- 模型推理:理解其内部原则
- 注意模型大小和主docker镜像大小,同时要注意匹配的环境
- cuda 11.8, pytorch2.1.2(貌似这个版本比较稳定)
- 另外本项目还存在优化空间:比如使用模型推理可以使用vllm\lightllm来加速.在推理时添加RAG等
