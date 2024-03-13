import copy
import warnings
from dataclasses import dataclass
from typing import Callable, List, Optional

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList
from transformers.utils import logging

logger = logging.get_logger(__name__)


@dataclass
class GenerationConfig:
    max_length: Optional[int] = None
    top_p: Optional[float] = None
    temperature: Optional[float] = None
    do_sample: Optional[bool] = True
    repetition_penalty: Optional[float] = 1.0


@torch.inference_mode()
def generate_interactive(
    model, 
    tokenizer,
    prompt,
    generation_config: Optional[GenerationConfig] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
    additional_eos_token_id: Optional[int] = None,
    **kwargs,
):
    # 将prompt加上中括号,然后进行分词操作
    inputs = tokenizer([prompt], padding=True, return_tensors="pt")
    # 得到输入tokens的长度.这里取0,是因为[prompt]只有一个元素
    input_length = len(inputs["input_ids"][0])
    # 将模型送到GPU上
    for k, v in inputs.items():
        inputs[k] = v.cuda()
    # 获取input_ids的pytorch tensor变量
    input_ids = inputs["input_ids"]
    # 获取input_ids的批尺寸大小和长度, 通常input_ids的shape是[batch_size, sequence_length]
    batch_size, input_ids_seq_length = input_ids.shape[0], input_ids.shape[-1]
    # 如果generation_config是空的,则将generation_config设置为模型的generation_config
    if generation_config is None:
        generation_config = model.generation_config
    # 复制generation_config,防止计算梯度,是这个意思吧!其实很多时候不复制也没有关系
    generation_config = copy.deepcopy(generation_config)
    # 如果在函数中手动设置了kwargs,则更新generation_config为model_kwargs
    model_kwargs = generation_config.update(**kwargs)
    # 获取generation_config的开始token id和结束token id
    bos_token_id, eos_token_id = generation_config.bos_token_id, generation_config.eos_token_id
    # 将结束token id设置为列表格式,方便在设置token生成的结束条件
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    # 额外设置的文本生成结束token
    if additional_eos_token_id is not None:
        eos_token_id.append(additional_eos_token_id)
    # 是否有默认最大长度设置
    has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
    # 如果有默认最大长度,且没有设置最大生成长度,则用默认最大长度来限制生成token长度
    if has_default_max_length and generation_config.max_new_tokens is None:
        warnings.warn(
            f"Using `max_length`'s default ({generation_config.max_length}) to control the generation length. "
            "This behaviour is deprecated and will be removed from the config in v5 of Transformers -- we"
            " recommend using `max_new_tokens` to control the maximum length of the generation.",
            UserWarning,
        )
    # 如果有最大生成新token长度,则设置最大生成长度为Input_ids长度加上最大新生成token长度
    elif generation_config.max_new_tokens is not None:
        generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length
        if not has_default_max_length:
            logger.warn(
                f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                "Please refer to the documentation for more information. "
                "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)",
                UserWarning,
            )
    # 如果input_ids_seq_length大于或等于generation_config.max_length
    # 其实上面的代码跑起来,则下面的条件也不会出现
    if input_ids_seq_length >= generation_config.max_length:
        input_ids_string = "input_ids"
        logger.warning(
            f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to"
            f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
            " increasing `max_new_tokens`."
        )

    # 2. Set generation parameters if not already defined
    # 设置logits的平滑操作和文本生成停止操作
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

    # 基于模型内部的函数来获取logits_processor
    logits_processor = model._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_seq_length,
        encoder_input_ids=input_ids,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
    )

    # 基于模型内部的函数来获取stopping_criteria
    stopping_criteria = model._get_stopping_criteria(
        generation_config=generation_config, stopping_criteria=stopping_criteria
    )

    # 得到模型的logits_warper
    logits_warper = model._get_logits_warper(generation_config)

    # 用来记录各个批次的结束token
    unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
    # 计算位置序列,从0到sequence_length-1
    position_ids = torch.arange(0, input_ids.shape[-1]).unsqueeze(dim=0).to(input_ids.device)
    # 得分
    scores = None
    while True:
        # model_inputs = model.prepare_inputs_for_generation(input_ids, position_ids=position_ids, **model_kwargs)
        model_inputs = input_ids
        # forward pass to get next token
        outputs = model(
            model_inputs,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )

        # 得到最后一个token的logits
        next_token_logits = outputs.logits[:, -1, :]

        # pre-process distribution,调整logits的分布,在随机采样时提高一些token的上场率
        next_token_scores = logits_processor(input_ids, next_token_logits)
        next_token_scores = logits_warper(input_ids, next_token_scores)

        # sample
        probs = nn.functional.softmax(next_token_scores, dim=-1)
        # 注意这里没有实现beam search
        if generation_config.do_sample:
            # 随机采样
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            # 贪婪算法
            next_tokens = torch.argmax(probs, dim=-1)

        # update generated ids, model inputs, and length for next step
        # 将新得到token和input_ids拼接,作为新的input_ids
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        # 更新模型参数
        model_kwargs = model._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=False
        )
        # 序列生成是否结束
        unfinished_sequences = unfinished_sequences.mul((min(next_tokens != i for i in eos_token_id)).long())
        
        # 得到input_ids
        output_token_ids = input_ids[0].cpu().tolist()
        # 去掉input_ids的长度,得到生成新的token
        output_token_ids = output_token_ids[input_length:]
        # 判断是否结束,并去掉结束token id
        for each_eos_token_id in eos_token_id:
            if output_token_ids[-1] == each_eos_token_id:
                output_token_ids = output_token_ids[:-1]
        # 将token id转换为字符串
        response = tokenizer.decode(output_token_ids)

        # 流式输出
        yield response
        # stop when each sentence is finished, or if we exceed the maximum length
        if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
            break
