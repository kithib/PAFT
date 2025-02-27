# Inspired by: https://github.com/huggingface/transformers/blob/v4.34.1/examples/pytorch/summarization/run_summarization.py

from typing import TYPE_CHECKING, List, Optional, Union, Dict, Tuple, Iterable, Iterator
from transformers import DataCollatorForSeq2Seq
from torch.utils.data import Dataset
from ...data import get_dataset, split_dataset
from ...extras.constants import IGNORE_INDEX
from ...extras.misc import get_logits_processor
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push
from .metric import ComputeMetrics
from .trainer import CustomSeq2SeqTrainer, PromptSeq2SeqTrainer


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments

def select_layer(model, layer_list: list) -> None:
    layer_set = set(map(str, layer_list))
    for name, param in model.named_parameters():
        if 'lora' in name and not any(layer in name for layer in layer_set):
            param.data.zero_() 

def save_line_chart(data, save_path):
    import matplotlib.pyplot as plt
    x_values = list(range(len(data)))
    plt.figure(figsize=(10, 5))
    plt.plot(x_values, data, marker='o')
    plt.title('Val loss Line Chart')
    plt.xlabel('time')
    plt.ylabel('Val loss')
    plt.grid(True)
    plt.savefig(save_path + "/val_loss.png")
    plt.close()


def freeze_specific_layers(model, layers_not_to_freeze = []):
    # 将layers_to_freeze_indices转换为字符串索引，用于匹配
    layers_to_freeze_indices = [str(index) for index in layers_not_to_freeze]
    # 遍历模型的所有模块
    for name, module in model.named_modules():
        # 检查模块是否是指定要冻结层的子模块
        if name.startswith('base_model.model.language_model.model.layers.'):
        #if name.startswith('base_model.model.model.layers.'):
        #if name.startswith("base_model.model.transformer.encoder.layers."):
            if name.split('.')[5] in layers_to_freeze_indices:
                #print(name.split('.')[4])
                continue
            # 冻结该层的所有参数
            else:
                for param in module.parameters():
                    param.requires_grad = False
    for name, module in model.named_modules():
        for param in module.parameters():
            if param.requires_grad == True:
                print(f"param {name} is train.")




def split_dataset_no_shuffle(
    dataset: Union["Dataset", "IterableDataset"], data_args: "DataArguments", training_args: "Seq2SeqTrainingArguments"
) -> Dict[str, "Dataset"]:
    if training_args.do_train:
        if data_args.val_size > 1e-6:  # Split the dataset
            if data_args.streaming:
                # No shuffling, directly take and skip
                val_set = dataset.take(int(data_args.val_size))
                train_set = dataset.skip(int(data_args.val_size))
                return {"train_dataset": train_set, "eval_dataset": val_set}
            else:
                val_size = int(data_args.val_size) if data_args.val_size > 1 else data_args.val_size
                # Set shuffle=False to avoid shuffling the dataset
                dataset = dataset.train_test_split(test_size=val_size, shuffle=False, seed=training_args.seed)
                return {"train_dataset": dataset["train"], "eval_dataset": dataset["test"]}
        else:
            if data_args.streaming:
                # No shuffling, just return the dataset as is
                pass  # No action needed, just return the dataset as is
            return {"train_dataset": dataset}
    else:  # do_eval or do_predict
        return {"eval_dataset": dataset}

def add_prompt_mask_to_dataset(dataset, prompt_ids_list_all):
    # def generate_continuous_prompt_mask(input_ids, prompt_ids):
    #     prompt_set = set(prompt_ids)
        
    #     prompt_mask = [0] * len(input_ids)
    #     for i in range(len(input_ids) - 1):
    #         if input_ids[i] in prompt_set and input_ids[i + 1] in prompt_set:
    #             prompt_mask[i] = 1
    #             prompt_mask[i + 1] = 1  # 也将下一个位置标记为 1
    #     return prompt_mask
    # # 为每一行生成 prompt_mask
    # prompt_masks = [generate_continuous_prompt_mask(input_ids, prompt_ids) 
    #                 for input_ids, prompt_ids in zip(dataset['input_ids'], prompt_ids_list_all)]

    # 将 prompt_mask 添加到 Dataset 中
    dataset = dataset.add_column('prompt', prompt_ids_list_all)
    # dataset = dataset.add_column('prompt_mask', prompt_masks)
    return dataset
            
def run_sft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    if finetuning_args.use_robust_tuning == True:
        dataset, prompt_ids_list_all = get_dataset(model_args, data_args, training_args, finetuning_args, stage="sft", **tokenizer_module)
        dataset = add_prompt_mask_to_dataset(dataset, prompt_ids_list_all[0])
    else:
        dataset = get_dataset(model_args, data_args, training_args, finetuning_args, stage="sft", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
    
    #llama3
    #select all layer 3,407,872
    #select_layer_list = [0, 2, 4, 5, 6, 8, 10, 16, 21, 26, 27, 28, 30, 31] # hellaswag 1000 1,490,944
    #select_layer_list = [1, 2, 3, 4, 8, 10, 11, 16, 30, 31] # hellaswag 2000 1,064,960
    #select_layer_list = [0, 1, 2, 3, 4, 8, 31] # hellaswag 5000 745,472
    #select_layer_list = [0, 1, 4, 10, 12, 14, 21, 24, 26, 27, 28, 29, 30, 31] # hellaswag 10000 1,490,944
    #select_layer_list = [0, 2, 3, 4, 5, 6, 14, 15, 19, 21, 23, 26, 27, 28, 29, 31] # hellaswag  1,703,936
    #select_layer_list = [0, 1, 2, 3, 4, 5, 6, 14, 19, 20, 21, 26, 27, 28, 29, 31] # hellaswag test  1,703,936
    #select_layer_list = [2, 5, 6, 8, 9, 10, 11, 12, 13, 14, 17, 19, 22, 25, 27, 28] # hellaswag random 0 1,703,936
    #select_layer_list = [1, 4, 6, 7, 8, 9, 10, 11, 13, 19, 20, 25, 26, 27, 28, 29] # hellaswag random 1 1,703,936
    #select_layer_list = [0, 2, 3, 5, 6, 8, 11, 14, 16, 20, 23, 25, 26, 27, 30, 31] # hellaswag random 2 1,703,936
    #select_layer_list = [0, 1, 2, 3, 4, 16, 25, 26, 27, 28, 29, 30, 31]# piqa 1000 1,384,448
    #select_layer_list = [0, 1, 2, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]# piqa 2000 1,490,944
    #select_layer_list = [0, 2, 3, 4, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]# piqa 5000 1,597,440
    #select_layer_list = [0, 1, 3, 4, 7, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30, 31]# piqa 10000 1,703,936
    #select_layer_list = [1, 2, 3, 4, 7, 20, 25, 26, 27, 28, 29, 30]# piqa = test 1,277,952
    #select_layer_list = [1, 3, 5, 7, 8, 9, 10, 14, 15, 19, 25, 26]# piqa random 0 1,277,952
    #select_layer_list = [0, 4, 12, 13, 15, 18, 20, 21, 22, 24, 26, 28]# piqa random 1 1,277,952
    #select_layer_list = [1, 2, 4, 6, 10, 11, 15, 17, 18, 19, 20, 23]# piqa random 2 1,277,952
    #select_layer_list =  [1, 2, 3, 4, 7, 8, 20, 23, 25, 26, 27, 28, 29, 31]# piqa test 1,277,952
    #select_layer_list = [0, 1, 2, 3, 4, 16, 21, 28, 29, 30, 31] # race  1000 1,171,456
    #select_layer_list = [0, 1, 2, 3, 4, 10, 20, 23, 27, 28, 29, 30, 31] # race  2000 1,384,448
    #select_layer_list = [1, 3, 4, 6, 9, 10, 11, 12, 14, 27, 28, 29, 30, 31] # race  5000 1,490,944
    #select_layer_list = [1, 2, 7, 13, 14, 23, 25, 26, 27, 28, 29, 31] # race  10000 1,277,952
    #select_layer_list = [0, 1, 3, 7, 8, 12, 13, 25, 27, 28, 29, 31] # race 1,277,952
    #select_layer_list = [0, 1, 2, 3, 4, 7, 9, 12, 14, 25, 26, 27, 28, 29, 31] # race test 1,277,952
    #select_layer_list = [2, 5, 6, 8, 11, 14, 15, 16, 19, 20, 26, 30] # race random 0 1,277,952
    #select_layer_list = [0, 2, 4, 5, 8, 11, 13, 18, 23, 26, 28, 30] # race random 1 ,277,952
    #select_layer_list = [0, 3, 4, 7, 8, 14, 15, 18, 19, 20, 22, 28, 31] # race random 2
    #select_layer_list = [0, 1, 2, 3, 4, 16, 20, 25, 26, 27, 28, 29, 30, 31] # winogrande 1000 1,490,944
    #select_layer_list = [0, 1, 2, 3, 4, 20, 25, 27, 30, 31] # winogrande 2000 1,064,960
    #select_layer_list = [1, 2, 3, 4, 6, 7, 8, 9, 26, 27, 30, 31] # winogrande 5000 1,277,952
    #select_layer_list = [6, 7, 9, 10, 15, 19, 20, 22, 26, 27, 30, 31] # winogrande 10000 1,277,952
    #select_layer_list = [0, 3, 20, 22, 23, 24, 25, 26, 27, 28, 29, 31] # winogrande = test 1,277,952  
    #select_layer_list = [1, 6, 8, 10, 14, 16, 17, 21, 22, 27, 28, 29] # winogrande random 0 1,277,952
    #select_layer_list = [1, 3, 12, 17, 19, 20, 22, 23, 24, 25, 28, 30] # winogrande random 1 1,277,952ß
    #select_layer_list = [0, 2, 4, 5, 7, 9, 10, 12, 15, 22, 25, 27] # winogrande random 2 1,277,952
    #select_layer_list = [0, 1, 2, 3, 4, 11, 14, 21, 27, 28, 29, 31] # commonsenseqa = test  1,277,952
    #select_layer_list = [4, 6, 7, 8, 10, 13, 16, 18, 24, 25, 28, 29] # commonsenseqa random 1,277,952
    #select_layer_list = [3, 8, 9, 12, 13, 16, 21, 22, 23, 26, 27, 29] # commonsenseqa random 1 1,277,952
    #select_layer_list = [2, 9, 10, 12, 13, 14, 18, 21, 23, 24, 26, 29] # commonsenseqa random 2 1,277,952
    #select_layer_list = [0, 1, 2, 3, 4, 5, 6, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 30, 31]  # pubmedqa 2,023,424
    #select_layer_list = [0, 1, 2, 3, 4, 27, 28, 30, 31]  # pubmedqa test 958,464
    #######################random 层数 random 选层 #######################################
    #select_layer_list = [2, 4, 11, 19, 23, 25]  # 6
    #select_layer_list = [1, 3, 4, 12, 14, 18, 20, 21, 22, 27, 29, 31]  #12 1,277,952
    #select_layer_list = [1, 2, 5, 8, 9, 10, 12, 13, 17, 18, 20, 21, 22, 23, 24, 25, 26, 30] # 18
    #select_layer_list = [0, 1, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 25, 26, 27, 28, 30, 31] # 24

    #######################固定层数########################################################
    #hellaswag
    #select_layer_list = [0, 26, 27, 28, 29, 31]  #6   638,976
    #select_layer_list = [0, 2, 3, 14, 15, 21, 23, 26, 27, 28, 29, 31]  #12   1,277,952
    #select_layer_list = [0, 1, 2, 3, 4, 5, 6, 14, 15, 19, 21, 23, 26, 27, 28, 29, 30, 31] # 18    1,916,928
    #select_layer_list = [0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 18, 19, 20, 21, 23, 24, 26, 27, 28, 29, 30, 31] # 24   2,555,904

    #piqa
    #select_layer_list = [2, 4, 26, 27, 28, 29]  #6  
    #select_layer_list = [1, 2, 3, 4, 7, 20, 25, 26, 27, 28, 29, 30]  #12 
    #select_layer_list = [0, 1, 2, 3, 4, 5, 7, 8, 19, 20, 23, 25, 26, 27, 28, 29, 30, 31] # 18
    #select_layer_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31] # 24

    #race
    #select_layer_list = [0, 7, 12, 27, 28, 29]  #6  
    #select_layer_list = [0, 1, 3, 7, 8, 12, 13, 25, 27, 28, 29, 31]  #12 
    #select_layer_list = [0, 1, 2, 3, 4, 7, 8, 9, 10, 12, 13, 15, 25, 27, 28, 29, 30, 31] # 18
    #select_layer_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 23, 24, 25, 27, 28, 29, 30, 31] # 24

    #winogrande
    #select_layer_list = [22, 23, 24, 26, 27, 28]  #6  
    #select_layer_list = [0, 3, 20, 22, 23, 24, 25, 26, 27, 28, 29, 31]  #12 
    #select_layer_list = [0, 1, 3, 5, 7, 9, 15, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31] # 18
    #select_layer_list = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] # 24


    #chatglm3
    # select all layer 1,949,696
    #select_layer_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 15] # commonsenseqa = test 765,952
    #select_layer_list = [1, 2, 3, 4, 5, 6, 7, 9, 12, 13, 16, 18] # hellaswag 835,584
    #select_layer_list = [1, 2, 3, 4, 5, 6, 7, 10, 12, 13, 16, 18, 20] # hellaswag test 905,216
    #select_layer_list = [0, 1, 2, 3, 5, 6, 7, 8, 9, 19, 21, 23, 25, 27] # piqa 974,848
    #select_layer_list = [0, 1, 2, 3, 5, 7, 9, 19, 20, 21, 23, 25, 27]# piqa test 905,216
    #select_layer_list = [0, 1, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27] # pubmedqa 1,323,008
    #select_layer_list = [0, 1, 4, 6, 9, 12, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27] # pubmedqa test 1,253,376
    #select_layer_list = [2, 6, 8, 9, 10, 11, 14, 15, 16, 17, 18, 20, 23, 26] # race 974,848
    #select_layer_list = [2, 5, 6, 8, 9, 14, 15, 16, 17, 18, 20, 23, 27] # race test 905,216 
    #select_layer_list = [0, 2, 6, 8, 9, 11, 12, 13, 16, 17, 18, 20, 25, 26] # winogrande 974,848
    #select_layer_list = [0, 2, 6, 8, 9, 12, 13, 15, 16, 18, 20, 25, 26] # winogrande test 905,216

    #Mistral
    # select all layer 3,407,872
    #select_layer_list = [0, 1, 2, 3, 4, 5, 6, 7, 14, 22, 26, 27, 30]  # hellaswag 1,490,944
    #select_layer_list = [6, 8, 14, 17, 18, 22, 23, 24, 25, 26, 27, 28, 29, 30] # piqa = test 1,703,936
    #select_layer_list = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 17, 30, 31] # race = test 1,703,936
    #select_layer_list = [0, 1, 2, 3, 4, 5, 6, 7, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] # winogrande = test 1,916,928

    #gemma 3,211,264
    #select_layer_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 15, 16, 18, 20, 23, 27]  # hellaswag 1,949,696
    #select_layer_list = [0, 1, 8, 9, 10, 12, 15, 16, 17, 20, 21, 22, 23, 24, 25, 26, 27] # piqa 1,949,696
    #select_layer_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16] # race 1,376,256
    #select_layer_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27] # winogrande  2,179,072

    #vicuna 4,194,304
    #select_layer_list = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12]  # hellaswag 1,572,864
    #select_layer_list = [1, 2, 3, 5, 7, 8, 11, 12, 13, 14, 21, 31] # piqa 1,572,864
    #select_layer_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13] # race 1,572,864
    #select_layer_list = [0, 2, 3, 4, 6, 8, 9, 12, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] # winogrande  2,621,440

    # zephyr 3,407,872 
    #select_layer_list = [1, 13, 15, 17, 18, 22, 23, 24, 25, 26, 27, 28, 30, 31] # hellaswag  1,490,944
    #select_layer_list = [2, 3, 6, 7, 14, 15, 16, 17, 22, 26, 27, 28]  # piqa 1,384,448
    #select_layer_list = [1, 2, 4, 6, 7, 9, 11, 13, 14, 17, 26, 30, 31] # race 1,376,256
    #select_layer_list =  [1, 3, 5, 6, 8, 13, 27, 28, 29, 30, 31] # winogrande  1,171,456

    #llama-7 4,194,304
    #select_layer_list = [0, 1, 2, 4, 5, 6, 8, 12, 16, 30, 31] # hellaswag  1,441,792
    #select_layer_list = [2, 12, 14, 15, 16, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]  # piqa 2,097,152
    #select_layer_list = [4, 5, 6, 7, 8, 10, 11, 23, 30, 31] # race 1,310,720
    #select_layer_list =  [0, 2, 3, 6, 7, 8, 10, 11, 13, 16, 23, 28, 29, 30, 31] # winogrande  1,966,080

    #drop-lora 
    #select_layer_list = [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] # hellaswag  1,441,792
    #select_layer_list = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]  # piqa 2,097,152
    #select_layer_list = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] # race 1,310,720
    #select_layer_list = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] # winogrande  1,966,080

    #yi
    #select_layer_list = [0, 1, 2, 3, 4, 6, 8, 9, 10, 19, 20, 21, 22] # hellaswag  1,331,200
    #select_layer_list = [1, 2, 3, 5, 6, 7, 8, 9, 12, 13, 15, 16, 17, 18, 20, 23]  # piqa 1,638,400
    #select_layer_list = [1, 3, 5, 6, 7, 9, 11, 12, 13, 14, 17, 21] # race 1,228,800
    #select_layer_list = [0, 1, 2, 3, 5, 6, 7, 11, 23, 26, 27, 30, 31] # winogrande  1,331,200

    #llama2 4,194,304
    #select_layer_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 12] # hellaswag  1,310,720
    #select_layer_list = [0, 1, 2, 3, 7, 8, 11, 13, 14, 21, 24, 29, 30, 31]  # piqa 1,835,008
    #select_layer_list = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16] # race 1,835,008
    #select_layer_list = [0, 1, 3, 4, 8, 14, 15, 16, 17, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30] # winogrande  2,490,368

    #dora 4,194,304
    #select_layer_list = [0, 1, 2, 4, 5, 14, 15, 19, 20, 21, 23, 26, 27, 28, 29, 31] # hellaswag  1,785,856
    #select_layer_list = [0, 1, 2, 4, 7, 23, 24, 25, 26, 27, 28, 29, 31]  # piqa 1,451,008
    #select_layer_list = [1, 3, 4, 7, 9, 12, 14, 23, 25, 27, 28, 29, 31]  # race 1,384,448
    #select_layer_list = [0, 1, 2, 3, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31] # winogrande  1,674,240

    #rslora 4,194,304
    #select_layer_list = [0, 1, 2, 4, 6, 14, 15, 19, 20, 21, 23, 25, 26, 27, 28, 29, 31] # hellaswag  1,810,432
    #select_layer_list = [0, 1, 2, 3, 15, 20, 21, 25, 26, 27, 28, 29, 31]  # piqa 1,384,448
    #select_layer_list = [0, 1, 2, 3, 7, 8, 12, 13, 25, 26, 27, 28, 29, 31] # race 1,490,944
    #select_layer_list = [1, 2, 3, 6, 14, 15, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31] # winogrande  1,916,928

    #xuanyuan 4,194,304
    #select_layer_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 17] # hellaswag  1,703,936
    #select_layer_list = [3, 4, 7, 8, 12, 14, 16, 17, 19, 21, 23, 25, 28, 29]  # piqa 1,835,008
    #select_layer_list = [0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 17, 20, 21, 22, 25, 28, 29] # race 2,490,368
    #select_layer_list = [2, 3, 4, 8, 9, 10, 14, 15, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29] # winogrande  2,490,368

    #qwen 4,194,304
    #select_layer_list = [0, 1, 2, 3, 4, 5, 6, 7, 9, 17] # hellaswag  1,310,720
    #select_layer_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 13, 14, 15, 17]  # piqa 1,835,008
    #select_layer_list =  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] # race 1,703,936
    #select_layer_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 21, 24, 25, 27, 28, 30] # winogrande  1,966,080

    #baichuan 4,194,304
    #select_layer_list =[0, 2, 3, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18] # hellaswag  1,966,080
    #select_layer_list = [0, 1, 2, 7, 8, 10, 16, 18, 23, 26, 27, 28, 29, 30, 31]  # piqa 1,966,080
    #select_layer_list = [3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 31] # race 1,703,936
    #select_layer_list = [2, 3, 4, 5, 11, 15, 16, 26, 27, 28, 29, 30, 31] # winogrande  1,703,936

    #llava 7b all layer [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
    #select_layer_list = [0, 2, 3, 4, 5, 20, 21, 23, 24, 25, 26, 27, 28, 29, 31] #gqa
    #select_layer_list = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] #mmbench
    #select_layer_list = [1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] #mme
    #select_layer_list = [1, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] #pope
    #select_layer_list = [1, 2, 3, 6, 7, 8, 20, 21, 22, 24, 25, 26, 28] #scienceqa
    #select_layer_list = [0, 2, 3, 4, 5, 6, 7, 9, 11, 12, 13, 14, 17, 18, 19, 20, 21, 22, 23, 24] #textqa
    #select_layer_list = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 21] #vizwiz
    # select_layer_list = [0, 2, 3, 4, 6, 7, 8, 9, 10, 12, 14, 15, 17, 18, 20, 21, 24, 25] #vqav2
    # if not finetuning_args.use_flexora:
    #     print(f"select layer nums is {len(select_layer_list)}")
    #     freeze_specific_layers(model, select_layer_list) 
   
    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"  # use left-padding in generation

    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if tokenizer.padding_side == "right" else None,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
    )

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = training_args.generation_max_length or data_args.cutoff_len
    training_args.generation_num_beams = data_args.eval_num_beams or training_args.generation_num_beams
    training_args.remove_unused_columns = False if model_args.visual_inputs else training_args.remove_unused_columns
    if finetuning_args.finetuning_type == "loraprune":
        from loraprune.trainer import LoRAPruneTrainer
        trainer = LoRAPruneTrainer(
            model=model,
            args=training_args,
            finetuning_args=finetuning_args,
            data_collator=data_collator,
            ratio=finetuning_args.prune_ratio,
            callbacks=callbacks,
            compute_metrics=ComputeMetrics(tokenizer) if training_args.predict_with_generate else None,
            **tokenizer_module,
            **split_dataset(dataset, data_args, training_args),
            init_ratio=0,
            warmup_iters=0.1,
            cooldown_iters=0.1,
            prune_freq=10,
            prune_metric='lora',
        )
    elif finetuning_args.finetuning_type == "rosa":
        from peft_rosa.tuners.rosa import RosaScheduler
        trainer = CustomSeq2SeqTrainer(
            model=model,
            args=training_args,
            finetuning_args=finetuning_args,
            data_collator=data_collator,
            callbacks=RosaScheduler,
            compute_metrics=ComputeMetrics(tokenizer) if training_args.predict_with_generate else None,
            **tokenizer_module,
            **split_dataset(dataset, data_args, training_args),
        )
    elif finetuning_args.use_robust_tuning == True:
        trainer = PromptSeq2SeqTrainer(
            model=model,
            args=training_args,
            finetuning_args=finetuning_args,
            data_collator=data_collator,
            callbacks=callbacks,
            compute_metrics=ComputeMetrics(tokenizer) if training_args.predict_with_generate else None,
            **tokenizer_module,
            **split_dataset_no_shuffle(dataset, data_args, training_args),
        )
    else:
        # Initialize our Trainer
        trainer = CustomSeq2SeqTrainer(
            model=model,
            args=training_args,
            finetuning_args=finetuning_args,
            data_collator=data_collator,
            callbacks=callbacks,
            compute_metrics=ComputeMetrics(tokenizer) if training_args.predict_with_generate else None,
            **tokenizer_module,
            **split_dataset(dataset, data_args, training_args),
        )

    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict()
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    gen_kwargs["logits_processor"] = get_logits_processor()

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        # lipschitz_constants = trainer.compute_lipschitz_constants()
        # print(lipschitz_constants)
        # from collections import defaultdict
        # def compute_lipschitz_products(lipschitz_constants):
        #     # Initialize a dictionary to store the products for each layer
        #     layer_products = defaultdict(float)
            
        #     # Group Lipschitz constants by layer
        #     for key, value in lipschitz_constants.items():
        #         # Extract the layer index from the key
        #         layer_index = key.split('.')[4]  # Assuming the layer index is the 4th part of the key
        #         layer_products[layer_index] += value.item()  # Sum the values for the same layer

        #     # Calculate the product for each layer
        #     cumulative_products = {}
        #     cumulative_product = 1.0

        #     for layer_index in sorted(layer_products.keys(), key=lambda x: int(x)):
        #         # Get the Lipschitz constant for the current layer
        #         lips_value = layer_products[layer_index]

        #         # Update the cumulative product
        #         cumulative_product *= lips_value

        #         # Store the cumulative product in the dictionary
        #         cumulative_products[f'layer_{layer_index}'] = cumulative_product

        #     return cumulative_products
        # cumulative_lipschitz_products = compute_lipschitz_products(lipschitz_constants)
        # print(cumulative_lipschitz_products)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
           plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])
        if finetuning_args.use_badam:
           from ..flora import val_loss_list
           save_line_chart(val_loss_list, training_args.output_dir)
    #Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        if training_args.predict_with_generate:  # eval_loss will be wrong if predict_with_generate is enabled
            metrics.pop("eval_loss", None)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        predict_results = trainer.predict(dataset, metric_key_prefix="predict", **gen_kwargs)
        if training_args.predict_with_generate:  # predict_loss will be wrong if predict_with_generate is enabled
            predict_results.metrics.pop("predict_loss", None)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(predict_results)

    #Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
