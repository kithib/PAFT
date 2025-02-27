import inspect
import os
import sys
import re
import random
from typing import TYPE_CHECKING, Literal, Optional, Union

import numpy as np
import pandas as pd
from datasets import load_dataset, load_from_disk, Dataset

from ..extras.constants import FILEEXT2TYPE
from ..extras.logging import get_logger
from ..extras.misc import has_tokenized_data
from .aligner import align_dataset
from .data_utils import merge_dataset
from .parser import get_dataset_list
from .preprocess import get_preprocess_and_print_func
from .template import get_template_and_fix_tokenizer


if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import ProcessorMixin, Seq2SeqTrainingArguments
    from transformers.tokenization_utils import PreTrainedTokenizer

    from ..hparams import DataArguments, ModelArguments, FinetuningArguments
    from .parser import DatasetAttr


logger = get_logger(__name__)



def clean_text(text):
    # 去掉 {} 及其内容
    cleaned_text = re.sub(r'\{.*?\}', '', text)
    
    # 去掉换行符、'A.'、'B.' 和 'Answer:'
    cleaned_text = re.sub(r'\n|A\.\s*|B\.\s*|Answer:\s*', '', cleaned_text)
    
    # 去掉多余的空格
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text




# win
# def modify_prompt(all_datasets, k, excel_file_path, tokenizer):
#     # Read prompts from the Excel file
#     try:
#         df = pd.read_excel(excel_file_path)
#         random_prompts = df['text'].tolist()
#         n = 1 * len(random_prompts) // 24
#         print(f"The nums of sample data is {n}")
#         random_prompts = random.sample(random_prompts, n)
#     except Exception as e:
#         print(f"Error reading Excel file: {e}")
#         return None

#     temp_data, prompt_ids_list = [], []
#     for index, example in enumerate(all_datasets):
#         # Change the random prompt template every k examples
#         if index % k == 0:
#             random_prompt_template = random.choice(random_prompts)
            
#             random_prompt = clean_text(random_prompt_template)
#             prompt_ids = tokenizer.encode(random_prompt)
#             #print(f"random_prompt_template : {random_prompt_template}, random_prompt : {random_prompt} , prompt_ids: {prompt_ids}")
#         # Extracting parts of the prompt
#         try:
#             content = example['prompt'][0]['content']
#             prompt = content.split("Question:")[1].split("A.")[0].strip()
#             only_option1 = content.split("A.")[1].split("B.")[0].strip()
#             only_option2 = content.split("A or B.")[1].split("B.")[1].split("Answer:")[0].strip()
            
#             # Format the new prompt
#             #random_prompt_template = "Question: {prompt}\nA. {only_option1}\nB. {only_option2}\nAnswer:"
#             #random_prompt_template = "Complete the following sentence by selecting the most contextually appropriate option. Carefully consider the meaning and context of the sentence to make your choice. Question: {prompt}\nA. {only_option1}\nB. {only_option2}\nAnswer:"
#             #random_prompt_template = "Question: Choose the correct modal verb: {prompt}\nA. {only_option1}\nB. {only_option2}\nAnswer:."
#             example['prompt'][0]['content'] = random_prompt_template.format(
#                 prompt=prompt,
#                 only_option1=only_option1,
#                 only_option2=only_option2
#             )
#         except Exception as e:
#             print(f"Error processing example {index}: {e}")
#             continue  # Skip this example if there's an error

#         temp_data.append(example)
#         prompt_ids_list.append(prompt_ids)
#     return Dataset.from_list(temp_data), prompt_ids_list

#hel
# def modify_prompt(all_datasets, k, excel_file_path, tokenizer):
#     # Read prompts from the Excel file
#     try:
#         df = pd.read_excel(excel_file_path)
#         random_prompts = df['text'].tolist()
#         n = 1 * len(random_prompts) // 24
#         print(f"The nums of sample data is {n}")
#         random_prompts = random.sample(random_prompts, n)
#     except Exception as e:
#         print(f"Error reading Excel file: {e}")
#         return None

#     temp_data, prompt_ids_list = [], []
#     for index, example in enumerate(all_datasets):
#         # Change the random prompt template every k examples
#         if index % k == 0:
#             random_prompt_template = random.choice(random_prompts)
#             random_prompt = clean_text(random_prompt_template)
#             prompt_ids = tokenizer.encode(random_prompt)
#             #print(f"random_prompt_template : {random_prompt_template}, random_prompt : {random_prompt} , prompt_ids: {prompt_ids}")
#         # Extracting parts of the prompt
#         #Based on the information provided, please select the most probable conclusion: {ctx}\n A. {A}\nB. {B}\nC. {C}\nD. {D}\n Remember to consider the implications of each option. Answer:
#         try:
#             content = example['prompt'][0]['content']
#             prompt = content.split("Question:")[0].strip()
#             A = content.split("Question:")[1].split("A.")[1].split("B.")[0].strip()
#             B = content.split("B.")[1].split("C.")[0].strip()
#             C = content.split("C.")[1].split("D.")[0].strip()
#             D = content.split("D.")[1].split("You may choose from")[0].strip()
#             # Format the new prompt
#             #random_prompt_template = "Given the context {ctx}, predict the correct ending by choosing the most logical option.\n A. {A}\nB. {B}\nC. {C}\nD. {D}\n You may choose from 'A', 'B', 'C', 'D'.\n Answer:"
#             #random_prompt_template = "Given the context below, predict the most logical ending by choosing the correct option from the provided choices. Ensure your choice aligns with the context and is the most coherent conclusion. \n Context: {ctx}\n Question: Which ending makes the most sense?\n A. {A}\nB. {B}\nC. {C}\nD. {D}\n You may choose from 'A', 'B', 'C', 'D'.\n Answer:"
#             #random_prompt_template = "Based on {ctx}, which option is the most likely correct ending? Consider the overall context, character motivations, and any foreshadowing. Trick: Analyze the consistency of each option with the established details. A. {A}\nB. {B}\nC. {C}\nD. {D}\n You may choose from 'A', 'B', 'C', 'D'.\n Answer:"
#             example['prompt'][0]['content'] = random_prompt_template.format(
#                 ctx=prompt,
#                 A=A,
#                 B=B,
#                 C=C,
#                 D=D
#             )
#         except Exception as e:
#             print(f"Error processing example {index}: {e}")
#             continue  # Skip this example if there's an error

#         temp_data.append(example)
#         prompt_ids_list.append(prompt_ids)
#     return Dataset.from_list(temp_data), prompt_ids_list


#piqa
# def modify_prompt(all_datasets, k, excel_file_path, tokenizer):
#     # Read prompts from the Excel file
#     try:
#         df = pd.read_excel(excel_file_path, engine='openpyxl')
#         random_prompts = df['text'].tolist()
#         n = 1 * len(random_prompts) // 24
#         print(f"The nums of sample data is {n}")
#         random_prompts = random.sample(random_prompts, n)
#         print(len(random_prompts))
#     except Exception as e:
#         print(f"Error reading Excel file: {e}")
#         return None

#     temp_data, prompt_ids_list = [], []
#     for index, example in enumerate(all_datasets):
#         # Change the random prompt template every k examples
#         if index % k == 0:
#             random_prompt_template = random.choice(random_prompts)
#             random_prompt = clean_text(random_prompt_template)
#             prompt_ids = tokenizer.encode(random_prompt)
#             #print(f"random_prompt_template : {random_prompt_template}, random_prompt : {random_prompt} , prompt_ids: {prompt_ids}")
#         # Extracting parts of the prompt
#         try:
#             content = example['prompt'][0]['content']
#             prompt = content.split("Question:")[1].split("A.")[0].strip()
#             sol1 = content.split("A.")[1].split("B.")[0].strip()
#             sol2 = content.split("Question:")[1].split("B.")[1].split("Answer:")[0].strip()
            
#             # Format the new prompt
#             #random_prompt_template = "Use both common sense and logical reasoning to determine the correct solution for the goal: {goal}\n A. {sol1}\nB. {sol2}\n Answer format: A/B \nAnswer:"
#             #random_prompt_template = "You should use both common sense and logical reasoning to determine the most appropriate solution for the following goal. Carefully evaluate the provided options and choose the one that best aligns with the goal. Goal: {goal}\nA. {sol1}\nB. {sol2}\nAnswer:"
#             #random_prompt_template = "To solve this common sense reasoning question, consider which of the two options seems more plausible based on everyday knowledge and logic.\nQuestion: {goal}\nA. {sol1}\nB. {sol2}\nThink about the practical implications of each choice to determine the correct answer.\nAnswer:"
#             example['prompt'][0]['content'] = random_prompt_template.format(
#                 goal=prompt,
#                 sol1=sol1,
#                 sol2=sol2
#             )
#         except Exception as e:
#             print(f"Error processing example {index}: {e}")
#             continue  # Skip this example if there's an error

#         temp_data.append(example)
#         prompt_ids_list.append(prompt_ids)
#     return Dataset.from_list(temp_data), prompt_ids_list

#race
def modify_prompt(all_datasets, k, excel_file_path, tokenizer):
    # Read prompts from the Excel file
    try:
        df = pd.read_excel(excel_file_path)
        random_prompts = df['text'].tolist()
        n = 1 * len(random_prompts) // 24
        print(f"The nums of sample data is {n}")
        random_prompts = random.sample(random_prompts, n)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return None

    temp_data, prompt_ids_list = [], []
    for index, example in enumerate(all_datasets):
        # Change the random prompt template every k examples
        if index % k == 0:
            random_prompt_template = random.choice(random_prompts)
            random_prompt = clean_text(random_prompt_template)
            prompt_ids = tokenizer.encode(random_prompt)
            #print(f"random_prompt_template : {random_prompt_template}, random_prompt : {random_prompt} , prompt_ids: {prompt_ids}")
        # Extracting parts of the prompt
        #Based on the information provided, please select the most probable conclusion: {ctx}\n A. {A}\nB. {B}\nC. {C}\nD. {D}\n Remember to consider the implications of each option. Answer:
        try:
            content = example['prompt'][0]['content']
            article = content.split("Article:")[1].split("Q:")[0].strip()
            prompt = content.split("Q:")[1].split("A.")[0].strip()
            A = content.split("A.")[1].split("B.")[0].strip()
            B = content.split("B.")[1].split("C.")[0].strip()
            C = content.split("C.")[1].split("D.")[0].strip()
            D = content.split("Article:")[1].split("D.")[1].split("Answer:")[0].strip()
            # Format the new prompt
            #random_prompt_template = "Read the following article carefully: {article}. After reading, answer the question: {question}. Choose the correct option from the choices provided: \nA. {A}\nB. {B}\nC. {C}\nD. {D} \n Trick: Focus on the main idea and supporting details in the article. Output: Only the letter of the correct answer.\nAnswer:"
            #random_prompt_template = "Please read the passage carefully, focusing on the main ideas and supporting details. Answer the question that follows by choosing the best option from the choices provided. Ensure your response is based solely on the information in the passage. Output only the letter of the correct answer. Article:\n{article}\nQuestion:\n{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D} \nAnswer:"
            random_prompt_template = "A reading comprehension question is before you. Read the article and answer the question by selecting A, B, C, or D.\n\nArticle:\n{article}\n\nQ: {question}\n\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: "
            example['prompt'][0]['content'] = random_prompt_template.format(
                article=article,
                question=prompt,
                A=A,
                B=B,
                C=C,
                D=D
            )
        except Exception as e:
            print(f"Error processing example {index}: {e}")
            continue  # Skip this example if there's an error

        temp_data.append(example)
        prompt_ids_list.append(prompt_ids)
    return Dataset.from_list(temp_data), prompt_ids_list


def load_single_dataset(
    dataset_attr: "DatasetAttr",
    model_args: "ModelArguments",
    data_args: "DataArguments",
) -> Union["Dataset", "IterableDataset"]:
    logger.info("Loading dataset {}...".format(dataset_attr))
    data_path, data_name, data_dir, data_files = None, None, None, None
    if dataset_attr.load_from in ["hf_hub", "ms_hub"]:
        data_path = dataset_attr.dataset_name
        data_name = dataset_attr.subset
        data_dir = dataset_attr.folder

    elif dataset_attr.load_from == "script":
        data_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
        data_name = dataset_attr.subset
        data_dir = dataset_attr.folder

    elif dataset_attr.load_from == "file":
        data_files = []
        local_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
        if os.path.isdir(local_path):  # is directory
            for file_name in os.listdir(local_path):
                data_files.append(os.path.join(local_path, file_name))
                if data_path is None:
                    data_path = FILEEXT2TYPE.get(file_name.split(".")[-1], None)
                elif data_path != FILEEXT2TYPE.get(file_name.split(".")[-1], None):
                    raise ValueError("File types should be identical.")
        elif os.path.isfile(local_path):  # is file
            data_files.append(local_path)
            data_path = FILEEXT2TYPE.get(local_path.split(".")[-1], None)
        else:
            raise ValueError("File {} not found.".format(local_path))

        if data_path is None:
            raise ValueError("Allowed file types: {}.".format(",".join(FILEEXT2TYPE.keys())))
    else:
        raise NotImplementedError("Unknown load type: {}.".format(dataset_attr.load_from))

    if dataset_attr.load_from == "ms_hub":
        try:
            from modelscope import MsDataset
            from modelscope.utils.config_ds import MS_DATASETS_CACHE

            cache_dir = model_args.cache_dir or MS_DATASETS_CACHE
            dataset = MsDataset.load(
                dataset_name=data_path,
                subset_name=data_name,
                data_dir=data_dir,
                data_files=data_files,
                split=data_args.split,
                cache_dir=cache_dir,
                token=model_args.ms_hub_token,
                use_streaming=(data_args.streaming and (dataset_attr.load_from != "file")),
            )
            if isinstance(dataset, MsDataset):
                dataset = dataset.to_hf_dataset()
        except ImportError:
            raise ImportError("Please install modelscope via `pip install modelscope -U`")
    else:
        if "trust_remote_code" in inspect.signature(load_dataset).parameters:  # for datasets==2.16.0
            kwargs = {"trust_remote_code": True}
        else:
            kwargs = {}

        dataset = load_dataset(
            path=data_path,
            name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=data_args.split,
            cache_dir=model_args.cache_dir,
            token=model_args.hf_hub_token,
            streaming=(data_args.streaming and (dataset_attr.load_from != "file")),
            **kwargs,
        )

    if data_args.streaming and (dataset_attr.load_from == "file"):  # faster than specifying streaming=True
        dataset = dataset.to_iterable_dataset()  # TODO: add num shards parameter

    if dataset_attr.num_samples is not None and not data_args.streaming:
        target_num = dataset_attr.num_samples
        indexes = np.random.permutation(len(dataset))[:target_num]
        target_num -= len(indexes)
        if target_num > 0:
            expand_indexes = np.random.choice(len(dataset), target_num)
            indexes = np.concatenate((indexes, expand_indexes), axis=0)

        assert len(indexes) == dataset_attr.num_samples, "Sample num mismatched."
        dataset = dataset.select(indexes)
        logger.info("Sampled {} examples from dataset {}.".format(dataset_attr.num_samples, dataset_attr))

    if data_args.max_samples is not None:  # truncate dataset
        max_samples = min(data_args.max_samples, len(dataset))
        dataset = dataset.select(range(max_samples))

    return align_dataset(dataset, dataset_attr, data_args)


def get_dataset(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
) -> Union["Dataset", "IterableDataset"]:
    template = get_template_and_fix_tokenizer(tokenizer, data_args.template)
    if data_args.train_on_prompt and template.efficient_eos:
        raise ValueError("Current template does not support `train_on_prompt`.")

    # Load tokenized dataset
    if data_args.tokenized_path is not None:
        if has_tokenized_data(data_args.tokenized_path):
            logger.warning("Loading dataset from disk will ignore other data arguments.")
            dataset = load_from_disk(data_args.tokenized_path)
            logger.info("Loaded tokenized dataset from {}.".format(data_args.tokenized_path))
            if data_args.streaming:
                dataset = dataset.to_iterable_dataset()
            return dataset

        if data_args.streaming:
            raise ValueError("Turn off `streaming` when saving dataset to disk.")

    with training_args.main_process_first(desc="load dataset"):
        all_datasets = []
        for dataset_attr in get_dataset_list(data_args):
            if (stage == "rm" and dataset_attr.ranking is False) or (stage != "rm" and dataset_attr.ranking is True):
                raise ValueError("The dataset is not applicable in the current training stage.")
            all_datasets.append(load_single_dataset(dataset_attr, model_args, data_args))
            if finetuning_args.use_robust_tuning:
                excel_file_path = finetuning_args.prompt_search_space
                prompt_ids_list_all = []
                for i in range(len(all_datasets)):
                    all_datasets[i], prompt_ids_list = modify_prompt(all_datasets[i], 1, excel_file_path, tokenizer)
                    prompt_ids_list_all.append(prompt_ids_list)
                first_ten_prompts = all_datasets[0].select(range(10))['prompt']
                print(f"using robust tuning , and the first_ten_prompts is {first_ten_prompts}, and the first_ten_prompts_ids is {prompt_ids_list_all[0][0:10]}")
        dataset = merge_dataset(all_datasets, data_args, training_args)

    with training_args.main_process_first(desc="pre-process dataset"):
        preprocess_func, print_function = get_preprocess_and_print_func(
            data_args, training_args, stage, template, tokenizer, processor
        )
        column_names = list(next(iter(dataset)).keys())
        kwargs = {}
        if not data_args.streaming:
            kwargs = dict(
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=(not data_args.overwrite_cache),
                desc="Running tokenizer on dataset",
            )

        dataset = dataset.map(preprocess_func, batched=True, remove_columns=column_names, **kwargs)

        if data_args.tokenized_path is not None:
            if training_args.should_save:
                dataset.save_to_disk(data_args.tokenized_path)
                logger.info("Tokenized dataset saved at {}.".format(data_args.tokenized_path))
                logger.info("Please restart the training with `tokenized_path: {}`.".format(data_args.tokenized_path))

            sys.exit(0)

        if training_args.should_log:
            try:
                print_function(next(iter(dataset)))
            except StopIteration:
                if stage == "pt":
                    raise RuntimeError("Cannot find sufficient samples, consider increasing dataset size.")
                else:
                    raise RuntimeError("Cannot find valid samples, check `data/README.md` for the data format.")

        if finetuning_args.use_robust_tuning:
            return dataset, prompt_ids_list_all
        else:
            return dataset
