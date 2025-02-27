import json
import os
import random
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import torch
import datasets
from transformers import Seq2SeqTrainer
from transformers.utils import is_datasets_available, is_sagemaker_mp_enabled
from transformers.trainer_utils import seed_worker
from torch.utils.data import DataLoader, SequentialSampler
from functorch import vmap, jvp, jacrev, make_functional_with_buffers

from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ...extras.misc import get_logits_processor, count_parameters
#from ..trainer_utils import create_custom_optimzer, create_custom_scheduler
from ..utils import create_custom_optimzer, create_custom_scheduler

if TYPE_CHECKING:
    from transformers import ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


logger = get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""
    Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE.
    """

    def __init__(
        self, finetuning_args: "FinetuningArguments", processor: Optional["ProcessorMixin"], **kwargs
    ) -> None:
        super().__init__(**kwargs)
        print("using fine-tuning")
        self.finetuning_args = finetuning_args
        self.processor = processor
        self.gradients = {}
        self._register_hooks()
        if finetuning_args.use_flexora:
            from ..flora import clip_grad_norm_for_sparse_tensor
            # from badam import clip_grad_norm_for_sparse_tensor
            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_for_sparse_tensor, self.accelerator)


    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimzer(self.model, self.args, self.finetuning_args)
            # flora add
            if self.finetuning_args.use_flexora:
                self.optimizer.register_model_and_trainer(self.model, self)
                self.optimizer.inject_hyper_param(model_train=True)
        return super().create_optimizer()

    def save_gradient(self, layer_name):
        def hook(grad):
            self.gradients[layer_name] = grad
            #print(f"{layer_name}'s grad : {grad}")
        return hook

    def _register_hooks(self):
        # Register hooks to save gradients
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.register_hook(self.save_gradient(name))


    def compute_lipschitz_constants(self, epsilon=1e-5, num_samples=10):
        lipschitz_constants = {}
        original_params = {name: param.clone() for name, param in self.model.named_parameters()}

        # Ensure model is in training mode
        self.model.train()

        # Initialize accumulators for Lipschitz constants
        lipschitz_sums = {name: 0.0 for name in original_params}

        train_dataloader = self.get_train_dataloader()
        count = 0
        for step, inputs in enumerate(train_dataloader):
            # Zero out gradients
            self.model.zero_grad()

            # Compute original gradients
            self.training_step(self.model, inputs)
            original_gradients = {name: param.grad.clone() for name, param in self.model.named_parameters() if param.grad is not None}

            # Apply random perturbation
            with torch.no_grad():
                for param in self.model.parameters():
                    param.add_(torch.randn_like(param) * epsilon)

            # Zero out gradients again
            self.model.zero_grad()

            # Compute perturbed gradients
            self.training_step(self.model, inputs)
            perturbed_gradients = {name: param.grad.clone() for name, param in self.model.named_parameters() if param.grad is not None}

            # Compute Lipschitz constants for each layer
            for name in original_gradients:
                if name in perturbed_gradients:  # Ensure we only compute for parameters with gradients
                    grad_diff = perturbed_gradients[name] - original_gradients[name]
                    param_diff = self.model.state_dict()[name] - original_params[name]

                    # Avoid division by zero
                    if torch.norm(param_diff) != 0:
                        lipschitz_sums[name] += torch.norm(grad_diff) / torch.norm(param_diff)
                    else:
                        lipschitz_sums[name] += float('inf')  # or some large value

            # Restore original parameters
            self.model.load_state_dict(original_params)
            count += 1
            if count == num_samples:
                break

        # Compute average Lipschitz constants
        lipschitz_constants = {name: lipschitz_sums[name] / num_samples for name in lipschitz_sums if lipschitz_sums[name] > 0}

        return lipschitz_constants

    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    def _save(self, output_dir: Optional[str] = None, state_dict: Optional[Dict[str, "torch.Tensor"]] = None) -> None:
        super()._save(output_dir, state_dict)
        if self.processor is not None:
            output_dir = output_dir if output_dir is not None else self.args.output_dir
            getattr(self.processor, "image_processor").save_pretrained(output_dir)

    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        r"""
        Removes the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        labels = inputs["labels"].detach().clone() if "labels" in inputs else None  # backup labels
        if self.args.predict_with_generate:
            assert self.tokenizer.padding_side == "left", "This method only accepts left-padded tensor."
            prompt_len, label_len = inputs["input_ids"].size(-1), inputs["labels"].size(-1)
            if prompt_len > label_len:
                inputs["labels"] = self._pad_tensors_to_target_len(inputs["labels"], inputs["input_ids"])
            if label_len > prompt_len:  # truncate the labels instead of padding the inputs (llama2 fp16 compatibility)
                inputs["labels"] = inputs["labels"][:, :prompt_len]

        loss, generated_tokens, _ = super().prediction_step(  # ignore the returned labels (may be truncated)
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, :prompt_len] = self.tokenizer.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def _pad_tensors_to_target_len(self, src_tensor: torch.Tensor, tgt_tensor: torch.Tensor) -> torch.Tensor:
        r"""
        Pads the tensor to the same length as the target tensor.
        """
        assert self.tokenizer.pad_token_id is not None, "Pad token is required."
        padded_tensor = self.tokenizer.pad_token_id * torch.ones_like(tgt_tensor)
        padded_tensor[:, -src_tensor.shape[-1] :] = src_tensor  # adopt left-padding
        return padded_tensor.contiguous()  # in contiguous memory

    def save_predictions(self, predict_results: "PredictionOutput") -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.tokenizer.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX, predict_results.predictions, self.tokenizer.pad_token_id
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.tokenizer.pad_token_id)[0]
            if len(pad_len):
                preds[i] = np.concatenate(
                    (preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1
                )  # move pad token to last

        decoded_labels = self.tokenizer.batch_decode(
            labels, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: List[str] = []
            for label, pred in zip(decoded_labels, decoded_preds):
                res.append(json.dumps({"label": label, "predict": pred}, ensure_ascii=False))
            writer.write("\n".join(res))
            
    def get_tfms_metric(self): # get training-free model selection metrics
        self.accelerator.free_memory()
        train_dataloader = self.get_train_dataloader()
        model = self._wrap_model(self.model_wrapped)
        # logger.info(f"======================trainable params====================")
        # for k, v in model.named_parameters():
        #     if v.requires_grad:
        #         logger.info(f"{k}")
        # trainable_params, all_param = count_parameters(model)
        
        num_samples = 10
        norms, grads = [], []
        for step, inputs in enumerate(train_dataloader):
            model.zero_grad()
            loss = self.training_step(model, inputs)
            
            tmp = [ # get sum of grad ** 2 and number of elements in grad
                [torch.sum(p.grad ** 2), p.grad.numel()] \
                    for p in model.parameters() \
                        if (p.grad is not None) and (not torch.isnan(p.grad).any())
            ]
            norm = torch.sqrt(sum([v[0] for v in tmp]) / sum([v[1] for v in tmp]))
            
            if torch.isnan(norm).item():
                logger.info(f"step: {step}, tmp: {tmp}")
                continue
            norms.append(norm.item())
            logger.info(f"step: {step}, norm: {norm.item()}")
            # for condition number
            grads.append(torch.cat([
                torch.nan_to_num(p.grad.clone(), nan=0.0).view(-1) \
                    for p in model.parameters() \
                        if (p.grad is not None)
            ]))

            if len(norms) >= num_samples:
                break
        
        grads = torch.stack(grads, dim=0)
        eigenvalues, _ = torch.sort(torch.linalg.eigvals(grads @ grads.t()).abs())
        cond_num = np.nan_to_num((eigenvalues[-1] / eigenvalues[0]).item(), copy=True, nan=100000.0)
        metrics = [sum(norms) / len(norms), cond_num]
        
        logger.info(f"metrics: {metrics}")
        return

class PromptSeq2SeqTrainer(Seq2SeqTrainer):
    r"""
    Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE.
    """

    def __init__(
        self, finetuning_args: "FinetuningArguments", processor: Optional["ProcessorMixin"], **kwargs
    ) -> None:
        super().__init__(**kwargs)
        print("using robust prompt tuning")
        self.finetuning_args = finetuning_args
        self.processor = processor

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`] in a sequential manner.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        
        data_collator = self.data_collator

        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")
        print(train_dataset)

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "drop_last": self.args.dataloader_drop_last,
            "worker_init_fn": seed_worker,
            "prefetch_factor": self.args.dataloader_prefetch_factor,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            # 使用 SequentialSampler 以顺序读取数据
            dataloader_params["sampler"] = SequentialSampler(train_dataset)

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def generate_continuous_prompt_mask(self, input_ids, prompt_ids):
        prompt_set = set(prompt_ids)
        prompt_mask = [0] * len(input_ids)
        for i in range(len(input_ids) - 1):
            if input_ids[i] in prompt_set and input_ids[i + 1] in prompt_set:
                prompt_mask[i] = 1
                prompt_mask[i + 1] = 1  # 也将下一个位置标记为 1
        # print(len(input_ids), len(prompt_mask))
        return prompt_mask
    # def training_step(self, model, inputs, num_items_in_batch=None):
    #     decoded_output = self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
    #     print("Decoded Output:", decoded_output) 
    #     return super().training_step(model, inputs)
    ##################### hook ######################
    # def training_step(self, model, inputs, num_items_in_batch=None):
    #     # Prepare inputs
    #     m, n, yita, loss_all = 1, 8 , 1e-2, 0
    #     self.gradients = None  # 将 gradients 设为实例变量
    #     self.prompt_mask = []
    #     self.last_update_value = None
    #     def backward_hook(module, grad_input, grad_output):
    #         print("Entering backward hook")
    #         print(f"Gradient of model layer: {grad_input[0]}")
    #         self.gradients = grad_input[0]  # 更新实例变量

    #     def forward_hook(module, input, output):
    #         if self.gradients is not None:
    #             mask_expanded = torch.tensor(self.prompt_mask).unsqueeze(-1).to(self.args.device)  # 确保在正确的设备上
    #             update_value = self.gradients * mask_expanded * yita 
                
    #             # 检查 update_value 是否包含 NaN
    #             if torch.isnan(update_value).any():
    #                 print("Update value contains NaN, keeping original output.")
    #                 modified_output = output + self.last_update_value if self.last_update_value is not None else output # TODO ，如果存在 NaN，保持保持为上一个输出
    #             else:
    #                 modified_output = output + update_value  # 更新embedding的输出
    #                 self.last_update_value = update_value
    #             print(f"ori_output , update_value, modified_output : {output}, {update_value}, {modified_output}")
    #         else:
    #             modified_output = output  # 如果没有梯度，保持原输出
    #             print(f"None grad , ori_output , modified_output : {output}, {modified_output}")
        
    #         return modified_output
    #     new_input = {
    #         'input_ids': inputs['input_ids'].to(self.args.device),
    #         'attention_mask': inputs['attention_mask'].to(self.args.device),
    #         'labels': inputs['labels'].to(self.args.device)
    #     }
    #     prompt_ids = inputs['prompt']
    #     input_ids = inputs['input_ids']
    #     self.prompt_mask = self.generate_continuous_prompt_mask(input_ids[0].tolist(), prompt_ids[0].tolist())

    #     inputs = self._prepare_inputs(new_input)
    #     for i in range(m):
    #         print("strat outer training!")
    #         print(f"input ids : {input_ids.shape}")
    #         print(f"prompt ids : {prompt_ids.shape}")
    #         print(f"prompt_mask : {len(self.prompt_mask)}, {self.prompt_mask}")
    #         embed_layer = model.get_input_embeddings()
    #         embed_token = embed_layer(input_ids)
    #         first_decoder_layer = model.base_model.model.model.layers[0]
    #         backward_hook_handle = first_decoder_layer.register_full_backward_hook(backward_hook)
    #         forward_hook_handle = embed_layer.register_forward_hook(forward_hook)
    #         # Compute loss
    #         with self.compute_loss_context_manager():
    #             loss = self.compute_loss(model, inputs)
    #         # Backpropagation
    #         self.accelerator.backward(loss)
    #         print(f"loss : {loss.detach()}")
    #         for j in range(n):
    #             # Compute loss
    #             print("strat inner training!")
    #             print(f"round : m = {i} , n = {j}")
    #             with self.compute_loss_context_manager():
    #                 rubost_loss = self.compute_loss(model, inputs)
    #             # Backpropagation
    #             self.accelerator.backward(rubost_loss)
    #             print(f"rubost_loss : {rubost_loss.detach()}")
                

    #         # Unregister the hook
    #         backward_hook_handle.remove()
    #         forward_hook_handle.remove()  # 也注销前向钩子
    #     # Return loss
    #     if num_items_in_batch is None:
    #         return loss.detach() / self.args.gradient_accumulation_steps
    #     return loss.detach()


    ############################ embedding 和 模型参数 分开更新 ####################
    # def training_step(self, model, inputs, num_items_in_batch=None):
    #     # Hyperparameters
    #     m, n = 4, 4  # m: adversarial steps, n: training steps
    #     epsilon = 1e-2  # Maximum perturbation size
    #     alpha = epsilon / m  # Step size for adversarial updates
        
    #     # Prepare inputs
    #     device = self.args.device
    #     new_inputs = {
    #         'input_ids': inputs['input_ids'].to(self.args.device),
    #         'attention_mask': inputs['attention_mask'].to(self.args.device),
    #         'labels': inputs['labels'].to(self.args.device)
    #     }

    #     prompt_ids = inputs['prompt'] # Get prompt ids
    #     input_ids = inputs['input_ids'] # Get input ids
    #     self.prompt_mask = self.generate_continuous_prompt_mask(input_ids[0].tolist(), prompt_ids[0].tolist()) # Get prompt mask ids

    #     inputs = self._prepare_inputs(new_inputs)
        
    #     # Get embedding layer
    #     embed_layer = model.get_input_embeddings()
    #     # Get initial embeddings
    #     embeddings = embed_layer(inputs['input_ids']).detach()
    #     embeddings.requires_grad = True
    #     embedding_size = embeddings.size(-1)  # Get embeddings size
    #     # Convert prompt mask to tensor once
    #     prompt_mask_tensor = torch.tensor(self.prompt_mask).to(device)
    #     mask_expanded = prompt_mask_tensor.unsqueeze(0).unsqueeze(-1).expand(-1, -1, embedding_size)

    #     # Outer loop: Generate adversarial examples
    #     for i in range(m):
    #         model_inputs = {**inputs, 'inputs_embeds': embeddings}
    #         model_inputs.pop('input_ids', None)
    #         # Forward pass with current embeddings
    #         with self.compute_loss_context_manager():
    #             loss = self.compute_loss(model, model_inputs)
    #         # Compute gradients w.r.t embeddings
    #         self.accelerator.backward(loss, retain_graph=True)

    #         # print(f"embeddings (non-zero rows): {embeddings[:, prompt_mask_tensor.bool(), :]}")
    #         # print(f"embeddings.grad (non-zero rows): {embeddings.grad[:, prompt_mask_tensor.bool(), :]}")
            
    #         # Update embeddings to maximize loss (gradient ascent)
    #         if embeddings.grad is not None:
    #             perturb = alpha * embeddings.grad * mask_expanded
    #             perturb[mask_expanded == 0] = 0

    #             # Handle NaNs in perturbations
    #             nan_mask = torch.isnan(perturb)
    #             # Convert epsilon to a tensor
    #             epsilon_tensor = torch.tensor(epsilon, device=self.args.device)
    #             if nan_mask.any():
    #                 # Randomly generate a tensor of the same shape as nan_mask, with values ​​of 0 or 1
    #                 random_signs = torch.randint(0, 2, nan_mask.shape, device=self.args.device) * 2 - 1  # Generate -1 or 1
    #                 perturb[nan_mask] = random_signs[nan_mask] * epsilon_tensor.half()  # Replace with positive and negative epsilon
    #             embeddings = embeddings + perturb
    #             # Project perturbations to epsilon-ball
    #             delta = torch.clamp(embedding
    # s - embed_layer(inputs['input_ids']), -epsilon, epsilon)
    #             # print(f"perturb (non-zero rows): {perturb[:, torch.tensor(self.prompt_mask).bool(), :]}")
    #             # print(f"delta (non-zero rows): {delta[:, torch.tensor(self.prompt_mask).bool(), :]}")
    #             embeddings = embed_layer(inputs['input_ids']) + delta # update embeddings
    #             embeddings = embeddings.detach()
    #             #print(f"embeddings (non-zero rows): {embeddings[:, torch.tensor(self.prompt_mask).bool(), :]}")
    #             embeddings.requires_grad = True
    #             #embeddings.grad.fill_(0)
    #             #print(embeddings.grad)
    #         print(f"Adversarial step {i+1}/{m}, Loss: {loss.item()}")
        
    #     # Inner loop: Standard training with adversarial examples
    #     total_loss = 0
    #     embeddings.requires_grad = False
    #     # previous_params = {name: param.data.clone() for name, param in model.named_parameters()}
    #     for j in range(n):
    #         model_inputs = {**inputs, 'inputs_embeds': embeddings}
    #         model_inputs.pop('input_ids', None)
    #         # Forward pass with adversarial embeddings
    #         self.optimizer.zero_grad()
    #         with self.compute_loss_context_manager():
    #             loss = self.compute_loss(model, model_inputs)
            
    #         # Standard backward pass to update model parameters
    #         self.accelerator.backward(loss, retain_graph=True)
    #         if not j == n - 1:
    #             self.optimizer.step()
    #         # 打印每个参数的梯度
    #         # 检查参数是否更新
    #         # for name, param in model.named_parameters():
    #         #     if not torch.equal(previous_params[name], param.data):
    #         #         print(f"Parameter '{name}' has been updated.")
    #         #     # 更新上一个参数值
    #         #     previous_params[name] = param.data.clone()
    #         total_loss += loss.detach()
    #         # print(f"Current Learning Rate: {self.optimizer.param_groups[0]['lr']}")
    #         print(f"Training step {j+1}/{n}, Loss: {loss.item()}")
        
    #     # Return average loss
    #     avg_loss = total_loss / n
    #     if num_items_in_batch is None:
    #         return avg_loss / self.args.gradient_accumulation_steps
    #     return avg_loss


    ############################ 更新一次embedding立即多次更新模型参数 ####################            
    # def training_step(self, model, inputs, num_items_in_batch=None):
    #     # Hyperparameters
    #     m, n = 4, 4  # m: adversarial steps, n: training steps
    #     epsilon = 1e-2  # Maximum perturbation size
    #     alpha = epsilon / m  # Step size for adversarial updates
        
    #     # Prepare inputs
    #     device = self.args.device
    #     new_inputs = {
    #         'input_ids': inputs['input_ids'].to(self.args.device),
    #         'attention_mask': inputs['attention_mask'].to(self.args.device),
    #         'labels': inputs['labels'].to(self.args.device)
    #     }

    #     prompt_ids = inputs['prompt'] # Get prompt ids
    #     input_ids = inputs['input_ids'] # Get input ids
    #     self.prompt_mask = self.generate_continuous_prompt_mask(input_ids[0].tolist(), prompt_ids[0].tolist()) # Get prompt mask ids

    #     inputs = self._prepare_inputs(new_inputs)
        
    #     # Get embedding layer
    #     embed_layer = model.get_input_embeddings()
    #     # Get initial embeddings
    #     embeddings = embed_layer(inputs['input_ids']).detach()
    #     embeddings.requires_grad = True
    #     embedding_size = embeddings.size(-1)  # Get embeddings size
    #     # Convert prompt mask to tensor once
    #     prompt_mask_tensor = torch.tensor(self.prompt_mask).to(device)
    #     mask_expanded = prompt_mask_tensor.unsqueeze(0).unsqueeze(-1).expand(-1, -1, embedding_size)

    #     # Outer loop: Generate adversarial examples
    #     for i in range(m):
    #         model_inputs = {**inputs, 'inputs_embeds': embeddings}
    #         model_inputs.pop('input_ids', None)
    #         # Forward pass with current embeddings
    #         with self.compute_loss_context_manager():
    #             loss = self.compute_loss(model, model_inputs)
    #         # Compute gradients w.r.t embeddings
    #         self.accelerator.backward(loss, retain_graph=True)

    #         # print(f"embeddings (non-zero rows): {embeddings[:, prompt_mask_tensor.bool(), :]}")
    #         # print(f"embeddings.grad (non-zero rows): {embeddings.grad[:, prompt_mask_tensor.bool(), :]}")
            
    #         # Update embeddings to maximize loss (gradient ascent)
    #         if embeddings.grad is not None:
    #             perturb = alpha * embeddings.grad * mask_expanded
    #             perturb[mask_expanded == 0] = 0

    #             # Handle NaNs in perturbations
    #             nan_mask = torch.isnan(perturb)
    #             # Convert epsilon to a tensor
    #             epsilon_tensor = torch.tensor(epsilon, device=self.args.device)
    #             if nan_mask.any():
    #                 # Randomly generate a tensor of the same shape as nan_mask, with values ​​of 0 or 1
    #                 random_signs = torch.randint(0, 2, nan_mask.shape, device=self.args.device) * 2 - 1  # Generate -1 or 1
    #                 perturb[nan_mask] = random_signs[nan_mask] * epsilon_tensor.half()  # Replace with positive and negative epsilon
    #             embeddings = embeddings + perturb
    #             # Project perturbations to epsilon-ball
    #             delta = torch.clamp(embeddings - embed_layer(inputs['input_ids']), -epsilon, epsilon)
    #             # print(f"perturb (non-zero rows): {perturb[:, torch.tensor(self.prompt_mask).bool(), :]}")
    #             # print(f"delta (non-zero rows): {delta[:, torch.tensor(self.prompt_mask).bool(), :]}")
    #             embeddings = embed_layer(inputs['input_ids']) + delta # update embeddings
    #             embeddings = embeddings.detach()
    #             #print(f"embeddings (non-zero rows): {embeddings[:, torch.tensor(self.prompt_mask).bool(), :]}")
    #             embeddings.requires_grad = True
    #             #embeddings.grad.fill_(0)
    #             #print(embeddings.grad)
    #             print(f"Adversarial step {i+1}/{m}, Loss: {loss.item()}")
        
    #             # Inner loop: Standard training with adversarial examples
    #             total_loss = 0
    #             embeddings.requires_grad = False
    #             # previous_params = {name: param.data.clone() for name, param in model.named_parameters()}
    #             for j in range(n):
    #                 model_inputs = {**inputs, 'inputs_embeds': embeddings}
    #                 model_inputs.pop('input_ids', None)
    #                 # Forward pass with adversarial embeddings
    #                 self.optimizer.zero_grad()
    #                 with self.compute_loss_context_manager():
    #                     loss = self.compute_loss(model, model_inputs)
                    
    #                 # Standard backward pass to update model parameters
    #                 self.accelerator.backward(loss, retain_graph=True)
    #                 if not j == n - 1:
    #                     self.optimizer.step()
    #                 # 打印每个参数的梯度
    #                 # 检查参数是否更新
    #                 # for name, param in model.named_parameters():
    #                 #     if not torch.equal(previous_params[name], param.data):
    #                 #         print(f"Parameter '{name}' has been updated.")
    #                 #     # 更新上一个参数值
    #                 #     previous_params[name] = param.data.clone()
    #                 total_loss += loss.detach()
    #                 # print(f"Current Learning Rate: {self.optimizer.param_groups[0]['lr']}")
    #                 print(f"Training step {j+1}/{n}, Loss: {loss.item()}")
        
    #     # Return average loss
    #     avg_loss = total_loss / n
    #     if num_items_in_batch is None:
    #         return avg_loss / self.args.gradient_accumulation_steps
    #     return avg_loss


    ############################ 存储embedding 同时进行更新 ####################      
    # def training_step(self, model, inputs, num_items_in_batch=None):
    #     # Hyperparameters
    #     m, n = 4, 4  # m: adversarial steps, n: training steps
    #     epsilon = 1e-2  # Maximum perturbation size
    #     alpha = epsilon / m  # Step size for adversarial updates
        
    #     # Prepare inputs
    #     device = self.args.device
    #     new_inputs = {
    #         'input_ids': inputs['input_ids'].to(self.args.device),
    #         'attention_mask': inputs['attention_mask'].to(self.args.device),
    #         'labels': inputs['labels'].to(self.args.device)
    #     }

    #     prompt_ids = inputs['prompt'] # Get prompt ids
    #     input_ids = inputs['input_ids'] # Get input ids
    #     self.prompt_mask = self.generate_continuous_prompt_mask(input_ids[0].tolist(), prompt_ids[0].tolist()) # Get prompt mask ids

    #     inputs = self._prepare_inputs(new_inputs)
        
    #     # Get embedding layer
    #     embed_layer = model.get_input_embeddings()
    #     # Get initial embeddings
    #     embeddings = embed_layer(inputs['input_ids']).detach()
    #     embeddings.requires_grad = True
    #     embedding_size = embeddings.size(-1)  # Get embeddings size
    #     # Convert prompt mask to tensor once
    #     prompt_mask_tensor = torch.tensor(self.prompt_mask).to(device)
    #     mask_expanded = prompt_mask_tensor.unsqueeze(0).unsqueeze(-1).expand(-1, -1, embedding_size)
    #     generated_embedding = [embeddings]

    #     # Outer loop: Generate adversarial examples
    #     for i in range(m):
    #         model_inputs = {**inputs, 'inputs_embeds': embeddings}
    #         model_inputs.pop('input_ids', None)
    #         # Forward pass with current embeddings
    #         with self.compute_loss_context_manager():
    #             loss = self.compute_loss(model, model_inputs)
    #         # Compute gradients w.r.t embeddings
    #         self.accelerator.backward(loss, retain_graph=True)

    #         # print(f"embeddings (non-zero rows): {embeddings[:, prompt_mask_tensor.bool(), :]}")
    #         # print(f"embeddings.grad (non-zero rows): {embeddings.grad[:, prompt_mask_tensor.bool(), :]}")
            
    #         # Update embeddings to maximize loss (gradient ascent)
    #         if embeddings.grad is not None:
    #             perturb = alpha * embeddings.grad * mask_expanded
    #             perturb[mask_expanded == 0] = 0

    #             # Handle NaNs in perturbations
    #             nan_mask = torch.isnan(perturb)
    #             # Convert epsilon to a tensor
    #             epsilon_tensor = torch.tensor(epsilon, device=self.args.device)
    #             if nan_mask.any():
    #                 # Randomly generate a tensor of the same shape as nan_mask, with values ​​of 0 or 1
    #                 random_signs = torch.randint(0, 2, nan_mask.shape, device=self.args.device) * 2 - 1  # Generate -1 or 1
    #                 perturb[nan_mask] = random_signs[nan_mask] * epsilon_tensor.half()  # Replace with positive and negative epsilon
    #             embeddings = embeddings + perturb
    #             # Project perturbations to epsilon-ball
    #             delta = torch.clamp(embeddings - embed_layer(inputs['input_ids']), -epsilon, epsilon)
    #             # print(f"perturb (non-zero rows): {perturb[:, torch.tensor(self.prompt_mask).bool(), :]}")
    #             # print(f"delta (non-zero rows): {delta[:, torch.tensor(self.prompt_mask).bool(), :]}")
    #             embeddings = embed_layer(inputs['input_ids']) + delta # update embeddings
    #             embeddings = embeddings.detach()
    #             #print(f"embeddings (non-zero rows): {embeddings[:, torch.tensor(self.prompt_mask).bool(), :]}")
    #             embeddings.requires_grad = True
    #             generated_embedding.append(embeddings)
    #             #embeddings.grad.fill_(0)
    #             #print(embeddings.grad)
    #             print(f"Adversarial step {i+1}/{m}, Loss: {loss.item()}")
        
    #             # Inner loop: Standard training with adversarial examples
    #             total_loss = 0
    #             embeddings.requires_grad = False
    #             # previous_params = {name: param.data.clone() for name, param in model.named_parameters()}
    #     for embedding in generated_embedding:
    #         for j in range(n):
    #             model_inputs = {**inputs, 'inputs_embeds': embedding}
    #             model_inputs.pop('input_ids', None)
    #             # Forward pass with adversarial embeddings
    #             self.optimizer.zero_grad()
    #             with self.compute_loss_context_manager():
    #                 loss = self.compute_loss(model, model_inputs)
                
    #             # Standard backward pass to update model parameters
    #             self.accelerator.backward(loss, retain_graph=True)
    #             if not j == n - 1:
    #                 self.optimizer.step()
    #             # 打印每个参数的梯度
    #             # 检查参数是否更新
    #             # for name, param in model.named_parameters():
    #             #     if not torch.equal(previous_params[name], param.data):
    #             #         print(f"Parameter '{name}' has been updated.")
    #             #     # 更新上一个参数值
    #             #     previous_params[name] = param.data.clone()
    #             total_loss += loss.detach()
    #             # print(f"Current Learning Rate: {self.optimizer.param_groups[0]['lr']}")
    #             print(f"Training step {j+1}/{n}, Loss: {loss.item()}")
        
    #     # Return average loss
    #     avg_loss = total_loss / n
    #     if num_items_in_batch is None:
    #         return avg_loss / self.args.gradient_accumulation_steps
    #     return avg_loss

    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimzer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    def _save(self, output_dir: Optional[str] = None, state_dict: Optional[Dict[str, "torch.Tensor"]] = None) -> None:
        super()._save(output_dir, state_dict)
        if self.processor is not None:
            output_dir = output_dir if output_dir is not None else self.args.output_dir
            getattr(self.processor, "image_processor").save_pretrained(output_dir)

    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        r"""
        Removes the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        labels = inputs["labels"].detach().clone() if "labels" in inputs else None  # backup labels
        if self.args.predict_with_generate:
            assert self.tokenizer.padding_side == "left", "This method only accepts left-padded tensor."
            prompt_len, label_len = inputs["input_ids"].size(-1), inputs["labels"].size(-1)
            if prompt_len > label_len:
                inputs["labels"] = self._pad_tensors_to_target_len(inputs["labels"], inputs["input_ids"])
            if label_len > prompt_len:  # truncate the labels instead of padding the inputs (llama2 fp16 compatibility)
                inputs["labels"] = inputs["labels"][:, :prompt_len]

        loss, generated_tokens, _ = super().prediction_step(  # ignore the returned labels (may be truncated)
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, :prompt_len] = self.tokenizer.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def _pad_tensors_to_target_len(self, src_tensor: torch.Tensor, tgt_tensor: torch.Tensor) -> torch.Tensor:
        r"""
        Pads the tensor to the same length as the target tensor.
        """
        assert self.tokenizer.pad_token_id is not None, "Pad token is required."
        padded_tensor = self.tokenizer.pad_token_id * torch.ones_like(tgt_tensor)
        padded_tensor[:, -src_tensor.shape[-1] :] = src_tensor  # adopt left-padding
        return padded_tensor.contiguous()  # in contiguous memory

    def save_predictions(self, predict_results: "PredictionOutput") -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.tokenizer.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX, predict_results.predictions, self.tokenizer.pad_token_id
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.tokenizer.pad_token_id)[0]
            if len(pad_len):
                preds[i] = np.concatenate(
                    (preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1
                )  # move pad token to last

        decoded_labels = self.tokenizer.batch_decode(
            labels, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: List[str] = []
            for label, pred in zip(decoded_labels, decoded_preds):
                res.append(json.dumps({"label": label, "predict": pred}, ensure_ascii=False))
            writer.write("\n".join(res))
            
    def get_tfms_metric(self): # get training-free model selection metrics
        self.accelerator.free_memory()
        train_dataloader = self.get_train_dataloader()
        model = self._wrap_model(self.model_wrapped)
        # logger.info(f"======================trainable params====================")
        # for k, v in model.named_parameters():
        #     if v.requires_grad:
        #         logger.info(f"{k}")
        # trainable_params, all_param = count_parameters(model)
        
        num_samples = 10
        norms, grads = [], []
        for step, inputs in enumerate(train_dataloader):
            model.zero_grad()
            loss = self.training_step(model, inputs)
            
            tmp = [ # get sum of grad ** 2 and number of elements in grad
                [torch.sum(p.grad ** 2), p.grad.numel()] \
                    for p in model.parameters() \
                        if (p.grad is not None) and (not torch.isnan(p.grad).any())
            ]
            norm = torch.sqrt(sum([v[0] for v in tmp]) / sum([v[1] for v in tmp]))
            
            if torch.isnan(norm).item():
                logger.info(f"step: {step}, tmp: {tmp}")
                continue
            norms.append(norm.item())
            logger.info(f"step: {step}, norm: {norm.item()}")
            # for condition number
            grads.append(torch.cat([
                torch.nan_to_num(p.grad.clone(), nan=0.0).view(-1) \
                    for p in model.parameters() \
                        if (p.grad is not None)
            ]))

            if len(norms) >= num_samples:
                break
        
        grads = torch.stack(grads, dim=0)
        eigenvalues, _ = torch.sort(torch.linalg.eigvals(grads @ grads.t()).abs())
        cond_num = np.nan_to_num((eigenvalues[-1] / eigenvalues[0]).item(), copy=True, nan=100000.0)
        metrics = [sum(norms) / len(norms), cond_num]
        
        logger.info(f"metrics: {metrics}")
        return
