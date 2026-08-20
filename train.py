import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Union
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMultipleChoice,
    Trainer,
    TrainingArguments,
    PreTrainedTokenizerBase,
    AutoConfig,
    AutoModelForSequenceClassification
)
from dataset_utils import preprocess_choice_data

# Set up logging / output
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataCollatorForMultipleChoice:
    """
    Custom multiple-choice data collator.
    Flattens the features batch for padding via tokenizer.pad(),
    and then unflattens them back into tensors of shape
    (batch_size, num_choices, sequence_length).
    """
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Extract label/labels if present
        label_name = "label" if "label" in features[0] else "labels"
        labels = [feature.pop(label_name) for feature in features]
        
        # Check number of choices from the first feature
        # Standard input_ids should be of shape (num_choices, seq_length)
        num_choices = len(features[0]["input_ids"])
        batch_size = len(features)
        
        # Flatten features: from list of dicts of lists to list of dicts
        flat_features = []
        for feature in features:
            for i in range(num_choices):
                flat_feature = {}
                for k, v in feature.items():
                    flat_feature[k] = v[i]
                flat_features.append(flat_feature)
        
        # Apply padding
        padded_features = self.tokenizer.pad(
            flat_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        # Unflatten/reshape back into (batch_size, num_choices, sequence_length)
        batch = {}
        for k, v in padded_features.items():
            # v has shape (batch_size * num_choices, seq_len)
            seq_len = v.shape[-1]
            batch[k] = v.view(batch_size, num_choices, seq_len)
            
        # Add labels back as a tensor
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


def get_tokenization_mapper(tokenizer, task_name, max_length=128):
    """
    Returns a batched tokenization mapping function for the dataset splits.
    """
    def tokenize_function(examples):
        # Handle batching
        first_sentences = []
        second_sentences = []
        labels = []
        
        # Process each example in the batch
        # We need to handle cases where preprocess_choice_data might fail or return incomplete keys
        batch_size = len(next(iter(examples.values())))
        
        for i in range(batch_size):
            single_example = {k: examples[k][i] for k in examples.keys()}
            processed = preprocess_choice_data(single_example, task_name)
            
            context = processed["context"]
            choices = processed["choices"]
            label = processed["label"]
            
            # Replicate context for each choice
            for choice in choices:
                first_sentences.append(context)
                second_sentences.append(choice)
                
            labels.append(label)
            
        # Tokenize flattened list
        tokenized_inputs = tokenizer(
            first_sentences,
            second_sentences,
            truncation=True,
            max_length=max_length,
        )
        
        # Determine number of choices per prompt (e.g. 5 for commonsense_qa, 3 for social_iqa)
        # We assume all examples in the batch have the same number of choices
        first_processed = preprocess_choice_data({k: examples[k][0] for k in examples.keys()}, task_name)
        num_choices = len(first_processed["choices"])
        
        # Regroup/unflatten the resulting fields
        batch_outputs = {}
        for k, v in tokenized_inputs.items():
            batch_outputs[k] = [
                v[i : i + num_choices] for i in range(0, len(v), num_choices)
            ]
            
        batch_outputs["label"] = labels
        return batch_outputs
        
    return tokenize_function


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # predictions shape: (batch_size, num_choices)
    preds = np.argmax(predictions, axis=1)
    acc = (preds == labels).mean()
    return {"accuracy": acc}


class CausalLMForMultipleChoice(torch.nn.Module):
    def __init__(self, model_name_or_path, config=None, use_lora=False):
        super().__init__()
        if config is None:
            config = AutoConfig.from_pretrained(model_name_or_path)
        config.num_labels = 1
        self.config = config
        
        if use_lora:
            if "deberta" in model_name_or_path.lower():
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_name_or_path,
                    config=config,
                    device_map="auto",
                    use_safetensors=True
                )
            else:
                from transformers import BitsAndBytesConfig
                compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=compute_dtype
                )
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_name_or_path,
                    config=config,
                    quantization_config=bnb_config,
                    device_map="auto",
                    use_safetensors=True
                )
                from peft import prepare_model_for_kbit_training
                self.model = prepare_model_for_kbit_training(self.model)
            
            from peft import LoraConfig, get_peft_model, TaskType
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "query_proj", "value_proj", "key_proj"],
                modules_to_save=["classifier", "pooler", "score"]
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=config, use_safetensors=True)
        
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Extract target labels (Trainer might pass singular 'label' from the dataset)
        target_labels = labels
        if target_labels is None:
            target_labels = kwargs.pop("label", None)
        else:
            kwargs.pop("label", None)
            
        kwargs.pop("labels", None)
        
        batch_size, num_choices, seq_len = input_ids.shape
        flat_input_ids = input_ids.view(batch_size * num_choices, seq_len)
        flat_attention_mask = attention_mask.view(batch_size * num_choices, seq_len) if attention_mask is not None else None
        
        kwargs.pop("num_items_in_batch", None)
        outputs = self.model(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask,
            **kwargs
        )
        logits = outputs.logits.view(batch_size, num_choices)
        
        loss = None
        if target_labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits, target_labels)
            
        from transformers.modeling_outputs import MultipleChoiceModelOutput
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
            attentions=outputs.attentions if hasattr(outputs, "attentions") else None,
        )

    def save_pretrained(self, save_directory, **kwargs):
        self.model.save_pretrained(save_directory, **kwargs)
        self.config.save_pretrained(save_directory)

    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()

    def enable_input_require_grads(self):
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()
        elif hasattr(self.model, "model") and hasattr(self.model.model, "enable_input_require_grads"):
            self.model.model.enable_input_require_grads()


def load_model_and_tokenizer(model_name_or_path, use_lora=False):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    # Configure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Check if this is a directory containing a saved lora adapter
    adapter_config_path = os.path.join(model_name_or_path, "adapter_config.json") if os.path.isdir(model_name_or_path) else None
    if adapter_config_path and os.path.exists(adapter_config_path):
        logger.info(f"Detected saved LoRA adapter in {model_name_or_path}. Loading base model and adapter...")
        from peft import PeftConfig, PeftModel
        peft_config = PeftConfig.from_pretrained(model_name_or_path)
        base_model_name = peft_config.base_model_name_or_path
        
        config = AutoConfig.from_pretrained(base_model_name)
        config.num_labels = 1
        config.pad_token_id = tokenizer.pad_token_id
        
        if "deberta" in base_model_name.lower():
            base_model = AutoModelForSequenceClassification.from_pretrained(
                base_model_name,
                config=config,
                device_map="auto",
                use_safetensors=True
            )
        else:
            from transformers import BitsAndBytesConfig
            compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype
            )
            base_model = AutoModelForSequenceClassification.from_pretrained(
                base_model_name,
                config=config,
                quantization_config=bnb_config,
                device_map="auto",
                use_safetensors=True
            )
        peft_model = PeftModel.from_pretrained(base_model, model_name_or_path)
        
        model = CausalLMForMultipleChoice(base_model_name, config=config, use_lora=False)
        model.model = peft_model
        
        # Load the full state dict if model.safetensors exists to restore classifier head
        model_safetensors_path = os.path.join(model_name_or_path, "model.safetensors")
        if os.path.exists(model_safetensors_path):
            logger.info("Found full model.safetensors. Loading state dict directly to restore classifier head...")
            import safetensors.torch
            state_dict = safetensors.torch.load_file(model_safetensors_path)
            model.load_state_dict(state_dict)
            
        return model, tokenizer
        
    if use_lora:
        logger.info(f"Loading {model_name_or_path} with LoRA configuration...")
        config = AutoConfig.from_pretrained(model_name_or_path)
        config.pad_token_id = tokenizer.pad_token_id
        model = CausalLMForMultipleChoice(model_name_or_path, config=config, use_lora=True)
        return model, tokenizer
        
    try:
        # Try loading using standard AutoModelForMultipleChoice
        config = AutoConfig.from_pretrained(model_name_or_path)
        model = AutoModelForMultipleChoice.from_pretrained(model_name_or_path, config=config, use_safetensors=True)
        if tokenizer.pad_token is not None:
            model.config.pad_token_id = tokenizer.pad_token_id
        logger.info(f"Successfully loaded {model_name_or_path} with AutoModelForMultipleChoice")
    except (ValueError, OSError) as e:
        logger.info(f"Failed loading standard or falling back: {e}")
        logger.info(f"AutoModelForMultipleChoice not supported or failed for {model_name_or_path}. Wrapping with CausalLMForMultipleChoice.")
        config = AutoConfig.from_pretrained(model_name_or_path)
        config.pad_token_id = tokenizer.pad_token_id
        model = CausalLMForMultipleChoice(model_name_or_path, config=config)
            
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Unified Multiple-Choice Fine-Tuning Pipeline")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Pre-trained language model name or Hugging Face model identifier (e.g. gpt2, bert-base-uncased)"
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["commonsense_qa", "social_iqa"],
        help="NLP Task name"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Max sequence length for tokenization"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory to save fine-tuned model and checkpoints"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="If > 0: set total number of training steps to perform."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Training/eval batch size per device (overrides automatic selection)"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=None,
        help="Number of updates steps to accumulate before performing a backward/update pass"
    )
    parser.add_argument(
        "--optim",
        type=str,
        default=None,
        help="Optimizer name to use"
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Whether to use Parameter-Efficient Fine-Tuning (LoRA/QLoRA) during training"
    )
    
    args = parser.parse_args()
    
    logger.info(f"Initializing pipeline with model={args.model_name}, task={args.task}")
    
    # 1. Load tokenizer & model
    model, tokenizer = load_model_and_tokenizer(args.model_name, use_lora=args.use_lora)
        
    # 2. Load and preprocess dataset
    logger.info(f"Loading dataset: {args.task}")
    if args.task == "commonsense_qa":
        # commonsense_qa does not have a test set with labels, train and validation splits are typically used.
        dataset = load_dataset("tau/commonsense_qa")
        train_split = "train"
        val_split = "validation"
    elif args.task == "social_iqa":
        dataset = load_dataset("allenai/social_i_qa", revision="refs/convert/parquet")
        train_split = "train"
        val_split = "validation"
        
    # 3. Tokenize splits
    logger.info("Tokenizing and preprocessing dataset splits...")
    tokenization_fn = get_tokenization_mapper(tokenizer, args.task, max_length=args.max_length)
    
    tokenized_dataset = dataset.map(
        tokenization_fn,
        batched=True,
        remove_columns=dataset[train_split].column_names
    )
    
    # 4. Determine batch sizes based on model architecture size to prevent OOM on 12GB/6GB VRAM
    if args.batch_size is not None:
        batch_size = args.batch_size
    elif "large" in args.model_name or "3B" in args.model_name or "3b" in args.model_name:
        batch_size = 1
    else:
        batch_size = 8

    if args.gradient_accumulation_steps is not None:
        gradient_accumulation_steps = args.gradient_accumulation_steps
    elif "large" in args.model_name or "3B" in args.model_name or "3b" in args.model_name:
        gradient_accumulation_steps = 8
    else:
        gradient_accumulation_steps = 1
        
    logger.info(f"Selected batch size: {batch_size} (accumulation steps: {gradient_accumulation_steps})")
    
    # 5. Define training arguments
    if args.optim is not None:
        optimizer_name = args.optim
    elif "large" in args.model_name or "3B" in args.model_name or "3b" in args.model_name:
        optimizer_name = "adafactor"
    else:
        optimizer_name = "adamw_torch"

    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, f"{args.model_name.replace('/', '_')}-{args.task}"),
        eval_strategy="epoch" if args.max_steps <= 0 else "no",
        save_strategy="epoch" if args.max_steps <= 0 else "no",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=args.epochs if args.max_steps <= 0 else 1.0,
        max_steps=args.max_steps,
        weight_decay=0.01,
        load_best_model_at_end=True if args.max_steps <= 0 else False,
        metric_for_best_model="accuracy" if args.max_steps <= 0 else None,
        fp16=torch.cuda.is_available() if "deberta" not in args.model_name.lower() else False,
        bf16=torch.cuda.is_bf16_supported() if "deberta" in args.model_name.lower() else False,
        gradient_checkpointing=True,
        logging_steps=50 if args.max_steps <= 0 else 1,
        report_to="none", # Disable reporting (wandb, etc.) for clean run
        optim=optimizer_name,
    )
    
    # 6. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset[train_split],
        eval_dataset=tokenized_dataset[val_split],
        processing_class=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    
    # 7. Start fine-tuning
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    logger.info("Starting training loop...")
    trainer.train()
    
    # 8. Save the best model
    logger.info("Training complete. Saving best model...")
    save_path = os.path.join(args.output_dir, f"{args.model_name.replace('/', '_')}-{args.task}-best")
    trainer.save_model(save_path)
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    logger.info("Done!")

if __name__ == "__main__":
    main()
