import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMultipleChoice, Trainer, AutoModelForCausalLM
from dataset_utils import preprocess_choice_data, SemanticFilters
from train import get_tokenization_mapper, DataCollatorForMultipleChoice, compute_metrics, load_model_and_tokenizer

def evaluate_causal_lm_zero_shot(model, tokenizer, standardized_examples, device="cuda"):
    model.to(device)
    model.eval()
    
    predictions = []
    
    import torch
    from tqdm import tqdm
    
    for idx, ex in enumerate(tqdm(standardized_examples, desc="Evaluating Zero-Shot Causal LM")):
        context = ex["context"]
        choices = ex["choices"]
        
        context_enc = tokenizer(context, return_tensors="pt")
        context_len = context_enc.input_ids.shape[1]
        
        choice_losses = []
        
        for choice in choices:
            sep = " " if not context.endswith(" ") else ""
            full_text = context + sep + choice
            
            enc = tokenizer(full_text, return_tensors="pt").to(device)
            input_ids = enc.input_ids
            attention_mask = enc.attention_mask
            
            if input_ids.shape[1] <= context_len:
                choice_losses.append(float('inf'))
                continue
                
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                logits = outputs.logits
                
                shift_logits = logits[..., context_len - 1 : -1, :].contiguous()
                shift_labels = input_ids[..., context_len:].contiguous()
                
                loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
                mean_loss = loss.mean().item()
                choice_losses.append(mean_loss)
                
        if len(choice_losses) == 0 or all(x == float('inf') for x in choice_losses):
            pred_idx = 0
        else:
            pred_idx = np.argmin(choice_losses)
        predictions.append(pred_idx)
        
    return np.array(predictions)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Fine-tuned Model Robustness on Semantic Sub-domains")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the fine-tuned model directory or HF model identifier"
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["commonsense_qa", "social_iqa"],
        help="NLP Task name"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Max sequence length for tokenization"
    )
    parser.add_argument(
        "--output_report",
        type=str,
        default="robustness_report.md",
        help="Path to write the markdown robustness report"
    )
    parser.add_argument(
        "--zero_shot",
        action="store_true",
        help="Whether to run zero-shot evaluation on raw causal LMs using token likelihood"
    )
    
    args = parser.parse_args()
    
    if args.zero_shot:
        print(f"Loading raw causal model for zero-shot likelihood evaluation: {args.model_path}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
    else:
        model, tokenizer = load_model_and_tokenizer(args.model_path)
        
    print(f"Loading dataset validation split for: {args.task}")
    if args.task == "commonsense_qa":
        dataset = load_dataset("tau/commonsense_qa", split="validation")
    else:
        dataset = load_dataset("allenai/social_i_qa", split="validation", revision="refs/convert/parquet")
        
    # Preprocess and extract context strings for filtering
    print("Preprocessing contexts for semantic classification...")
    standardized_examples = [preprocess_choice_data(example, args.task) for example in dataset]
    contexts = [ex["context"] for ex in standardized_examples]
    
    if args.zero_shot:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Running zero-shot Causal LM evaluation on device: {device}...")
        pred_indices = evaluate_causal_lm_zero_shot(model, tokenizer, standardized_examples, device=device)
        labels = np.array([ex["label"] for ex in standardized_examples])
    else:
        # Tokenize evaluation split
        print("Tokenizing evaluation split...")
        tokenization_fn = get_tokenization_mapper(tokenizer, args.task, max_length=args.max_length)
        tokenized_dataset = dataset.map(
            tokenization_fn,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Use Trainer to run prediction
        from transformers import TrainingArguments
        eval_args = TrainingArguments(
            output_dir="./tmp_eval",
            per_device_eval_batch_size=1,
            fp16=torch.cuda.is_available(),
            report_to="none"
        )
        trainer = Trainer(
            model=model,
            args=eval_args,
            data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
            compute_metrics=compute_metrics,
        )
        
        print("Running evaluation loop...")
        eval_output = trainer.predict(tokenized_dataset)
        predictions = eval_output.predictions
        labels = eval_output.label_ids
        
        # Get flat predictions (argmax)
        pred_indices = np.argmax(predictions, axis=1)
    
    # Categorize index list for semantic categories
    # Fine-grained sub-categories list
    subcategories = [
        "Negation_Direct", "Negation_Lexical", "Negation_Prepositional", "Negation_Contrastive",
        "Spatial_Directional", "Spatial_Vertical", "Spatial_Proximity",
        "Causal_Attribution", "Causal_Outcome",
        "Temporal_Sequence", "Temporal_Simultaneity",
        "Social_Emotion", "Social_Motivation", "Social_Interpersonal",
        "Comparative_Superlative", "Comparative_Relative",
        "Quantifier_Universal", "Quantifier_Partial", "Quantifier_Numeric",
        "Baseline"
    ]
    
    # Dictionary mapping sub-category name to list of matching sample indices
    indices_map = {sub: [] for sub in subcategories}
    
    for i, context in enumerate(contexts):
        sub_matches = SemanticFilters.profile_subcategories(context)
        if sub_matches:
            for sub in sub_matches:
                indices_map[sub].append(i)
        else:
            indices_map["Baseline"].append(i)
            
    # Calculate accuracies
    overall_acc = (pred_indices == labels).mean() * 100.0
    
    table_rows = []
    table_rows.append(f"| **Overall** | {overall_acc:.2f}% | {len(labels)} | 100.0% |")
    
    for sub in subcategories:
        indices = indices_map[sub]
        if not indices:
            acc, cnt = 0.0, 0
        else:
            sub_preds = pred_indices[indices]
            sub_labels = labels[indices]
            acc = (sub_preds == sub_labels).mean() * 100.0
            cnt = len(indices)
        pct = (cnt / len(labels)) * 100.0
        table_rows.append(f"| **{sub}** | {acc:.2f}% | {cnt} | {pct:.1f}% |")
        
    table_content = "\n".join(table_rows)
    
    report_content = f"""# Robustness & Error Analysis Report (Fine-Grained Analysis)
 
 * **Model**: `{args.model_path}`
 * **Task**: `{args.task}`
 * **Total Evaluation Samples**: {len(labels)}
 
 ## Performance Across Fine-Grained Semantic Sub-domains
 
 | Sub-domain | Subset Accuracy | Count | % of Dev Set |
 |---|---|---|---|
 {table_content}
 
 ## Observations & Analysis
 - **Negation**: Analyzed across Direct negation (*not*), Lexical (*never*), Prepositional (*without*), and Contrastive (*neither*). Helps pinpoint exactly which types of negations cause structural prediction flips.
 - **Spatial**: Broken down into Directional, Vertical, and Proximity to separate geometric containment from relative coordinate alignment.
 - **Causal**: Decomposed into Attribution (why it happened) and Outcome (what it results in).
 - **Temporal**: Analyzed via Sequence (before/after/next) and Simultaneity (during/while).
 - **Social/Emotional**: Split into Affective Emotion, Goal Motivation, and Interpersonal Relationships to isolate Theory of Mind dimensions.
 - **Comparative**: Differentiates relative comparison (*more/less*) from extreme bounds/superlatives (*best/most*).
 - **Quantifier**: Profiles universal scope (*all/each*), partial scope (*some/many*), and exact numerics (*one/double*).
 """
    
    with open(args.output_report, "w", encoding="utf-8") as f:
        f.write(report_content)
        
    print(f"\nReport successfully generated and saved to: {args.output_report}\n")
    print(report_content)

if __name__ == "__main__":
    main()
