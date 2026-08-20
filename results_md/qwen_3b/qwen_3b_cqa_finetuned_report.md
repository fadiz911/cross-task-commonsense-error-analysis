# Robustness & Error Analysis Report (Fine-Grained Analysis)
 
 * **Model**: `./results/Qwen_Qwen2.5-3B-Instruct-commonsense_qa-best`
 * **Task**: `commonsense_qa`
 * **Total Evaluation Samples**: 1221
 
 ## Performance Across Fine-Grained Semantic Sub-domains
 
 | Sub-domain | Subset Accuracy | Count | % of Dev Set |
 |---|---|---|---|
 | **Overall** | 84.93% | 1221 | 100.0% |
| **Negation_Direct** | 81.82% | 132 | 10.8% |
| **Negation_Lexical** | 83.33% | 12 | 1.0% |
| **Negation_Prepositional** | 50.00% | 4 | 0.3% |
| **Negation_Contrastive** | 77.78% | 9 | 0.7% |
| **Spatial_Directional** | 70.00% | 20 | 1.6% |
| **Spatial_Vertical** | 90.00% | 10 | 0.8% |
| **Spatial_Proximity** | 77.78% | 36 | 2.9% |
| **Causal_Attribution** | 84.21% | 76 | 6.2% |
| **Causal_Outcome** | 87.88% | 33 | 2.7% |
| **Temporal_Sequence** | 85.26% | 95 | 7.8% |
| **Temporal_Simultaneity** | 83.33% | 42 | 3.4% |
| **Social_Emotion** | 90.62% | 32 | 2.6% |
| **Social_Motivation** | 81.25% | 80 | 6.6% |
| **Social_Interpersonal** | 92.86% | 14 | 1.1% |
| **Comparative_Superlative** | 81.25% | 48 | 3.9% |
| **Comparative_Relative** | 84.00% | 25 | 2.0% |
| **Quantifier_Universal** | 84.09% | 44 | 3.6% |
| **Quantifier_Partial** | 77.03% | 74 | 6.1% |
| **Quantifier_Numeric** | 84.13% | 63 | 5.2% |
| **Baseline** | 86.42% | 611 | 50.0% |
 
 ## Observations & Analysis
 - **Negation**: Analyzed across Direct negation (*not*), Lexical (*never*), Prepositional (*without*), and Contrastive (*neither*). Helps pinpoint exactly which types of negations cause structural prediction flips.
 - **Spatial**: Broken down into Directional, Vertical, and Proximity to separate geometric containment from relative coordinate alignment.
 - **Causal**: Decomposed into Attribution (why it happened) and Outcome (what it results in).
 - **Temporal**: Analyzed via Sequence (before/after/next) and Simultaneity (during/while).
 - **Social/Emotional**: Split into Affective Emotion, Goal Motivation, and Interpersonal Relationships to isolate Theory of Mind dimensions.
 - **Comparative**: Differentiates relative comparison (*more/less*) from extreme bounds/superlatives (*best/most*).
 - **Quantifier**: Profiles universal scope (*all/each*), partial scope (*some/many*), and exact numerics (*one/double*).
 