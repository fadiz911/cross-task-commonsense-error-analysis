# Robustness & Error Analysis Report (Fine-Grained Analysis)
 
 * **Model**: `./results/Qwen_Qwen2.5-0.5B-Instruct-social_iqa-best`
 * **Task**: `social_iqa`
 * **Total Evaluation Samples**: 1954
 
 ## Performance Across Fine-Grained Semantic Sub-domains
 
 | Sub-domain | Subset Accuracy | Count | % of Dev Set |
 |---|---|---|---|
 | **Overall** | 71.95% | 1954 | 100.0% |
| **Negation_Direct** | 73.76% | 141 | 7.2% |
| **Negation_Lexical** | 91.67% | 24 | 1.2% |
| **Negation_Prepositional** | 81.25% | 16 | 0.8% |
| **Negation_Contrastive** | 70.00% | 20 | 1.0% |
| **Spatial_Directional** | 75.00% | 72 | 3.7% |
| **Spatial_Vertical** | 66.67% | 9 | 0.5% |
| **Spatial_Proximity** | 70.18% | 597 | 30.6% |
| **Causal_Attribution** | 73.53% | 374 | 19.1% |
| **Causal_Outcome** | 72.40% | 192 | 9.8% |
| **Temporal_Sequence** | 73.17% | 999 | 51.1% |
| **Temporal_Simultaneity** | 71.23% | 73 | 3.7% |
| **Social_Emotion** | 72.97% | 455 | 23.3% |
| **Social_Motivation** | 72.62% | 789 | 40.4% |
| **Social_Interpersonal** | 73.60% | 322 | 16.5% |
| **Comparative_Superlative** | 71.29% | 101 | 5.2% |
| **Comparative_Relative** | 68.97% | 58 | 3.0% |
| **Quantifier_Universal** | 73.28% | 116 | 5.9% |
| **Quantifier_Partial** | 65.00% | 100 | 5.1% |
| **Quantifier_Numeric** | 70.77% | 65 | 3.3% |
| **Baseline** | 65.20% | 204 | 10.4% |
 
 ## Observations & Analysis
 - **Negation**: Analyzed across Direct negation (*not*), Lexical (*never*), Prepositional (*without*), and Contrastive (*neither*). Helps pinpoint exactly which types of negations cause structural prediction flips.
 - **Spatial**: Broken down into Directional, Vertical, and Proximity to separate geometric containment from relative coordinate alignment.
 - **Causal**: Decomposed into Attribution (why it happened) and Outcome (what it results in).
 - **Temporal**: Analyzed via Sequence (before/after/next) and Simultaneity (during/while).
 - **Social/Emotional**: Split into Affective Emotion, Goal Motivation, and Interpersonal Relationships to isolate Theory of Mind dimensions.
 - **Comparative**: Differentiates relative comparison (*more/less*) from extreme bounds/superlatives (*best/most*).
 - **Quantifier**: Profiles universal scope (*all/each*), partial scope (*some/many*), and exact numerics (*one/double*).
 