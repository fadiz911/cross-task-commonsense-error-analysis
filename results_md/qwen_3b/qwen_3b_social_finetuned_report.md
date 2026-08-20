# Robustness & Error Analysis Report (Fine-Grained Analysis)
 
 * **Model**: `./results/Qwen_Qwen2.5-3B-Instruct-social_iqa-best`
 * **Task**: `social_iqa`
 * **Total Evaluation Samples**: 1954
 
 ## Performance Across Fine-Grained Semantic Sub-domains
 
 | Sub-domain | Subset Accuracy | Count | % of Dev Set |
 |---|---|---|---|
| **Overall** | 82.79% | 1954 | 100.0% |
| **Negation_Direct** | 79.31% | 141 | 7.2% |
| **Negation_Lexical** | 85.42% | 24 | 1.2% |
| **Negation_Prepositional** | 81.25% | 16 | 0.8% |
| **Negation_Contrastive** | 73.57% | 20 | 1.0% |
| **Spatial_Directional** | 87.50% | 72 | 3.7% |
| **Spatial_Vertical** | 77.78% | 9 | 0.5% |
| **Spatial_Proximity** | 76.61% | 597 | 30.6% |
| **Causal_Attribution** | 81.17% | 374 | 19.1% |
| **Causal_Outcome** | 83.85% | 192 | 9.8% |
| **Temporal_Sequence** | 82.20% | 999 | 51.1% |
| **Temporal_Simultaneity** | 82.74% | 73 | 3.7% |
| **Social_Emotion** | 83.30% | 455 | 23.3% |
| **Social_Motivation** | 80.49% | 789 | 40.4% |
| **Social_Interpersonal** | 81.68% | 322 | 16.5% |
| **Comparative_Superlative** | 81.34% | 101 | 5.2% |
| **Comparative_Relative** | 92.24% | 58 | 3.0% |
| **Quantifier_Universal** | 87.93% | 116 | 5.9% |
| **Quantifier_Partial** | 81.66% | 100 | 5.1% |
| **Quantifier_Numeric** | 84.31% | 65 | 3.3% |
| **Baseline** | 86.14% | 204 | 10.4% |
 
 ## Observations & Analysis
 - **Negation**: Analyzed across Direct negation (*not*), Lexical (*never*), Prepositional (*without*), and Contrastive (*neither*). Helps pinpoint exactly which types of negations cause structural prediction flips.
 - **Spatial**: Broken down into Directional, Vertical, and Proximity to separate geometric containment from relative coordinate alignment.
 - **Causal**: Decomposed into Attribution (why it happened) and Outcome (what it results in).
 - **Temporal**: Analyzed via Sequence (before/after/next) and Simultaneity (during/while).
 - **Social/Emotional**: Split into Affective Emotion, Goal Motivation, and Interpersonal Relationships to isolate Theory of Mind dimensions.
 - **Comparative**: Differentiates relative comparison (*more/less*) from extreme bounds/superlatives (*best/most*).
 - **Quantifier**: Profiles universal scope (*all/each*), partial scope (*some/many*), and exact numerics (*one/double*).
