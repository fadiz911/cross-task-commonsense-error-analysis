# Robustness & Error Analysis Report (Fine-Grained Analysis)
 
 * **Model**: `.\results\bert-base-uncased-social_iqa-best`
 * **Task**: `social_iqa`
 * **Total Evaluation Samples**: 1954
 
 ## Performance Across Fine-Grained Semantic Sub-domains
 
 | Sub-domain | Subset Accuracy | Count | % of Dev Set |
 |---|---|---|---|
 | **Overall** | 62.23% | 1954 | 100.0% |
| **Negation_Direct** | 57.45% | 141 | 7.2% |
| **Negation_Lexical** | 79.17% | 24 | 1.2% |
| **Negation_Prepositional** | 68.75% | 16 | 0.8% |
| **Negation_Contrastive** | 65.00% | 20 | 1.0% |
| **Spatial_Directional** | 63.89% | 72 | 3.7% |
| **Spatial_Vertical** | 55.56% | 9 | 0.5% |
| **Spatial_Proximity** | 60.80% | 597 | 30.6% |
| **Causal_Attribution** | 65.51% | 374 | 19.1% |
| **Causal_Outcome** | 60.42% | 192 | 9.8% |
| **Temporal_Sequence** | 62.76% | 999 | 51.1% |
| **Temporal_Simultaneity** | 63.01% | 73 | 3.7% |
| **Social_Emotion** | 62.64% | 455 | 23.3% |
| **Social_Motivation** | 61.98% | 789 | 40.4% |
| **Social_Interpersonal** | 64.29% | 322 | 16.5% |
| **Comparative_Superlative** | 65.35% | 101 | 5.2% |
| **Comparative_Relative** | 56.90% | 58 | 3.0% |
| **Quantifier_Universal** | 67.24% | 116 | 5.9% |
| **Quantifier_Partial** | 57.00% | 100 | 5.1% |
| **Quantifier_Numeric** | 61.54% | 65 | 3.3% |
| **Baseline** | 58.82% | 204 | 10.4% |
 
 ## Observations & Analysis
 - **Negation**: Analyzed across Direct negation (*not*), Lexical (*never*), Prepositional (*without*), and Contrastive (*neither*). Helps pinpoint exactly which types of negations cause structural prediction flips.
 - **Spatial**: Broken down into Directional, Vertical, and Proximity to separate geometric containment from relative coordinate alignment.
 - **Causal**: Decomposed into Attribution (why it happened) and Outcome (what it results in).
 - **Temporal**: Analyzed via Sequence (before/after/next) and Simultaneity (during/while).
 - **Social/Emotional**: Split into Affective Emotion, Goal Motivation, and Interpersonal Relationships to isolate Theory of Mind dimensions.
 - **Comparative**: Differentiates relative comparison (*more/less*) from extreme bounds/superlatives (*best/most*).
 - **Quantifier**: Profiles universal scope (*all/each*), partial scope (*some/many*), and exact numerics (*one/double*).
 