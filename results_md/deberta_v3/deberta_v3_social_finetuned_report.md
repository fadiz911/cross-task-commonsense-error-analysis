# Robustness & Error Analysis Report (Fine-Grained Analysis)
 
 * **Model**: `./results/microsoft_deberta-v3-large-social_iqa-best`
 * **Task**: `social_iqa`
 * **Total Evaluation Samples**: 1954
 
 ## Performance Across Fine-Grained Semantic Sub-domains
 
 | Sub-domain | Subset Accuracy | Count | % of Dev Set |
 |---|---|---|---|
 | **Overall** | 82.19% | 1954 | 100.0% |
| **Negation_Direct** | 80.14% | 141 | 7.2% |
| **Negation_Lexical** | 83.33% | 24 | 1.2% |
| **Negation_Prepositional** | 68.75% | 16 | 0.8% |
| **Negation_Contrastive** | 75.00% | 20 | 1.0% |
| **Spatial_Directional** | 87.50% | 72 | 3.7% |
| **Spatial_Vertical** | 77.78% | 9 | 0.5% |
| **Spatial_Proximity** | 80.90% | 597 | 30.6% |
| **Causal_Attribution** | 82.35% | 374 | 19.1% |
| **Causal_Outcome** | 83.85% | 192 | 9.8% |
| **Temporal_Sequence** | 82.68% | 999 | 51.1% |
| **Temporal_Simultaneity** | 80.82% | 73 | 3.7% |
| **Social_Emotion** | 83.30% | 455 | 23.3% |
| **Social_Motivation** | 82.64% | 789 | 40.4% |
| **Social_Interpersonal** | 81.68% | 322 | 16.5% |
| **Comparative_Superlative** | 82.18% | 101 | 5.2% |
| **Comparative_Relative** | 84.48% | 58 | 3.0% |
| **Quantifier_Universal** | 87.93% | 116 | 5.9% |
| **Quantifier_Partial** | 80.00% | 100 | 5.1% |
| **Quantifier_Numeric** | 83.08% | 65 | 3.3% |
| **Baseline** | 81.86% | 204 | 10.4% |
 
 ## Observations & Analysis
 - **Negation**: Analyzed across Direct negation (*not*), Lexical (*never*), Prepositional (*without*), and Contrastive (*neither*). Helps pinpoint exactly which types of negations cause structural prediction flips.
 - **Spatial**: Broken down into Directional, Vertical, and Proximity to separate geometric containment from relative coordinate alignment.
 - **Causal**: Decomposed into Attribution (why it happened) and Outcome (what it results in).
 - **Temporal**: Analyzed via Sequence (before/after/next) and Simultaneity (during/while).
 - **Social/Emotional**: Split into Affective Emotion, Goal Motivation, and Interpersonal Relationships to isolate Theory of Mind dimensions.
 - **Comparative**: Differentiates relative comparison (*more/less*) from extreme bounds/superlatives (*best/most*).
 - **Quantifier**: Profiles universal scope (*all/each*), partial scope (*some/many*), and exact numerics (*one/double*).
 