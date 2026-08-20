# Robustness & Error Analysis Report (Fine-Grained Analysis)
 
 * **Model**: `.\results\bert-base-uncased-commonsense_qa-best`
 * **Task**: `commonsense_qa`
 * **Total Evaluation Samples**: 1221
 
 ## Performance Across Fine-Grained Semantic Sub-domains
 
 | Sub-domain | Subset Accuracy | Count | % of Dev Set |
 |---|---|---|---|
 | **Overall** | 50.45% | 1221 | 100.0% |
| **Negation_Direct** | 45.45% | 132 | 10.8% |
| **Negation_Lexical** | 58.33% | 12 | 1.0% |
| **Negation_Prepositional** | 50.00% | 4 | 0.3% |
| **Negation_Contrastive** | 22.22% | 9 | 0.7% |
| **Spatial_Directional** | 35.00% | 20 | 1.6% |
| **Spatial_Vertical** | 60.00% | 10 | 0.8% |
| **Spatial_Proximity** | 58.33% | 36 | 2.9% |
| **Causal_Attribution** | 50.00% | 76 | 6.2% |
| **Causal_Outcome** | 48.48% | 33 | 2.7% |
| **Temporal_Sequence** | 46.32% | 95 | 7.8% |
| **Temporal_Simultaneity** | 42.86% | 42 | 3.4% |
| **Social_Emotion** | 46.88% | 32 | 2.6% |
| **Social_Motivation** | 52.50% | 80 | 6.6% |
| **Social_Interpersonal** | 64.29% | 14 | 1.1% |
| **Comparative_Superlative** | 47.92% | 48 | 3.9% |
| **Comparative_Relative** | 40.00% | 25 | 2.0% |
| **Quantifier_Universal** | 50.00% | 44 | 3.6% |
| **Quantifier_Partial** | 51.35% | 74 | 6.1% |
| **Quantifier_Numeric** | 55.56% | 63 | 5.2% |
| **Baseline** | 51.06% | 611 | 50.0% |
 
 ## Observations & Analysis
 - **Negation**: Analyzed across Direct negation (*not*), Lexical (*never*), Prepositional (*without*), and Contrastive (*neither*). Helps pinpoint exactly which types of negations cause structural prediction flips.
 - **Spatial**: Broken down into Directional, Vertical, and Proximity to separate geometric containment from relative coordinate alignment.
 - **Causal**: Decomposed into Attribution (why it happened) and Outcome (what it results in).
 - **Temporal**: Analyzed via Sequence (before/after/next) and Simultaneity (during/while).
 - **Social/Emotional**: Split into Affective Emotion, Goal Motivation, and Interpersonal Relationships to isolate Theory of Mind dimensions.
 - **Comparative**: Differentiates relative comparison (*more/less*) from extreme bounds/superlatives (*best/most*).
 - **Quantifier**: Profiles universal scope (*all/each*), partial scope (*some/many*), and exact numerics (*one/double*).
 