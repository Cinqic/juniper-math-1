# Juniper Math 1 Lessons Learned

Juniper Math 1 is complete. These lessons preserve what the project learned for future research; they do not prescribe a preselected successor model.

- **Scale and interference:** the tested 5,004,032-parameter architecture learned component skills, but the evidence supports a practical capacity/interference boundary for the complete joint objective.
- **Data and evaluation:** synthetic data can provide reproducible ground truth, but independently structured held-out prompts are essential to expose template overfitting and limited linguistic generalization.
- **Tools:** supervise call construction, trusted runtime results, and post-tool continuation separately. Never treat model-shaped tool results as trusted execution.
- **Continual learning:** masked SFT, replay, curricula, and learning rate are coupled. Measure unmasked Base retention at every milestone rather than treating a low masked SFT loss as preservation.
- **Evaluation identity:** freeze datasets and evaluators, hash the actual rendered representation rather than only a split ID, and label changed denominators, parsers, generation budgets, and argument semantics.
- **Reproducibility:** clean Git provenance, immutable configurations, release-backed approved checkpoints, artifact hashes, dynamic padding, bounded preflights, and CI are research controls, not paperwork.
- **Process:** keep engineer and reviewer roles separate; preserve rejected experiments; and stop honestly when materially different interventions answer the question instead of forcing an approval.

The full evidence and implications are in [the final research conclusion](../reports/FINAL_RESEARCH_CONCLUSION.md).
