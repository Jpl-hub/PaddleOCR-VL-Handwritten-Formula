# Competition Rules Summary

Source pages:

- Challenge issue: https://github.com/PaddlePaddle/PaddleOCR/issues/17858
- Full rules: https://github.com/PaddlePaddle/community/blob/master/hackathon/hackathon_10th/%E3%80%90Hackathon_10th%E3%80%91PaddleOCR%E5%85%A8%E7%90%83%E8%A1%8D%E7%94%9F%E6%A8%A1%E5%9E%8B%E6%8C%91%E6%88%98%E8%B5%9B.md
- Scoring rubric: https://github.com/PaddlePaddle/community/blob/master/hackathon/hackathon_10th/%E3%80%90Hackathon_10th%E3%80%91PaddleOCR%E8%AF%A6%E7%BB%86%E8%AF%84%E5%88%86%E8%A1%A8.md

## Deadlines

- Preliminary submission: June 28, 2026.
- Preliminary review: June 30 to July 5, 2026.
- Preliminary result: July 6, 2026.
- Final version for Top 10 teams: July 20, 2026.
- Final defense: July 22 to July 24, 2026.
- Final result: July 28, 2026.

## Deliverables

1. Evaluation set package.
   - Images/documents, annotations, task description, evaluation script, dataset description.
   - The evaluation set does not need to be public, but must be submitted as a link.
2. Training-data construction report.
   - Data collection method, synthesis method if any, key code, annotation guideline, tools, quality-control workflow.
3. Public open-source project.
   - GitHub repository with training/evaluation code, documentation, and demo.
   - Hugging Face model repository with a complete model card.
4. Final defense materials if promoted to Top 10.

## Scoring Priorities

| Dimension | Points | Repository Coverage |
|---|---:|---|
| Evaluation set quality | 20 | Dataset report, audit manifests, difficulty split, eval script |
| Scenario scarcity | 15 | Task motivation and comparison with standard OCR |
| Task complexity | 15 | LaTeX structure, visual noise, formula semantics |
| Training-data construction | 20 | Collection protocol, annotation guide, QC scripts, statistics |
| Fine-tuning strategy | 10 | LoRA/full SFT configs, ablation plan, evaluation table |
| Technical docs and open-source | 20 | Reproducible README, demo, model card, scripts |

## Authenticity Risks

- Award candidates may receive authenticity verification before preliminary results.
- Evaluation-set quality and authenticity can be checked against submitted code, model, and documents.
- If the evaluation set has an excessively high synthetic-data ratio, it can receive zero points and block prize ranking.

## Action Items

- Keep sample provenance fields and manifests for all training/evaluation data.
- Maintain a manual audit sheet for at least a sampled subset of labels.
- Prefer real collected evaluation samples; public datasets are suitable for baseline training but are weaker as a prize-level evaluation set by themselves.
- Keep commands, model versions, metrics, and artifacts reproducible.
