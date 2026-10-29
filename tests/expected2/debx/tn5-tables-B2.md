## B2. Debate, deeper cuts (6 agents, 225 items complete in all rounds; truncated answers are missing)

| round | mean pairwise φ (pairwise-complete) | mean tetrachoric ρ (defined pairs) | ICC(1) (items with 6 answers) | n_eff (φ) | unanimous | unanimous & wrong | all answered wrong | items with 6 answers |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.34 | 0.77 (14/15) | 0.33 | 2.21 | 152 | 0 | 1 | 162 |
| 1 | 0.71 | 0.97 (10/15) | 0.78 | 1.32 | 208 | 3 | 3 | 185 |
| 2 | 0.83 | 0.99 (8/15) | 0.88 | 1.16 | 218 | 6 | 6 | 203 |
| 3 | 0.85 | 0.99 (9/15) | 0.93 | 1.15 | 219 | 4 | 4 | 207 |

### Items every answering agent got wrong at round 0 (the 1 − β ceiling) and what debate did with them

| layer | items | all answered wrong r0 | correct majority r3 among them | Wilson 95% | r0 majority | r3 majority | best single r0 (truncated = 0) | best single r0 (own answered) |
|---|---|---|---|---|---|---|---|---|
| mmlu-pro | 58 | 0 | 0 | [0.00, 0.00] | 0.931 | 0.948 | 0.966 | 1.000 |
| gpqa-diamond | 56 | 0 | 0 | [0.00, 0.00] | 0.929 | 0.982 | 0.893 | 0.980 |
| gsm-hard | 57 | 0 | 0 | [0.00, 0.00] | 0.965 | 0.965 | 0.982 | 0.982 |
| olympiad | 54 | 1 | 1 | [0.21, 1.00] | 0.870 | 1.000 | 0.963 | 1.000 |

### Round-0 correct count among answering agents → round-3 majority (transition table)

| correct at r0 (of answered) | items | of which all 6 answered | r0 majority correct | r3 majority correct | mean correct at r3 (answered) |
|---|---|---|---|---|---|
| 0 | 1 | 0 | 0.00 | 1.00 | 0.60 |
| 1 | 9 | 2 | 0.00 | 0.56 | 0.49 |
| 2 | 14 | 1 | 0.57 | 0.86 | 0.86 |
| 3 | 21 | 3 | 0.95 | 1.00 | 0.99 |
| 4 | 21 | 8 | 1.00 | 1.00 | 1.00 |
| 5 | 20 | 9 | 1.00 | 1.00 | 0.99 |
| 6 | 139 | 139 | 1.00 | 1.00 | 1.00 |

### Flips between rounds (agent correct at round t and answering at t+1): P(wrong at t+1) by number of answering peers holding a different answer (answer identity)

| peers disagreeing | cases | flipped to wrong |
|---|---|---|
| 0 | 3329 | 0.00 |
| 1 | 138 | 0.01 |
| 2 | 105 | 0.06 |
| 3 | 24 | 0.29 |
| 4 | 5 | 0.60 |
| 5 | 6 | 0.83 |

Per agent, when it alone was correct against 5 answering, disagreeing peers: kept the correct answer deepseek/deepseek-v3.2 0/2, z-ai/glm-5.3-flash 0/1, mistralai/mistral-large-2512 1/1, claude:sonnet55 0/2.

Mean completion tokens per agent-turn: round 1 1439, round 2 1122, round 3 1088; 3 rounds × 6 agents ≈ 21889 tokens per item on top of round 0.

