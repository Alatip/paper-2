## Phase 2 (1). Genuine W-only debate: four agents, three rounds, round 0 = matrix answers

Rows collected: 2352 of 2352; items complete in all rounds: 196 of 196 (contested 146 of 146, control 40 of 40, allwrong 10 of 10).
Tie rules: first in order = DeepSeek, GLM, Mistral, Haiku; strongest first = W sorted by round-0 accuracy on the completed contested items (in-sample): GLM, Haiku, DeepSeek, Mistral (GLM 0.815, Haiku 0.623, DeepSeek 0.575, Mistral 0.322); ties split = 1/k credit. A cut-off row (no FINAL line) is a missing vote.

### contested-for-W: 146 completed problems (gpqa-diamond 29, gsm-hard 16, mmlu-pro 95, olympiad 6)

| round | agent acc (answered rows) | plurality, first in order | plurality, strongest first | plurality, ties split | ties at top | unanimous | fixed / broken (first) | fixed / broken (strongest) | mean φ (pairs) | n_eff | all answering wrong | unanimous & wrong | cut-off rows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.584 | 0.726 | 0.801 | 0.740 | 23 | 0 | 0 / 0 | 0 / 0 | -0.098 (6/6) | 5.657 | 0 | 0 | 0 |
| 1 | 0.801 | 0.849 | 0.856 | 0.829 | 11 | 91 | 24 / 6 | 11 / 3 | 0.424 (6/6) | 1.760 | 8 | 6 | 22 |
| 2 | 0.857 | 0.870 | 0.870 | 0.856 | 9 | 114 | 26 / 5 | 13 / 3 | 0.536 (6/6) | 1.533 | 10 | 8 | 18 |
| 3 | 0.851 | 0.877 | 0.870 | 0.872 | 6 | 114 | 27 / 5 | 14 / 4 | 0.578 (6/6) | 1.463 | 11 | 9 | 14 |

Round 0 → 3 gain: first in order +0.151 (27 fixed / 5 broken, exact McNemar p = 0.000); strongest first +0.068 (14 / 4, p = 0.031); ties split +0.132. Fixed problems whose round-0 plurality was a tie: 14 of 27 (first in order).

Per layer, round 0 → 3 (first in order / strongest first / ties split): mmlu-pro (95): 0.716 → 0.853 / 0.768 → 0.842 / 0.716 → 0.846; gpqa-diamond (29): 0.655 → 0.931 / 0.828 → 0.931 / 0.707 → 0.931; gsm-hard (16): 0.875 → 0.875 / 0.875 → 0.875 / 0.875 → 0.875; olympiad (6): 0.833 → 1.000 / 1.000 → 1.000 / 0.917 → 1.000.

### control (all W right at round 0): 40 completed problems (gpqa-diamond 3, gsm-hard 7, mmlu-pro 28, olympiad 2)

| round | agent acc (answered rows) | plurality, first in order | plurality, strongest first | plurality, ties split | ties at top | unanimous | fixed / broken (first) | fixed / broken (strongest) | mean φ (pairs) | n_eff | all answering wrong | unanimous & wrong | cut-off rows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0 | 38 | 0 / 0 | 0 / 0 | — (0/6) | — | 0 | 0 | 0 |
| 1 | 1.000 | 1.000 | 1.000 | 1.000 | 0 | 38 | 0 / 0 | 0 / 0 | — (0/6) | — | 0 | 0 | 0 |
| 2 | 1.000 | 1.000 | 1.000 | 1.000 | 0 | 40 | 0 / 0 | 0 / 0 | — (0/6) | — | 0 | 0 | 0 |
| 3 | 1.000 | 1.000 | 1.000 | 1.000 | 0 | 40 | 0 / 0 | 0 / 0 | — (0/6) | — | 0 | 0 | 1 |

Round 0 → 3 gain: first in order +0.000 (0 fixed / 0 broken, exact McNemar p = 1.000); strongest first +0.000 (0 / 0, p = 1.000); ties split +0.000. Fixed problems whose round-0 plurality was a tie: 0 of 0 (first in order).

Per layer, round 0 → 3 (first in order / strongest first / ties split): mmlu-pro (28): 1.000 → 1.000 / 1.000 → 1.000 / 1.000 → 1.000; gpqa-diamond (3): 1.000 → 1.000 / 1.000 → 1.000 / 1.000 → 1.000; gsm-hard (7): 1.000 → 1.000 / 1.000 → 1.000 / 1.000 → 1.000; olympiad (2): 1.000 → 1.000 / 1.000 → 1.000 / 1.000 → 1.000.

### all W wrong at round 0: 10 completed problems (gsm-hard 1, mmlu-pro 9)

| round | agent acc (answered rows) | plurality, first in order | plurality, strongest first | plurality, ties split | ties at top | unanimous | fixed / broken (first) | fixed / broken (strongest) | mean φ (pairs) | n_eff | all answering wrong | unanimous & wrong | cut-off rows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.000 | 0.000 | 0.000 | 0.000 | 1 | 4 | 0 / 0 | 0 / 0 | — (0/6) | — | 10 | 4 | 0 |
| 1 | 0.079 | 0.100 | 0.100 | 0.100 | 1 | 6 | 1 / 0 | 1 / 0 | 0.667 (1/6) | 1.333 | 8 | 6 | 2 |
| 2 | 0.158 | 0.200 | 0.200 | 0.200 | 0 | 5 | 2 / 0 | 2 / 0 | 0.583 (3/6) | 1.455 | 7 | 4 | 2 |
| 3 | 0.211 | 0.200 | 0.200 | 0.200 | 0 | 5 | 2 / 0 | 2 / 0 | 0.611 (6/6) | 1.412 | 6 | 4 | 2 |

Round 0 → 3 gain: first in order +0.200 (2 fixed / 0 broken, exact McNemar p = 0.500); strongest first +0.200 (2 / 0, p = 0.500); ties split +0.200. Fixed problems whose round-0 plurality was a tie: 0 of 2 (first in order).

Per layer, round 0 → 3 (first in order / strongest first / ties split): mmlu-pro (9): 0.000 → 0.222 / 0.000 → 0.222 / 0.000 → 0.222; gsm-hard (1): 0.000 → 0.000 / 0.000 → 0.000 / 0.000 → 0.000.

### all run items: 196 completed problems (gpqa-diamond 32, gsm-hard 24, mmlu-pro 132, olympiad 8)

| round | agent acc (answered rows) | plurality, first in order | plurality, strongest first | plurality, ties split | ties at top | unanimous | fixed / broken (first) | fixed / broken (strongest) | mean φ (pairs) | n_eff | all answering wrong | unanimous & wrong | cut-off rows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.639 | 0.745 | 0.801 | 0.755 | 24 | 42 | 0 / 0 | 0 / 0 | 0.156 (6/6) | 2.724 | 10 | 4 | 0 |
| 1 | 0.807 | 0.842 | 0.847 | 0.827 | 12 | 135 | 25 / 6 | 12 / 3 | 0.560 (6/6) | 1.493 | 16 | 12 | 24 |
| 2 | 0.852 | 0.862 | 0.862 | 0.852 | 9 | 159 | 28 / 5 | 15 / 3 | 0.664 (6/6) | 1.337 | 17 | 12 | 20 |
| 3 | 0.850 | 0.867 | 0.862 | 0.864 | 6 | 159 | 29 / 5 | 16 / 4 | 0.662 (6/6) | 1.340 | 17 | 13 | 17 |

Round 0 → 3 gain: first in order +0.122 (29 fixed / 5 broken, exact McNemar p = 0.000); strongest first +0.061 (16 / 4, p = 0.012); ties split +0.109. Fixed problems whose round-0 plurality was a tie: 14 of 29 (first in order).

Per layer, round 0 → 3 (first in order / strongest first / ties split): mmlu-pro (132): 0.727 → 0.841 / 0.765 → 0.833 / 0.727 → 0.836; gpqa-diamond (32): 0.688 → 0.938 / 0.844 → 0.938 / 0.734 → 0.938; gsm-hard (24): 0.875 → 0.875 / 0.875 → 0.875 / 0.875 → 0.875; olympiad (8): 0.875 → 1.000 / 1.000 → 1.000 / 0.938 → 1.000.

### Round-3 W plurality against single W members on the completed run items

| member | round-0 accuracy | member right & r3 plurality (first) wrong / reverse, p | same vs r3 plurality (strongest first) |
|---|---|---|---|
| DeepSeek | 0.633 | 6 / 52, p = 0.000 | 7 / 52, p = 0.000 |
| GLM | 0.811 | 6 / 17, p = 0.035 | 6 / 16, p = 0.052 |
| Mistral | 0.444 | 8 / 91, p = 0.000 | 8 / 90, p = 0.000 |
| Haiku | 0.668 | 5 / 44, p = 0.000 | 6 / 44, p = 0.000 |

Projection to the 733 problems (strongest-first rule; the 577 all-W-right problems outside the control were not debated): round-0 W plurality 0.947; round-3 0.963 if none of the undebated all-right problems breaks, 0.963 if they break at the control's rate (0 of 40, Wilson [-0.000, 0.088]). Reference (weak-pool-voting.md, P4 on the 733): in-sample per-set best member pooled 0.952, oracle 0.986, plurality 0.933 / 0.935.

Flips: a correct agent at round t, answering at t + 1, by number of answering peers holding a different answer → share that flipped to wrong: 0 peers: 11/1211 (0.01); 1 peers: 35/356 (0.10); 2 peers: 33/133 (0.25); 3 peers: 19/41 (0.46).

Mean completion tokens per agent-turn (rounds 1–3): DeepSeek 4377, GLM 2241, Mistral 181, Haiku 2631.

## Phase 2 (2). Synthesis: one DeepSeek call reads the four W round-0 answers

- contested-for-W: 146 items; synthesis correct on 124 (0.849, Wilson [0.782, 0.898]); unparsed 0; novel answers (not among the four inputs) 0, of them correct 0. Round-0 W plurality (first in order) 0.726 vs synthesis 0.849: plurality right & synthesis wrong / reverse 5 / 23, p = 0.001. Round-0 W plurality (strongest first) 0.801 vs synthesis 0.849: plurality right & synthesis wrong / reverse 8 / 15, p = 0.210. Round-0 W plurality (ties split) 0.740 vs synthesis 0.849: plurality right & synthesis wrong / reverse 4 / 13. DeepSeek's own round-0 answer on these items: 0.575; synthesis picked an answer equal to DeepSeek's own on 76 of 146. On the 146 items also debated: round-3 W plurality (strongest first) 0.870 vs synthesis 0.849.
- all W wrong: 10 items; synthesis correct on 1 (0.100, Wilson [0.018, 0.404]); unparsed 0; novel answers (not among the four inputs) 1, of them correct 1.

Synthesis rows 156, OpenRouter cost $0.24.

## Phase 2 (3). J3 panel (Haiku, DeepSeek, GLM) picking among the W candidates

Contested items with all three picks: 146 of 146; all-W-wrong items with all three picks: 10. Candidates per item: 2: 110, 3: 34, 4: 2.

| judge | pick acc (unnamed = wrong) | named picks | pick acc on named picks | own round-0 acc on these items | picked own | picked own when own wrong | said NONE on the all-W-wrong items |
|---|---|---|---|---|---|---|---|
| Haiku | 0.644 | 138 | 0.681 | 0.623 | 0.658 | 0.455 (55) | 0 / 10 |
| DeepSeek | 0.829 | 143 | 0.846 | 0.575 | 0.596 | 0.226 (62) | 2 / 10 |
| GLM | 0.452 | 76 | 0.868 | 0.815 | 0.459 | 0.185 (27) | 0 / 10 |

Mean pairwise φ of 'pick correct' over the 146 items: 0.204 (3 pairs; tetrachoric 0.393); n_eff of 3 judges 2.131.

| tie rule | W solvers' plurality (round 0) | J3 panel | panel − solvers | won / lost, McNemar p |
|---|---|---|---|---|
| first in fixed order (solvers DeepSeek; judges Haiku) | 0.726 | 0.760 | +0.034 | 19 / 14, p = 0.487 |
| ties split evenly | 0.740 | 0.808 | +0.068 | – |
| strongest first (solvers GLM; judges DeepSeek) | 0.801 | 0.856 | +0.055 | 17 / 9, p = 0.169 |

Strict majority (≥ 2 of 3 name the same candidate): answers 116 of 146; right on 0.871 [Wilson 0.798, 0.920]; W plurality (strongest first) on those: 0.845; on the 30 abstained: 0.633.
Panel (strongest-judge ties) against the round-3 debate plurality (strongest first) on the 146 items with both: 0.856 vs 0.870; won / lost 8 / 10, p = 0.815.

Whole set, the 733 problems (all-W-right problems keep their answer, the 10 all-W-wrong are wrong, contested problems go to the panel): W plurality 0.932 (first in order) / 0.947 (strongest first); W plurality + J3 0.939 (first-judge ties; empty panel = wrong) / 0.947 (tied or empty panel falls back to the plurality) / 0.958 (strongest-judge ties); best single W member in-sample GLM 0.950; DeepSeek fixed 0.902; oracle over W 0.986.

Judge rows 468; cost $4.40 (Haiku's share is CLI-reported).

## Spend

| stage | rows | OpenRouter $ (billed) | Haiku $ (claude CLI report, subscription) | total $ |
|---|---|---|---|---|
| debate-w | 2352 | 2.09 | 10.27 | 12.36 |
| synth-w | 156 | 0.24 | 0.00 | 0.24 |
| judge-w | 468 | 0.53 | 3.87 | 4.40 |
| total | 2976 | 2.86 | 14.14 | 17.00 |

