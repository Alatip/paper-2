## Phase 1 (a). Debate: W votes inside the six-agent debate (225 problems)

Round 0 = the matrix answers; rounds 1–3 = the stored debate rows, in which every agent (W included) saw all five peers, Sonnet and Qwen among them. A cut-off row is a missing vote. Ties: first in the paper's agent order restricted to the pool; strongest first = the pool sorted by round-0 accuracy on these problems with a cut-off scoring 0 (section T's rule); ties split = 1/k credit.

### Check: all six agents (must reproduce tables.md §B, §B2 and §T): 225 problems; pool order DeepSeek, GLM, Mistral, Qwen, Haiku, Sonnet; strongest-first order Sonnet, Haiku, DeepSeek, Qwen, Mistral, GLM (round-0 accuracy with a cut-off scoring 0: Sonnet 0.947, Haiku 0.858, DeepSeek 0.831, Qwen 0.818, Mistral 0.764, GLM 0.751)

| round | agent acc (answered rows) | plurality, first in order | plurality, strongest first | plurality, ties split | ties at top | unanimous | fixed / broken (first in order) | fixed / broken (strongest first) | mean φ (pairs) | n_eff | all answering wrong | unanimous & wrong | truncated rows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.884 | 0.924 | 0.951 | 0.934 | 8 | 152 | 0 / 0 | 0 / 0 | 0.342 (15/15) | 2.213 | 1 | 0 | 86 |
| 1 | 0.953 | 0.947 | 0.956 | 0.949 | 2 | 208 | 7 / 2 | 3 / 2 | 0.709 (15/15) | 1.320 | 3 | 3 | 55 |
| 2 | 0.967 | 0.973 | 0.973 | 0.969 | 2 | 218 | 11 / 0 | 5 / 0 | 0.830 (15/15) | 1.165 | 6 | 6 | 29 |
| 3 | 0.971 | 0.973 | 0.973 | 0.973 | 0 | 219 | 11 / 0 | 5 / 0 | 0.845 (15/15) | 1.148 | 4 | 4 | 21 |

Round 0 → 3, first in order: 11 fixed / 0 broken, exact McNemar p = 0.001; strongest first: 5 / 0, p = 0.062; ties split: +0.039. Problems with a pool agent cut off at round 0: 63 of 225; fixes on them: 10 of 11 (first in order), 4 of 5 (strongest first). Fixed problems where the round-0 plurality was a tie: 7 of 11.

Per layer, round 0 → 3 (first in order / strongest first / ties split): mmlu-pro (58): 0.931 → 0.948 / 0.948 → 0.948 / 0.940 → 0.948; gpqa-diamond (56): 0.929 → 0.982 / 0.929 → 0.982 / 0.929 → 0.982; gsm-hard (57): 0.965 → 0.965 / 0.965 → 0.965 / 0.965 → 0.965; olympiad (54): 0.870 → 1.000 / 0.963 → 1.000 / 0.902 → 1.000.

### W (four non-frontier agents): 225 problems; pool order DeepSeek, GLM, Mistral, Haiku; strongest-first order Haiku, DeepSeek, Mistral, GLM (round-0 accuracy with a cut-off scoring 0: Haiku 0.858, DeepSeek 0.831, Mistral 0.764, GLM 0.751)

| round | agent acc (answered rows) | plurality, first in order | plurality, strongest first | plurality, ties split | ties at top | unanimous | fixed / broken (first in order) | fixed / broken (strongest first) | mean φ (pairs) | n_eff | all answering wrong | unanimous & wrong | truncated rows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.848 | 0.862 | 0.884 | 0.875 | 19 | 156 | 0 / 0 | 0 / 0 | 0.374 (6/6) | 1.886 | 9 | 3 | 50 |
| 1 | 0.947 | 0.933 | 0.933 | 0.937 | 4 | 212 | 20 / 4 | 15 / 4 | 0.741 (6/6) | 1.241 | 8 | 6 | 35 |
| 2 | 0.966 | 0.964 | 0.960 | 0.962 | 1 | 220 | 24 / 1 | 19 / 2 | 0.850 (6/6) | 1.126 | 6 | 6 | 22 |
| 3 | 0.971 | 0.973 | 0.973 | 0.973 | 0 | 223 | 26 / 1 | 21 / 1 | 0.907 (6/6) | 1.075 | 6 | 6 | 13 |

Round 0 → 3, first in order: 26 fixed / 1 broken, exact McNemar p = 0.000; strongest first: 21 / 1, p = 0.000; ties split: +0.098. Problems with a pool agent cut off at round 0: 50 of 225; fixes on them: 21 of 26 (first in order), 16 of 21 (strongest first). Fixed problems where the round-0 plurality was a tie: 16 of 26.

Per layer, round 0 → 3 (first in order / strongest first / ties split): mmlu-pro (58): 0.948 → 0.948 / 0.948 → 0.948 / 0.945 → 0.948; gpqa-diamond (56): 0.821 → 0.982 / 0.839 → 0.982 / 0.848 → 0.982; gsm-hard (57): 0.895 → 0.965 / 0.895 → 0.965 / 0.905 → 0.965; olympiad (54): 0.778 → 1.000 / 0.852 → 1.000 / 0.796 → 1.000.

### W (four non-frontier agents), subset where every pool member answered at round 0: 175 problems; pool order DeepSeek, GLM, Mistral, Haiku; strongest-first order GLM, Haiku, DeepSeek, Mistral (round-0 accuracy with a cut-off scoring 0: GLM 0.966, Haiku 0.926, DeepSeek 0.920, Mistral 0.857)

| round | agent acc (answered rows) | plurality, first in order | plurality, strongest first | plurality, ties split | ties at top | unanimous | fixed / broken (first in order) | fixed / broken (strongest first) | mean φ (pairs) | n_eff | all answering wrong | unanimous & wrong | truncated rows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.917 | 0.949 | 0.971 | 0.956 | 6 | 143 | 0 / 0 | 0 / 0 | 0.393 (6/6) | 1.836 | 2 | 0 | 0 |
| 1 | 0.968 | 0.960 | 0.966 | 0.965 | 2 | 169 | 3 / 1 | 0 / 1 | 0.720 (6/6) | 1.266 | 3 | 3 | 9 |
| 2 | 0.978 | 0.977 | 0.977 | 0.977 | 0 | 174 | 5 / 0 | 1 / 0 | 0.933 (6/6) | 1.053 | 4 | 4 | 5 |
| 3 | 0.977 | 0.977 | 0.977 | 0.977 | 0 | 174 | 5 / 0 | 1 / 0 | 0.941 (6/6) | 1.046 | 4 | 4 | 3 |

Round 0 → 3, first in order: 5 fixed / 0 broken, exact McNemar p = 0.062; strongest first: 1 / 0, p = 1.000; ties split: +0.021. Problems with a pool agent cut off at round 0: 0 of 175; fixes on them: 0 of 5 (first in order), 0 of 1 (strongest first). Fixed problems where the round-0 plurality was a tie: 5 of 5.

Per layer, round 0 → 3 (first in order / strongest first / ties split): mmlu-pro (55): 0.982 → 0.982 / 0.982 → 0.982 / 0.973 → 0.982; gpqa-diamond (35): 0.886 → 0.971 / 0.971 → 0.971 / 0.929 → 0.971; gsm-hard (52): 0.942 → 0.962 / 0.962 → 0.962 / 0.947 → 0.962; olympiad (33): 0.970 → 1.000 / 0.970 → 1.000 / 0.970 → 1.000.

### W3 (DeepSeek, GLM, Mistral): 225 problems; pool order DeepSeek, GLM, Mistral; strongest-first order DeepSeek, Mistral, GLM (round-0 accuracy with a cut-off scoring 0: DeepSeek 0.831, Mistral 0.764, GLM 0.751)

| round | agent acc (answered rows) | plurality, first in order | plurality, strongest first | plurality, ties split | ties at top | unanimous | fixed / broken (first in order) | fixed / broken (strongest first) | mean φ (pairs) | n_eff | all answering wrong | unanimous & wrong | truncated rows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.845 | 0.849 | 0.849 | 0.844 | 38 | 162 | 0 / 0 | 0 / 0 | 0.371 (3/3) | 1.723 | 19 | 5 | 50 |
| 1 | 0.950 | 0.933 | 0.933 | 0.938 | 5 | 214 | 23 / 4 | 23 / 4 | 0.733 (3/3) | 1.216 | 10 | 8 | 35 |
| 2 | 0.968 | 0.973 | 0.973 | 0.967 | 3 | 219 | 29 / 1 | 29 / 1 | 0.818 (3/3) | 1.138 | 6 | 6 | 22 |
| 3 | 0.971 | 0.969 | 0.969 | 0.970 | 1 | 224 | 28 / 1 | 28 / 1 | 0.945 (3/3) | 1.038 | 6 | 6 | 13 |

Round 0 → 3, first in order: 28 fixed / 1 broken, exact McNemar p = 0.000; strongest first: 28 / 1, p = 0.000; ties split: +0.126. Problems with a pool agent cut off at round 0: 50 of 225; fixes on them: 22 of 28 (first in order), 22 of 28 (strongest first). Fixed problems where the round-0 plurality was a tie: 23 of 28.

Per layer, round 0 → 3 (first in order / strongest first / ties split): mmlu-pro (58): 0.948 → 0.948 / 0.948 → 0.948 / 0.945 → 0.948; gpqa-diamond (56): 0.839 → 0.982 / 0.839 → 0.982 / 0.815 → 0.982; gsm-hard (57): 0.895 → 0.965 / 0.895 → 0.965 / 0.901 → 0.965; olympiad (54): 0.704 → 0.981 / 0.704 → 0.981 / 0.707 → 0.988.

### W3 (DeepSeek, GLM, Mistral), subset where every pool member answered at round 0: 175 problems; pool order DeepSeek, GLM, Mistral; strongest-first order GLM, DeepSeek, Mistral (round-0 accuracy with a cut-off scoring 0: GLM 0.966, DeepSeek 0.920, Mistral 0.857)

| round | agent acc (answered rows) | plurality, first in order | plurality, strongest first | plurality, ties split | ties at top | unanimous | fixed / broken (first in order) | fixed / broken (strongest first) | mean φ (pairs) | n_eff | all answering wrong | unanimous & wrong | truncated rows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.914 | 0.943 | 0.966 | 0.949 | 7 | 143 | 0 / 0 | 0 / 0 | 0.362 (3/3) | 1.741 | 3 | 0 | 0 |
| 1 | 0.973 | 0.971 | 0.971 | 0.971 | 2 | 170 | 5 / 0 | 2 / 1 | 0.686 (3/3) | 1.265 | 3 | 3 | 9 |
| 2 | 0.979 | 0.977 | 0.977 | 0.977 | 0 | 174 | 6 / 0 | 3 / 1 | 0.902 (3/3) | 1.070 | 4 | 4 | 5 |
| 3 | 0.979 | 0.977 | 0.977 | 0.977 | 0 | 175 | 6 / 0 | 3 / 1 | 1.000 (3/3) | 1.000 | 4 | 4 | 3 |

Round 0 → 3, first in order: 6 fixed / 0 broken, exact McNemar p = 0.031; strongest first: 3 / 1, p = 0.625; ties split: +0.029. Problems with a pool agent cut off at round 0: 0 of 175; fixes on them: 0 of 6 (first in order), 0 of 3 (strongest first). Fixed problems where the round-0 plurality was a tie: 5 of 6.

Per layer, round 0 → 3 (first in order / strongest first / ties split): mmlu-pro (55): 0.982 → 0.982 / 1.000 → 0.982 / 0.988 → 0.982; gpqa-diamond (35): 0.886 → 0.971 / 0.943 → 0.971 / 0.905 → 0.971; gsm-hard (52): 0.942 → 0.962 / 0.962 → 0.962 / 0.949 → 0.962; olympiad (33): 0.939 → 1.000 / 0.939 → 1.000 / 0.929 → 1.000.

Round-0 accuracy of each agent on the 225 problems, a cut-off scoring 0 (and on its own answered rows): DeepSeek 0.831 (0.831 on its 225 answered); GLM 0.751 (0.966 on its 175 answered); Mistral 0.764 (0.764 on its 225 answered); Qwen 0.818 (0.974 on its 189 answered); Haiku 0.858 (0.858 on its 225 answered); Sonnet 0.947 (0.947 on its 225 answered).

### Existing non-frontier homogeneous debate: six DeepSeek copies (tables.md §B5, reproduced)

| arm | items complete | r0 agent acc | r0 majority | r3 agent acc | r3 majority | r0 unanimous | r3 unanimous | unanimous&wrong r0→r3 | all-wrong r0 | fixed | broken | r0 φ | r3 φ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| plain | 236 | 0.820 | 0.856 | 0.898 | 0.894 | 158 | 232 | 6→22 | 15 | 12 | 3 | 0.58 | 0.97 |
| selfonly | 230 | 0.832 | 0.865 | 0.850 | 0.883 | 157 | 164 | 6→2 | 13 | 8 | 4 | 0.57 | 0.58 |
| pressure | 236 | 0.820 | 0.856 | 0.907 | 0.903 | 158 | 229 | 6→18 | 15 | 13 | 2 | 0.58 | 0.95 |

On the 230 items all three arms completed: round-0 majority 0.865; round-3 majority plain 0.900 (+0.035; fixed 11 / broken 3), selfonly 0.883 (+0.017; fixed 8 / broken 4), pressure 0.909 (+0.043; fixed 12 / broken 2).

selfonly = each copy reconsiders its own answer without seeing peers (controls for 'thinking again'); pressure = told to defer to the majority. Compare with the mixed-family debate in B/B2.
## Phase 1 (b). Judges: J3 = Haiku, DeepSeek, GLM on the stored picks

Contested items, candidates and the solvers' plurality come from the full six-proposer pool (caveat). Unnamed picks (NONE, cut off, a letter outside the list) score as wrong, as in Table 7.1. Solver strongest-first order over the judge items (cut-off = 0): Sonnet, Haiku, DeepSeek, Qwen, GLM, Mistral.

### Check: five judges (must reproduce Table 7.1 / tables.md §C, §C2, §T): 74 contested items with all 5 judges

| judge | contested items seen | pick acc (unnamed pick = wrong) | named picks | pick acc on named picks | own solve acc (answered) | picked own |
|---|---|---|---|---|---|---|
| Haiku | 75 | 0.733 | 73 | 0.753 | 0.560 (75) | 0.627 |
| Sonnet | 74 | 0.797 | 68 | 0.868 | 0.851 (74) | 0.770 |
| DeepSeek | 75 | 0.520 | 57 | 0.684 | 0.453 (75) | 0.387 |
| Qwen | 75 | 0.653 | 55 | 0.891 | 0.889 (45) | 0.520 |
| GLM | 75 | 0.480 | 43 | 0.837 | 0.824 (34) | 0.253 |

Mean pairwise φ of 'pick correct' (unnamed = wrong) over the 74 items: 0.157 (10 pairs; tetrachoric 0.275), n_eff of 5 judges 3.068.

| tie rule | solvers' plurality | panel | panel − solvers | panel won / lost, McNemar p |
|---|---|---|---|---|
| first in fixed order (solvers: DeepSeek; judges: Haiku) | 0.757 | 0.838 | +0.081 | 11 / 5, p = 0.210 |
| ties split evenly | 0.802 | 0.842 | +0.040 | – |
| strongest first (solvers: Sonnet; judges: Sonnet) | 0.865 | 0.851 | -0.014 | 4 / 5, p = 1.000 |

Strict majority (≥ 3 of 5 judges name the same candidate): answers 54 of 74; right on 0.944 [Wilson 0.849, 0.981]; solvers' plurality (fixed order) on those 54: 0.796; on the 20 abstained: 0.650.

Whole set, 235 items with all 5 judges (unanimous items keep their answer, contested items go to the panel, none-correct items are wrong): solvers' plurality 0.915; plurality + panel 0.940 (first-judge ties) / 0.953 (tied panel falls back to the plurality); best single solver with a cut-off scoring 0 (in-sample) Sonnet 0.945; DeepSeek fixed 0.817; W members: DeepSeek 0.817, GLM 0.753, Mistral 0.732, Haiku 0.851.

### J3 panel: 75 contested items with all 3 judges

| judge | contested items seen | pick acc (unnamed pick = wrong) | named picks | pick acc on named picks | own solve acc (answered) | picked own |
|---|---|---|---|---|---|---|
| Haiku | 75 | 0.733 | 73 | 0.753 | 0.560 (75) | 0.627 |
| DeepSeek | 75 | 0.520 | 57 | 0.684 | 0.453 (75) | 0.387 |
| GLM | 75 | 0.480 | 43 | 0.837 | 0.824 (34) | 0.253 |

Mean pairwise φ of 'pick correct' (unnamed = wrong) over the 75 items: 0.353 (3 pairs; tetrachoric 0.583), n_eff of 3 judges 1.758.

| tie rule | solvers' plurality | panel | panel − solvers | panel won / lost, McNemar p |
|---|---|---|---|---|
| first in fixed order (solvers: DeepSeek; judges: Haiku) | 0.747 | 0.720 | -0.027 | 10 / 12, p = 0.832 |
| ties split evenly | 0.792 | 0.720 | -0.072 | – |
| strongest first (solvers: Sonnet; judges: Haiku) | 0.853 | 0.720 | -0.133 | 4 / 14, p = 0.031 |

Strict majority (≥ 2 of 3 judges name the same candidate): answers 56 of 75; right on 0.786 [Wilson 0.662, 0.873]; solvers' plurality (fixed order) on those 56: 0.768; on the 19 abstained: 0.684.

Whole set, 236 items with all 3 judges (unanimous items keep their answer, contested items go to the panel, none-correct items are wrong): solvers' plurality 0.911; plurality + panel 0.903 (first-judge ties) / 0.911 (tied panel falls back to the plurality); best single solver with a cut-off scoring 0 (in-sample) Sonnet 0.941; DeepSeek fixed 0.818; W members: DeepSeek 0.818, GLM 0.750, Mistral 0.729, Haiku 0.852.

### Certification by J3 on the paper's candidate set: committee of Haiku, DeepSeek, GLM over the 94 candidates judged by all five (the paper's Table 7.2 set) (40 correct, 54 wrong, 23 decoys)

| accept if ≥ q of 3 | accepted | precision | recall of correct | wrong accepted (of 54) | false-accept rate, Wilson 95% | if every format must be accepted: wrong accepted / recall |
|---|---|---|---|---|---|---|
| 1 | 64 | 0.625 | 1.000 | 24 | 0.444 [0.320, 0.576] | 24 / 1.000 |
| 2 | 53 | 0.736 | 0.975 | 14 | 0.259 [0.161, 0.389] | 13 / 0.925 |
| 3 | 44 | 0.818 | 0.900 | 8 | 0.148 [0.077, 0.266] | 6 / 0.725 |

False-accept φ between the judges over the 54 wrong candidates: mean 0.521 (3 pairs), n_eff of 3 judges 1.469. Per-judge false-accept rate on them: Haiku 0.259, DeepSeek 0.333, GLM 0.259.

### Certification by J3 on every candidate all three ruled on: committee of Haiku, DeepSeek, GLM over the 109 candidates judged by all of Haiku, DeepSeek, GLM (46 correct, 63 wrong, 25 decoys)

| accept if ≥ q of 3 | accepted | precision | recall of correct | wrong accepted (of 63) | false-accept rate, Wilson 95% | if every format must be accepted: wrong accepted / recall |
|---|---|---|---|---|---|---|
| 1 | 78 | 0.590 | 1.000 | 32 | 0.508 [0.388, 0.627] | 32 / 1.000 |
| 2 | 64 | 0.703 | 0.978 | 19 | 0.302 [0.202, 0.424] | 18 / 0.935 |
| 3 | 51 | 0.804 | 0.891 | 10 | 0.159 [0.089, 0.268] | 8 / 0.739 |

False-accept φ between the judges over the 63 wrong candidates: mean 0.479 (3 pairs), n_eff of 3 judges 1.533. Per-judge false-accept rate on them: Haiku 0.317, DeepSeek 0.381, GLM 0.270.

### Check: five-judge committee (must reproduce Table 7.2): committee of Haiku, Sonnet, DeepSeek, Qwen, GLM over the 94 candidates judged by all five (the paper's Table 7.2 set) (40 correct, 54 wrong, 23 decoys)

| accept if ≥ q of 5 | accepted | precision | recall of correct | wrong accepted (of 54) | false-accept rate, Wilson 95% | if every format must be accepted: wrong accepted / recall |
|---|---|---|---|---|---|---|
| 1 | 66 | 0.606 | 1.000 | 26 | 0.481 [0.354, 0.611] | 26 / 1.000 |
| 2 | 56 | 0.714 | 1.000 | 16 | 0.296 [0.191, 0.428] | 16 / 1.000 |
| 3 | 49 | 0.816 | 1.000 | 9 | 0.167 [0.090, 0.287] | 8 / 0.950 |
| 4 | 44 | 0.864 | 0.950 | 6 | 0.111 [0.052, 0.222] | 5 / 0.825 |
| 5 | 35 | 0.971 | 0.850 | 1 | 0.019 [0.003, 0.098] | 0 / 0.675 |

False-accept φ between the judges over the 54 wrong candidates: mean 0.380 (10 pairs), n_eff of 5 judges 1.984. Per-judge false-accept rate on them: Haiku 0.259, Sonnet 0.019, DeepSeek 0.333, Qwen 0.204, GLM 0.259.

### J3 on a candidate set rebuilt from W's matrix answers only

Candidates = the distinct answers of the W proposers that answered; an item is contested-for-W when they disagree with at least one right and one wrong. The judges saw the full six-proposer list, so a stored pick may name a candidate that only a non-W proposer gave; such picks, NONE and unparsed picks count as 'none' (wrong). Solvers' plurality is over the W proposers (order DeepSeek, GLM, Mistral, Haiku; strongest first by round-0 accuracy with a cut-off scoring 0 over the judge items).

Contested-for-W items among the 236 judge items with all three J3 picks: 65 (aime24 10, aime25 15, gpqa 22, gsmhard 11, mmlupro 7); of them inside the full-pool contested set: 65; W strongest-first order Haiku, DeepSeek, GLM, Mistral (Haiku 0.852, DeepSeek 0.818, GLM 0.750, Mistral 0.729).

| judge | pick inside the W set | outside / NONE / unparsed (= none) | share none | pick acc (none = wrong) | pick acc on picks inside the set |
|---|---|---|---|---|---|
| Haiku | 63 | 2 | 0.031 | 0.769 | 0.794 |
| DeepSeek | 49 | 16 | 0.246 | 0.523 | 0.694 |
| GLM | 38 | 27 | 0.415 | 0.508 | 0.868 |

| tie rule | W solvers' plurality | J3 panel | panel − solvers | won / lost, McNemar p |
|---|---|---|---|---|
| first in fixed order (solvers DeepSeek; judges Haiku) | 0.631 | 0.754 | +0.123 | 17 / 9, p = 0.169 |
| ties split evenly | 0.683 | 0.746 | +0.063 | – |
| strongest first (solvers Haiku; judges Haiku) | 0.723 | 0.754 | +0.031 | 10 / 8, p = 0.815 |

Strict majority (≥ 2 of 3 inside the W set): answers 48 of 65; right on 0.812 [Wilson 0.681, 0.898].

Whole set, 236 judge items, W proposers only: W plurality 0.852; W plurality + J3 0.886 (first-judge ties; empty panel = wrong) / 0.890 (tied or empty panel falls back to the plurality); best W member with a cut-off scoring 0 (in-sample) Haiku 0.852; DeepSeek 0.818.

