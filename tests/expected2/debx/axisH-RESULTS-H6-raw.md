# H6 raw tables (stats_h6.py) 
## H6a. Hoeffding and the Pr[p > ½] voting ceiling (k=16 samples, in-sample p̂) 
| model | layer | n | ceiling Pr[p̂>½] | MAP ceiling (modal correct) | MV@4 obs | Hoeffding LB@4 | MV@8 obs | LB@8 | MV@16 obs | LB@16 | (ceil−LB4)/(ceil−MV4) | Hoeffding ½-width | Wilson ½-width | ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| claude_haiku | mmlu-pro | 59 | 0.924 | 0.915 | 0.898 | 0.776 | 0.915 | 0.890 | 0.915 | 0.912 | 5.8 | 0.177 | 0.073 | 2.41 |
| claude_haiku | gsm-hard | 60 | 0.867 | 0.867 | 0.900 | 0.712 | 0.867 | 0.820 | 0.867 | 0.846 | inf | 0.175 | 0.086 | 2.03 |
| deepseek_deepseek-v3.2 | mmlu-pro | 60 | 0.917 | 0.933 | 0.917 | 0.780 | 0.917 | 0.890 | 0.933 | 0.911 | inf | 0.175 | 0.067 | 2.64 |
| deepseek_deepseek-v3.2 | gpqa-diamond | 55 | 0.864 | 0.873 | 0.855 | 0.595 | 0.836 | 0.710 | 0.873 | 0.763 | 29.6 | 0.183 | 0.104 | 1.76 |
| deepseek_deepseek-v3.2 | gsm-hard | 59 | 0.864 | 0.864 | 0.864 | 0.695 | 0.864 | 0.811 | 0.864 | 0.844 | inf | 0.177 | 0.091 | 1.93 |
| deepseek_deepseek-v3.2 | olympiad | 49 | 0.837 | 0.898 | 0.878 | 0.514 | 0.857 | 0.631 | 0.898 | 0.690 | inf | 0.194 | 0.117 | 1.65 |
| z-ai_glm-5.3-flash | mmlu-pro | 55 | 0.982 | 0.982 | 0.982 | 0.849 | 0.982 | 0.964 | 0.982 | 0.981 | inf | 0.183 | 0.046 | 3.94 |
| z-ai_glm-5.3-flash | gpqa-diamond | 26 | 1.000 | 1.000 | 1.000 | 0.852 | 1.000 | 0.974 | 1.000 | 0.998 | inf | 0.266 | 0.064 | 4.14 |
| z-ai_glm-5.3-flash | gsm-hard | 52 | 0.923 | 0.923 | 0.885 | 0.752 | 0.885 | 0.865 | 0.923 | 0.891 | 4.4 | 0.188 | 0.082 | 2.29 |
| z-ai_glm-5.3-flash | olympiad | 19 | 1.000 | 1.000 | 1.000 | 0.771 | 1.000 | 0.916 | 1.000 | 0.964 | inf | 0.312 | 0.119 | 2.63 |
 Rows where MV@k exceeds the ceiling by > 0.01: 3 of 10. Rows where the Hoeffding gap is ≥ 2× the observed gap at k=4: 10 of 10. Min Hoeffding/Wilson half-width ratio: 1.65. 
## H6b. de Finetti: are draws exchangeable, and is a Beta-binomial mixture enough? 
### (i) Draw order: accuracy by draw index (chi-square homogeneity) and φ(i,j) on |i−j| 
| model | layer | n | acc by index (min–max) | χ² p | slope of φ on |i−j| | r |
|---|---|---|---|---|---|---|
| claude_haiku | mmlu-pro | 59 | 0.88–0.95 | 1.00 | -0.0001 | -0.00 |
| claude_haiku | gsm-hard | 60 | 0.83–0.90 | 1.00 | -0.0030 | -0.13 |
| deepseek_deepseek-v3.2 | mmlu-pro | 60 | 0.90–0.97 | 0.98 | -0.0010 | -0.03 |
| deepseek_deepseek-v3.2 | gpqa-diamond | 55 | 0.75–0.87 | 0.99 | +0.0020 | +0.07 |
| deepseek_deepseek-v3.2 | gsm-hard | 59 | 0.81–0.90 | 1.00 | +0.0005 | +0.02 |
| deepseek_deepseek-v3.2 | olympiad | 49 | 0.63–0.84 | 0.71 | +0.0014 | +0.04 |
| z-ai_glm-5.3-flash | mmlu-pro | 55 | 0.98–1.00 | 0.99 | +0.0000 | +0.00 |
| z-ai_glm-5.3-flash | gpqa-diamond | 26 | 0.96–1.00 | 0.60 | +0.0000 | +0.00 |
| z-ai_glm-5.3-flash | gsm-hard | 52 | 0.87–0.92 | 1.00 | -0.0002 | -0.01 |
| z-ai_glm-5.3-flash | olympiad | 19 | 0.84–1.00 | 0.33 | +0.0146 | +0.15 |
 p > 0.05 on 10 of 10 rows; |slope| < 0.01 on 9 of 10. 
### (ii) Beta-binomial vs two-state on the per-item correct counts (max log-likelihood) 
| model | layer | n | BB (α, β) | BB loglik | two-state (w, q) | 2S loglik | BB − 2S | BB-pred MV@16 | 2S-pred MV@16 | observed MV@16 |
|---|---|---|---|---|---|---|---|---|---|---|
| claude_haiku | mmlu-pro | 59 | (0.41, 0.05) | -50.8 | (0.03, 0.95) | -153.3 | +102.5 | 0.895 | 0.970 | 0.915 |
| claude_haiku | gsm-hard | 60 | (0.27, 0.05) | -73.3 | (0.07, 0.92) | -187.6 | +114.3 | 0.844 | 0.930 | 0.867 |
| deepseek_deepseek-v3.2 | mmlu-pro | 60 | (0.55, 0.05) | -45.9 | (0.02, 0.94) | -174.1 | +128.2 | 0.921 | 0.980 | 0.933 |
| deepseek_deepseek-v3.2 | gpqa-diamond | 55 | (0.55, 0.15) | -107.6 | (0.07, 0.87) | -178.6 | +71.0 | 0.787 | 0.930 | 0.873 |
| deepseek_deepseek-v3.2 | gsm-hard | 59 | (0.47, 0.09) | -85.8 | (0.03, 0.88) | -248.8 | +163.1 | 0.849 | 0.970 | 0.864 |
| deepseek_deepseek-v3.2 | olympiad | 49 | (0.96, 0.36) | -116.6 | (0.06, 0.80) | -184.8 | +68.2 | 0.747 | 0.933 | 0.898 |
| z-ai_glm-5.3-flash | mmlu-pro | 55 | (2.95, 0.05) | -12.7 | (0.00, 0.99) | -51.0 | +38.3 | 0.994 | 1.000 | 0.982 |
| z-ai_glm-5.3-flash | gpqa-diamond | 26 | (6.85, 0.05) | -6.6 | (0.00, 0.99) | -11.6 | +5.0 | 0.999 | 1.000 | 1.000 |
| z-ai_glm-5.3-flash | gsm-hard | 52 | (0.36, 0.05) | -52.5 | (0.04, 0.94) | -155.8 | +103.3 | 0.880 | 0.960 | 0.923 |
| z-ai_glm-5.3-flash | olympiad | 19 | (7.89, 0.47) | -24.6 | (0.00, 0.94) | -29.6 | +5.0 | 0.995 | 1.000 | 1.000 |
 Beta-binomial within 2 log-lik of two-state on 10 of 10 rows. 
### (iii) 7-model matrix: Beta-binomial fit to the count of correct models per item 
| layer | n | observed counts 0..7 | BB expected | χ² p | per-model acc range |
|---|---|---|---|---|---|
| mmlu-pro | 504 | 16 5 11 9 16 22 47 378 | 11 11 11 13 16 22 39 380 | 0.161 | 0.83–0.95 |
| gpqa-diamond | 92 | 1 1 0 2 6 7 13 62 | 0 1 2 3 4 7 13 62 | 0.293 | 0.75–0.98 |
| gsm-hard | 133 | 5 2 0 0 1 3 14 108 | 3 2 2 2 3 4 7 110 | 0.001 | 0.86–0.95 |
| olympiad | 29 | 0 0 0 0 1 2 3 23 | 0 0 0 0 1 1 4 23 | 0.913 | 0.83–1.00 |
| pooled | 758 | 22 8 11 11 24 34 77 571 | 17 16 17 20 24 34 59 572 | 0.005 | 0.83–0.95 |

## H6c. Fano / MAP ceiling: homogeneous DeepSeek debate vs DeepSeek's own modal answer 
| layer | items | DeepSeek MAP ceiling | single draw | homo plain r3 maj (Δ vs MAP) [boot] | homo selfonly r3 maj (Δ vs MAP) [boot] | homo pressure r3 maj (Δ vs MAP) [boot] | mixed r3 maj (Δ vs MAP) [boot] |
|---|---|---|---|---|---|---|---|
| mmlu-pro | 60 | 0.933 | 0.917 | 0.933 (+0.000) [-0.050, +0.050] n=60 | 0.933 (+0.000) [-0.050, +0.050] n=60 | 0.950 (+0.017) [+0.000, +0.050] n=60 | 0.932 (+0.000) [-0.051, +0.051] n=59 |
| gpqa-diamond | 55 | 0.873 | 0.818 | 0.873 (+0.000) [-0.073, +0.073] n=55 | 0.849 (-0.019) [-0.094, +0.038] n=53 | 0.873 (+0.000) [-0.055, +0.055] n=55 | 0.981 (+0.096) [+0.019, +0.173] n=52 |
| gsm-hard | 59 | 0.864 | 0.847 | 0.881 (+0.017) [+0.000, +0.068] n=59 | 0.881 (+0.017) [-0.034, +0.068] n=59 | 0.898 (+0.034) [+0.000, +0.085] n=59 | 0.915 (+0.051) [+0.000, +0.119] n=59 |
| olympiad | 49 | 0.898 | 0.837 | 0.918 (+0.020) [+0.000, +0.061] n=49 | 0.894 (-0.021) [-0.064, +0.000] n=47 | 0.898 (+0.000) [-0.061, +0.061] n=49 | 1.000 (+0.114) [+0.023, +0.205] n=44 |
| pooled | 223 | 0.892 | 0.857 | 0.901 (+0.009) [-0.013, +0.036] n=223 | 0.890 (-0.005) [-0.032, +0.023] n=219 | 0.906 (+0.013) [-0.009, +0.036] n=223 | 0.953 (+0.061) [+0.028, +0.098] n=214 |
 ### Fano lower bound on the modal-answer error from H(gold | modal answer) 
| layer | items | observed modal error | H(gold \| modal) bits | \|answers\| | Fano LB on error | LB / observed |
|---|---|---|---|---|---|---|
| mmlu-pro | 60 | 0.067 | 0.262 | 10 | 0.000 | 0.00 |
| gpqa-diamond | 55 | 0.127 | 0.619 | 4 | 0.000 | 0.00 |
 Letter-answer layers only (gsm-hard and olympiad gold strings are not normalised to the model output). Fano: P_e ≥ (H(Y|X) − 1)/log₂(|Y|−1); with H(Y|X) < 1 bit the bound is vacuous (0). 
## H6d. Ashby / I(D;R): persona variety vs plain draws (DeepSeek, 4 personas × 4 draws vs 4 plain draws) 
| layer | items | mean V plain@4 | mean V personas (1 draw each) | ΔV | oracle−MV plain@4 | oracle−MV personas | Δ | ICC across plain draws | ICC across personas | persona×item interaction share |
|---|---|---|---|---|---|---|---|---|---|---|
| mmlu-pro | 60 | 1.10 | 1.10 | +0.00 | 0.067 | 0.033 | -0.033 | 0.65 | 0.64 | 0.05 |
| gsm-hard | 59 | 1.29 | 1.31 | +0.02 | 0.068 | 0.051 | -0.017 | 0.67 | 0.80 | 0.12 |
| olympiad | 12 | 1.50 | 1.83 | +0.33 | 0.000 | 0.000 | +0.000 | 0.76 | 0.79 | 0.04 |
 Layers where personas' oracle gain exceeds plain by > 0.03: 0. Interaction share = SS(persona×item)/SS(total) of cell means over 4 personas × 4 draws (noise floor ≈ share expected from binomial draw noise, not subtracted). 
