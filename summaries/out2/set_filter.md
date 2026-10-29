Filter: weakest <= 0.80 and prize >= 5 pp.

| set | group | pool | N | weakest (model) | best (model) | oracle | prize, pp | passes |
|---|---|---|---|---|---|---|---|---|
| mmlu-pro | all | P7 | 553 | 0.819 (mistralai_mistral-large-2512) | 0.960 (claude_opus) | 0.998 | 3.8 |  |
| mmlu-pro | biology | P7 | 41 | 0.927 (mistralai_mistral-large-2512) | 0.976 (deepseek_deepseek-v3.2) | 1.000 | 2.4 |  |
| mmlu-pro | business | P7 | 40 | 0.825 (mistralai_mistral-large-2512) | 0.975 (claude_sonnet55) | 1.000 | 2.5 |  |
| mmlu-pro | chemistry | P7 | 40 | 0.900 (z-ai_glm-5.3-flash) | 1.000 (claude_opus) | 1.000 | 0.0 |  |
| mmlu-pro | computer science | P7 | 41 | 0.756 (mistralai_mistral-large-2512) | 1.000 (claude_opus) | 1.000 | 0.0 |  |
| mmlu-pro | economics | P7 | 39 | 0.897 (claude_haiku) | 0.974 (claude_opus) | 1.000 | 2.6 |  |
| mmlu-pro | engineering | P7 | 40 | 0.700 (z-ai_glm-5.3-flash) | 0.925 (claude_opus) | 1.000 | 7.5 | yes |
| mmlu-pro | health | P7 | 38 | 0.895 (claude_haiku) | 1.000 (claude_opus) | 1.000 | 0.0 |  |
| mmlu-pro | history | P7 | 40 | 0.725 (claude_haiku) | 0.950 (claude_opus) | 0.975 | 2.5 |  |
| mmlu-pro | law | P7 | 37 | 0.514 (mistralai_mistral-large-2512) | 0.946 (claude_opus) | 1.000 | 5.4 | yes |
| mmlu-pro | math | P7 | 42 | 0.929 (mistralai_mistral-large-2512) | 1.000 (claude_opus) | 1.000 | 0.0 |  |
| mmlu-pro | other | P7 | 40 | 0.800 (claude_haiku) | 0.950 (claude_opus) | 1.000 | 5.0 | yes |
| mmlu-pro | philosophy | P7 | 34 | 0.676 (deepseek_deepseek-v3.2) | 0.912 (claude_opus) | 1.000 | 8.8 | yes |
| mmlu-pro | physics | P7 | 41 | 0.878 (mistralai_mistral-large-2512) | 0.951 (claude_haiku) | 1.000 | 4.9 |  |
| mmlu-pro | psychology | P7 | 40 | 0.800 (deepseek_deepseek-v3.2) | 0.925 (claude_haiku) | 1.000 | 7.5 | yes |
| mmlu-pro | all | P4 | 553 | 0.819 (mistralai_mistral-large-2512) | 0.866 (z-ai_glm-5.3-flash) | 0.962 | 9.6 |  |
| mmlu-pro | biology | P4 | 41 | 0.927 (mistralai_mistral-large-2512) | 0.976 (deepseek_deepseek-v3.2) | 0.976 | 0.0 |  |
| mmlu-pro | business | P4 | 40 | 0.825 (mistralai_mistral-large-2512) | 0.950 (claude_haiku) | 0.975 | 2.5 |  |
| mmlu-pro | chemistry | P4 | 40 | 0.900 (z-ai_glm-5.3-flash) | 0.950 (claude_haiku) | 0.950 | 0.0 |  |
| mmlu-pro | computer science | P4 | 41 | 0.756 (mistralai_mistral-large-2512) | 0.927 (deepseek_deepseek-v3.2) | 1.000 | 7.3 | yes |
| mmlu-pro | economics | P4 | 39 | 0.897 (claude_haiku) | 0.949 (deepseek_deepseek-v3.2) | 0.974 | 2.6 |  |
| mmlu-pro | engineering | P4 | 40 | 0.700 (z-ai_glm-5.3-flash) | 0.850 (deepseek_deepseek-v3.2) | 0.950 | 10.0 | yes |
| mmlu-pro | health | P4 | 38 | 0.895 (claude_haiku) | 0.921 (z-ai_glm-5.3-flash) | 0.947 | 2.6 |  |
| mmlu-pro | history | P4 | 40 | 0.725 (claude_haiku) | 0.775 (z-ai_glm-5.3-flash) | 0.900 | 12.5 | yes |
| mmlu-pro | law | P4 | 37 | 0.514 (mistralai_mistral-large-2512) | 0.730 (z-ai_glm-5.3-flash) | 0.919 | 18.9 | yes |
| mmlu-pro | math | P4 | 42 | 0.929 (mistralai_mistral-large-2512) | 1.000 (z-ai_glm-5.3-flash) | 1.000 | 0.0 |  |
| mmlu-pro | other | P4 | 40 | 0.800 (claude_haiku) | 0.875 (z-ai_glm-5.3-flash) | 0.925 | 5.0 | yes |
| mmlu-pro | philosophy | P4 | 34 | 0.676 (deepseek_deepseek-v3.2) | 0.824 (z-ai_glm-5.3-flash) | 0.971 | 14.7 | yes |
| mmlu-pro | physics | P4 | 41 | 0.878 (mistralai_mistral-large-2512) | 0.951 (claude_haiku) | 1.000 | 4.9 |  |
| mmlu-pro | psychology | P4 | 40 | 0.800 (deepseek_deepseek-v3.2) | 0.925 (claude_haiku) | 0.975 | 5.0 |  |
| mmlu-pro | all | P3 | 553 | 0.819 (mistralai_mistral-large-2512) | 0.864 (claude_haiku) | 0.942 | 7.8 |  |
| mmlu-pro | biology | P3 | 41 | 0.927 (mistralai_mistral-large-2512) | 0.976 (deepseek_deepseek-v3.2) | 0.976 | 0.0 |  |
| mmlu-pro | business | P3 | 40 | 0.825 (mistralai_mistral-large-2512) | 0.950 (claude_haiku) | 0.950 | 0.0 |  |
| mmlu-pro | chemistry | P3 | 40 | 0.925 (deepseek_deepseek-v3.2) | 0.950 (claude_haiku) | 0.950 | 0.0 |  |
| mmlu-pro | computer science | P3 | 41 | 0.756 (mistralai_mistral-large-2512) | 0.927 (deepseek_deepseek-v3.2) | 1.000 | 7.3 | yes |
| mmlu-pro | economics | P3 | 39 | 0.897 (claude_haiku) | 0.949 (deepseek_deepseek-v3.2) | 0.974 | 2.6 |  |
| mmlu-pro | engineering | P3 | 40 | 0.800 (claude_haiku) | 0.850 (deepseek_deepseek-v3.2) | 0.950 | 10.0 | yes |
| mmlu-pro | health | P3 | 38 | 0.895 (claude_haiku) | 0.895 (claude_haiku) | 0.921 | 2.6 |  |
| mmlu-pro | history | P3 | 40 | 0.725 (claude_haiku) | 0.750 (deepseek_deepseek-v3.2) | 0.875 | 12.5 | yes |
| mmlu-pro | law | P3 | 37 | 0.514 (mistralai_mistral-large-2512) | 0.595 (deepseek_deepseek-v3.2) | 0.811 | 21.6 | yes |
| mmlu-pro | math | P3 | 42 | 0.929 (mistralai_mistral-large-2512) | 0.976 (claude_haiku) | 1.000 | 2.4 |  |
| mmlu-pro | other | P3 | 40 | 0.800 (claude_haiku) | 0.825 (deepseek_deepseek-v3.2) | 0.925 | 10.0 | yes |
| mmlu-pro | philosophy | P3 | 34 | 0.676 (deepseek_deepseek-v3.2) | 0.794 (claude_haiku) | 0.912 | 11.8 | yes |
| mmlu-pro | physics | P3 | 41 | 0.878 (mistralai_mistral-large-2512) | 0.951 (claude_haiku) | 1.000 | 4.9 |  |
| mmlu-pro | psychology | P3 | 40 | 0.800 (deepseek_deepseek-v3.2) | 0.925 (claude_haiku) | 0.925 | 0.0 |  |
| gpqa-diamond | all | P7 | 164 | 0.634 (mistralai_mistral-large-2512) | 0.921 (claude_opus) | 0.988 | 6.7 | yes |
| gpqa-diamond | Biology | P7 | 15 | 0.533 (claude_haiku) | 0.800 (claude_opus) | 1.000 | 20.0 | yes |
| gpqa-diamond | Chemistry | P7 | 78 | 0.449 (mistralai_mistral-large-2512) | 0.897 (claude_opus) | 0.974 | 7.7 | yes |
| gpqa-diamond | Physics | P7 | 71 | 0.817 (mistralai_mistral-large-2512) | 0.972 (claude_opus) | 1.000 | 2.8 |  |
| gpqa-diamond | all | P4 | 164 | 0.634 (mistralai_mistral-large-2512) | 0.738 (deepseek_deepseek-v3.2) | 0.933 | 19.5 | yes |
| gpqa-diamond | Biology | P4 | 15 | 0.533 (claude_haiku) | 0.733 (mistralai_mistral-large-2512) | 1.000 | 26.7 | yes |
| gpqa-diamond | Chemistry | P4 | 78 | 0.449 (mistralai_mistral-large-2512) | 0.603 (deepseek_deepseek-v3.2) | 0.872 | 26.9 | yes |
| gpqa-diamond | Physics | P4 | 71 | 0.817 (mistralai_mistral-large-2512) | 0.915 (z-ai_glm-5.3-flash) | 0.986 | 7.0 |  |
| gpqa-diamond | all | P3 | 164 | 0.634 (mistralai_mistral-large-2512) | 0.738 (deepseek_deepseek-v3.2) | 0.872 | 13.4 | yes |
| gpqa-diamond | Biology | P3 | 15 | 0.533 (claude_haiku) | 0.733 (mistralai_mistral-large-2512) | 0.800 | 6.7 | yes |
| gpqa-diamond | Chemistry | P3 | 78 | 0.449 (mistralai_mistral-large-2512) | 0.603 (deepseek_deepseek-v3.2) | 0.808 | 20.5 | yes |
| gpqa-diamond | Physics | P3 | 71 | 0.817 (mistralai_mistral-large-2512) | 0.901 (deepseek_deepseek-v3.2) | 0.958 | 5.6 |  |
| gsm-hard | all | P7 | 141 | 0.872 (mistralai_mistral-large-2512) | 0.986 (claude_sonnet55) | 1.000 | 1.4 |  |
| gsm-hard | all | P4 | 141 | 0.872 (mistralai_mistral-large-2512) | 0.957 (claude_haiku) | 0.993 | 3.5 |  |
| gsm-hard | all | P3 | 141 | 0.872 (mistralai_mistral-large-2512) | 0.957 (claude_haiku) | 0.986 | 2.8 |  |
| olympiad | all | P7 | 83 | 0.470 (mistralai_mistral-large-2512) | 0.964 (claude_opus) | 1.000 | 3.6 |  |
| olympiad | aime24 | P7 | 27 | 0.630 (mistralai_mistral-large-2512) | 1.000 (claude_opus) | 1.000 | 0.0 |  |
| olympiad | aime25 | P7 | 27 | 0.556 (deepseek_deepseek-v3.2) | 1.000 (claude_opus) | 1.000 | 0.0 |  |
| olympiad | hmmt25 | P7 | 29 | 0.241 (mistralai_mistral-large-2512) | 0.897 (claude_opus) | 1.000 | 10.3 | yes |
| olympiad | all | P4 | 83 | 0.470 (mistralai_mistral-large-2512) | 0.771 (z-ai_glm-5.3-flash) | 0.880 | 10.8 | yes |
| olympiad | aime24 | P4 | 27 | 0.630 (mistralai_mistral-large-2512) | 0.926 (z-ai_glm-5.3-flash) | 0.963 | 3.7 |  |
| olympiad | aime25 | P4 | 27 | 0.556 (deepseek_deepseek-v3.2) | 0.741 (claude_haiku) | 0.852 | 11.1 | yes |
| olympiad | hmmt25 | P4 | 29 | 0.241 (mistralai_mistral-large-2512) | 0.759 (z-ai_glm-5.3-flash) | 0.828 | 6.9 | yes |
| olympiad | all | P3 | 83 | 0.470 (mistralai_mistral-large-2512) | 0.747 (claude_haiku) | 0.831 | 8.4 | yes |
| olympiad | aime24 | P3 | 27 | 0.630 (mistralai_mistral-large-2512) | 0.852 (deepseek_deepseek-v3.2) | 0.926 | 7.4 | yes |
| olympiad | aime25 | P3 | 27 | 0.556 (deepseek_deepseek-v3.2) | 0.741 (claude_haiku) | 0.815 | 7.4 | yes |
| olympiad | hmmt25 | P3 | 29 | 0.241 (mistralai_mistral-large-2512) | 0.690 (claude_haiku) | 0.759 | 6.9 | yes |
