Filter: weakest <= 0.80 and prize >= 5 pp.

| set | group | pool | N | weakest (model) | best (model) | oracle | prize, pp | passes |
|---|---|---|---|---|---|---|---|---|
| mmlu-pro | all | P6 | 553 | 0.857 (deepseek_deepseek-v3.2) | 0.960 (claude_opus) | 0.996 | 3.6 |  |
| mmlu-pro | biology | P6 | 41 | 0.951 (claude_haiku) | 0.976 (deepseek_deepseek-v3.2) | 1.000 | 2.4 |  |
| mmlu-pro | business | P6 | 40 | 0.850 (deepseek_deepseek-v3.2) | 0.975 (claude_sonnet55) | 1.000 | 2.5 |  |
| mmlu-pro | chemistry | P6 | 40 | 0.900 (z-ai_glm-5.3-flash) | 1.000 (claude_opus) | 1.000 | 0.0 |  |
| mmlu-pro | computer science | P6 | 41 | 0.878 (claude_haiku) | 1.000 (claude_opus) | 1.000 | 0.0 |  |
| mmlu-pro | economics | P6 | 39 | 0.897 (claude_haiku) | 0.974 (claude_opus) | 1.000 | 2.6 |  |
| mmlu-pro | engineering | P6 | 40 | 0.700 (z-ai_glm-5.3-flash) | 0.925 (claude_opus) | 0.975 | 5.0 |  |
| mmlu-pro | health | P6 | 38 | 0.895 (claude_haiku) | 1.000 (claude_opus) | 1.000 | 0.0 |  |
| mmlu-pro | history | P6 | 40 | 0.725 (claude_haiku) | 0.950 (claude_opus) | 0.975 | 2.5 |  |
| mmlu-pro | law | P6 | 37 | 0.568 (claude_haiku) | 0.946 (claude_opus) | 1.000 | 5.4 | yes |
| mmlu-pro | math | P6 | 42 | 0.976 (claude_haiku) | 1.000 (claude_opus) | 1.000 | 0.0 |  |
| mmlu-pro | other | P6 | 40 | 0.800 (claude_haiku) | 0.950 (claude_opus) | 1.000 | 5.0 | yes |
| mmlu-pro | philosophy | P6 | 34 | 0.676 (deepseek_deepseek-v3.2) | 0.912 (claude_opus) | 1.000 | 8.8 | yes |
| mmlu-pro | physics | P6 | 41 | 0.927 (claude_sonnet55) | 0.951 (claude_haiku) | 1.000 | 4.9 |  |
| mmlu-pro | psychology | P6 | 40 | 0.800 (deepseek_deepseek-v3.2) | 0.925 (claude_haiku) | 1.000 | 7.5 | yes |
| mmlu-pro | all | W3 | 553 | 0.857 (deepseek_deepseek-v3.2) | 0.866 (z-ai_glm-5.3-flash) | 0.944 | 7.8 |  |
| mmlu-pro | biology | W3 | 41 | 0.951 (claude_haiku) | 0.976 (deepseek_deepseek-v3.2) | 0.976 | 0.0 |  |
| mmlu-pro | business | W3 | 40 | 0.850 (deepseek_deepseek-v3.2) | 0.950 (claude_haiku) | 0.975 | 2.5 |  |
| mmlu-pro | chemistry | W3 | 40 | 0.900 (z-ai_glm-5.3-flash) | 0.950 (claude_haiku) | 0.950 | 0.0 |  |
| mmlu-pro | computer science | W3 | 41 | 0.878 (claude_haiku) | 0.927 (deepseek_deepseek-v3.2) | 0.976 | 4.9 |  |
| mmlu-pro | economics | W3 | 39 | 0.897 (claude_haiku) | 0.949 (deepseek_deepseek-v3.2) | 0.974 | 2.6 |  |
| mmlu-pro | engineering | W3 | 40 | 0.700 (z-ai_glm-5.3-flash) | 0.850 (deepseek_deepseek-v3.2) | 0.900 | 5.0 | yes |
| mmlu-pro | health | W3 | 38 | 0.895 (claude_haiku) | 0.921 (z-ai_glm-5.3-flash) | 0.947 | 2.6 |  |
| mmlu-pro | history | W3 | 40 | 0.725 (claude_haiku) | 0.775 (z-ai_glm-5.3-flash) | 0.875 | 10.0 | yes |
| mmlu-pro | law | W3 | 37 | 0.568 (claude_haiku) | 0.730 (z-ai_glm-5.3-flash) | 0.838 | 10.8 | yes |
| mmlu-pro | math | W3 | 42 | 0.976 (claude_haiku) | 1.000 (z-ai_glm-5.3-flash) | 1.000 | 0.0 |  |
| mmlu-pro | other | W3 | 40 | 0.800 (claude_haiku) | 0.875 (z-ai_glm-5.3-flash) | 0.925 | 5.0 | yes |
| mmlu-pro | philosophy | W3 | 34 | 0.676 (deepseek_deepseek-v3.2) | 0.824 (z-ai_glm-5.3-flash) | 0.941 | 11.8 | yes |
| mmlu-pro | physics | W3 | 41 | 0.927 (z-ai_glm-5.3-flash) | 0.951 (claude_haiku) | 0.951 | 0.0 |  |
| mmlu-pro | psychology | W3 | 40 | 0.800 (deepseek_deepseek-v3.2) | 0.925 (claude_haiku) | 0.975 | 5.0 |  |
| mmlu-pro | all | W2 | 553 | 0.857 (deepseek_deepseek-v3.2) | 0.864 (claude_haiku) | 0.913 | 4.9 |  |
| mmlu-pro | biology | W2 | 41 | 0.951 (claude_haiku) | 0.976 (deepseek_deepseek-v3.2) | 0.976 | 0.0 |  |
| mmlu-pro | business | W2 | 40 | 0.850 (deepseek_deepseek-v3.2) | 0.950 (claude_haiku) | 0.950 | 0.0 |  |
| mmlu-pro | chemistry | W2 | 40 | 0.925 (deepseek_deepseek-v3.2) | 0.950 (claude_haiku) | 0.950 | 0.0 |  |
| mmlu-pro | computer science | W2 | 41 | 0.878 (claude_haiku) | 0.927 (deepseek_deepseek-v3.2) | 0.976 | 4.9 |  |
| mmlu-pro | economics | W2 | 39 | 0.897 (claude_haiku) | 0.949 (deepseek_deepseek-v3.2) | 0.974 | 2.6 |  |
| mmlu-pro | engineering | W2 | 40 | 0.800 (claude_haiku) | 0.850 (deepseek_deepseek-v3.2) | 0.900 | 5.0 | yes |
| mmlu-pro | health | W2 | 38 | 0.895 (claude_haiku) | 0.895 (claude_haiku) | 0.921 | 2.6 |  |
| mmlu-pro | history | W2 | 40 | 0.725 (claude_haiku) | 0.750 (deepseek_deepseek-v3.2) | 0.800 | 5.0 | yes |
| mmlu-pro | law | W2 | 37 | 0.568 (claude_haiku) | 0.595 (deepseek_deepseek-v3.2) | 0.703 | 10.8 | yes |
| mmlu-pro | math | W2 | 42 | 0.976 (claude_haiku) | 0.976 (claude_haiku) | 0.976 | 0.0 |  |
| mmlu-pro | other | W2 | 40 | 0.800 (claude_haiku) | 0.825 (deepseek_deepseek-v3.2) | 0.900 | 7.5 | yes |
| mmlu-pro | philosophy | W2 | 34 | 0.676 (deepseek_deepseek-v3.2) | 0.794 (claude_haiku) | 0.853 | 5.9 | yes |
| mmlu-pro | physics | W2 | 41 | 0.951 (claude_haiku) | 0.951 (claude_haiku) | 0.951 | 0.0 |  |
| mmlu-pro | psychology | W2 | 40 | 0.800 (deepseek_deepseek-v3.2) | 0.925 (claude_haiku) | 0.925 | 0.0 |  |
| gpqa-diamond | all | P6 | 164 | 0.689 (claude_haiku) | 0.921 (claude_opus) | 0.982 | 6.1 | yes |
| gpqa-diamond | Biology | P6 | 15 | 0.533 (claude_haiku) | 0.800 (claude_opus) | 1.000 | 20.0 | yes |
| gpqa-diamond | Chemistry | P6 | 78 | 0.577 (claude_haiku) | 0.897 (claude_opus) | 0.974 | 7.7 | yes |
| gpqa-diamond | Physics | P6 | 71 | 0.845 (claude_haiku) | 0.972 (claude_opus) | 0.986 | 1.4 |  |
| gpqa-diamond | all | W3 | 164 | 0.689 (claude_haiku) | 0.738 (deepseek_deepseek-v3.2) | 0.902 | 16.5 | yes |
| gpqa-diamond | Biology | W3 | 15 | 0.533 (claude_haiku) | 0.667 (deepseek_deepseek-v3.2) | 0.933 | 26.7 | yes |
| gpqa-diamond | Chemistry | W3 | 78 | 0.577 (claude_haiku) | 0.603 (deepseek_deepseek-v3.2) | 0.833 | 23.1 | yes |
| gpqa-diamond | Physics | W3 | 71 | 0.845 (claude_haiku) | 0.915 (z-ai_glm-5.3-flash) | 0.972 | 5.6 |  |
| gpqa-diamond | all | W2 | 164 | 0.689 (claude_haiku) | 0.738 (deepseek_deepseek-v3.2) | 0.823 | 8.5 | yes |
| gpqa-diamond | Biology | W2 | 15 | 0.533 (claude_haiku) | 0.667 (deepseek_deepseek-v3.2) | 0.733 | 6.7 | yes |
| gpqa-diamond | Chemistry | W2 | 78 | 0.577 (claude_haiku) | 0.603 (deepseek_deepseek-v3.2) | 0.744 | 14.1 | yes |
| gpqa-diamond | Physics | W2 | 71 | 0.845 (claude_haiku) | 0.901 (deepseek_deepseek-v3.2) | 0.930 | 2.8 |  |
| gsm-hard | all | P6 | 141 | 0.922 (deepseek_deepseek-v3.2) | 0.986 (claude_sonnet55) | 1.000 | 1.4 |  |
| gsm-hard | all | W3 | 141 | 0.922 (deepseek_deepseek-v3.2) | 0.957 (claude_haiku) | 0.986 | 2.8 |  |
| gsm-hard | all | W2 | 141 | 0.922 (deepseek_deepseek-v3.2) | 0.957 (claude_haiku) | 0.979 | 2.1 |  |
| olympiad | all | P6 | 83 | 0.614 (deepseek_deepseek-v3.2) | 0.964 (claude_opus) | 1.000 | 3.6 |  |
| olympiad | aime24 | P6 | 27 | 0.815 (claude_haiku) | 1.000 (claude_opus) | 1.000 | 0.0 |  |
| olympiad | aime25 | P6 | 27 | 0.556 (deepseek_deepseek-v3.2) | 1.000 (claude_opus) | 1.000 | 0.0 |  |
| olympiad | hmmt25 | P6 | 29 | 0.448 (deepseek_deepseek-v3.2) | 0.897 (claude_opus) | 1.000 | 10.3 | yes |
| olympiad | all | W3 | 83 | 0.614 (deepseek_deepseek-v3.2) | 0.771 (z-ai_glm-5.3-flash) | 0.880 | 10.8 | yes |
| olympiad | aime24 | W3 | 27 | 0.815 (claude_haiku) | 0.926 (z-ai_glm-5.3-flash) | 0.963 | 3.7 |  |
| olympiad | aime25 | W3 | 27 | 0.556 (deepseek_deepseek-v3.2) | 0.741 (claude_haiku) | 0.852 | 11.1 | yes |
| olympiad | hmmt25 | W3 | 29 | 0.448 (deepseek_deepseek-v3.2) | 0.759 (z-ai_glm-5.3-flash) | 0.828 | 6.9 | yes |
| olympiad | all | W2 | 83 | 0.614 (deepseek_deepseek-v3.2) | 0.747 (claude_haiku) | 0.819 | 7.2 | yes |
| olympiad | aime24 | W2 | 27 | 0.815 (claude_haiku) | 0.852 (deepseek_deepseek-v3.2) | 0.926 | 7.4 |  |
| olympiad | aime25 | W2 | 27 | 0.556 (deepseek_deepseek-v3.2) | 0.741 (claude_haiku) | 0.778 | 3.7 |  |
| olympiad | hmmt25 | W2 | 29 | 0.448 (deepseek_deepseek-v3.2) | 0.690 (claude_haiku) | 0.759 | 6.9 | yes |
