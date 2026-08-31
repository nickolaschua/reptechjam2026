# Messy benchmark report

298 scored cases. `*` = n < 20.

## First-question quality (`question_hit`)

| slice | hit rate |
|---|---|
| all=all | 0.537 [0.477, 0.594] n=298 |
| style=compatibility | 0.714 [0.429, 1.000] n=7 * |
| style=exact | 0.667 [0.333, 1.000] n=6 * |
| style=feature | 0.425 [0.275, 0.575] n=40 |
| style=lay | 0.604 [0.479, 0.750] n=48 |
| style=plain | 0.556 [0.426, 0.685] n=54 |
| style=product_type | 0.560 [0.420, 0.700] n=50 |
| style=symptom | 0.488 [0.349, 0.628] n=43 |
| style=use_case | 0.520 [0.380, 0.660] n=50 |
| intent=browsing | 0.534 [0.476, 0.589] n=292 |
| intent=buying | 0.667 [0.333, 1.000] n=6 * |

## template_rank

### Overall — `template_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| all | 0.081 [0.050, 0.111] n=298 | 0.034 [0.018, 0.052] n=298 |

### By style — `template_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| compatibility | 0.000 [0.000, 0.000] n=7 * | 0.000 [0.000, 0.000] n=7 * |
| exact | 0.167 [0.000, 0.500] n=6 * | 0.167 [0.000, 0.500] n=6 * |
| feature | 0.175 [0.075, 0.300] n=40 | 0.141 [0.048, 0.245] n=40 |
| lay | 0.000 [0.000, 0.000] n=48 | 0.000 [0.000, 0.000] n=48 |
| plain | 0.111 [0.037, 0.204] n=54 | 0.026 [0.007, 0.051] n=54 |
| product_type | 0.180 [0.080, 0.300] n=50 | 0.041 [0.015, 0.071] n=50 |
| symptom | 0.000 [0.000, 0.000] n=43 | 0.000 [0.000, 0.000] n=43 |
| use_case | 0.020 [0.000, 0.060] n=50 | 0.004 [0.000, 0.012] n=50 |

### By intent label (style prior) — `template_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| browsing | 0.079 [0.048, 0.110] n=292 | 0.032 [0.017, 0.050] n=292 |
| buying | 0.167 [0.000, 0.500] n=6 * | 0.167 [0.000, 0.500] n=6 * |

### By card hard-constraints voiced — `template_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| card_hard_said=0 | 0.044 [0.011, 0.088] n=91 | 0.010 [0.001, 0.024] n=91 |
| card_hard_said=1 | 0.063 [0.021, 0.116] n=95 | 0.013 [0.003, 0.024] n=95 |
| card_hard_said=2+ | 0.125 [0.062, 0.196] n=112 | 0.072 [0.032, 0.120] n=112 |

### By parsed hard slots — `template_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| n_hard=0 | 0.066 [0.035, 0.101] n=198 | 0.030 [0.011, 0.053] n=198 |
| n_hard=1 | 0.128 [0.070, 0.198] n=86 | 0.050 [0.017, 0.091] n=86 |
| n_hard=2+ | 0.000 [0.000, 0.000] n=14 * | 0.000 [0.000, 0.000] n=14 * |

### By resolver confidence — `template_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| conf <0.2 | 0.071 [0.040, 0.107] n=225 | 0.020 [0.010, 0.033] n=225 |
| conf >=0.2 | 0.110 [0.041, 0.192] n=73 | 0.079 [0.027, 0.142] n=73 |

### By generator — `template_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| gemma2:9b | 0.102 [0.064, 0.140] n=235 | 0.044 [0.023, 0.068] n=235 |
| llama3.1:8b | 0.000 [0.000, 0.000] n=63 | 0.000 [0.000, 0.000] n=63 |

### By modifier negation — `template_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| negation=False | 0.081 [0.047, 0.119] n=236 | 0.040 [0.020, 0.064] n=236 |
| negation=True | 0.081 [0.016, 0.161] n=62 | 0.013 [0.003, 0.026] n=62 |

### By modifier for_other — `template_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| for_other=False | 0.083 [0.048, 0.118] n=229 | 0.038 [0.018, 0.061] n=229 |
| for_other=True | 0.072 [0.014, 0.145] n=69 | 0.022 [0.004, 0.047] n=69 |

### By modifier vague_budget — `template_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| vague_budget=False | 0.084 [0.048, 0.120] n=249 | 0.038 [0.020, 0.061] n=249 |
| vague_budget=True | 0.061 [0.000, 0.143] n=49 | 0.014 [0.000, 0.033] n=49 |

### By modifier format_noise — `template_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| format_noise=False | 0.082 [0.051, 0.112] n=294 | 0.035 [0.019, 0.053] n=294 |
| format_noise=True | 0.000 [0.000, 0.000] n=4 * | 0.000 [0.000, 0.000] n=4 * |

### By listing overlap quartile — `template_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| overlap Q1 | 0.000 [0.000, 0.000] n=74 | 0.000 [0.000, 0.000] n=74 |
| overlap Q2 | 0.000 [0.000, 0.000] n=75 | 0.000 [0.000, 0.000] n=75 |
| overlap Q3 | 0.072 [0.014, 0.145] n=69 | 0.011 [0.003, 0.023] n=69 |
| overlap Q4 | 0.237 [0.150, 0.338] n=80 | 0.119 [0.061, 0.180] n=80 |

### By descriptiveness quartile — `template_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| descriptiveness Q1 | 0.041 [0.000, 0.096] n=73 | 0.008 [0.000, 0.018] n=73 |
| descriptiveness Q2 | 0.080 [0.027, 0.147] n=75 | 0.032 [0.005, 0.070] n=75 |
| descriptiveness Q3 | 0.053 [0.013, 0.107] n=75 | 0.034 [0.003, 0.074] n=75 |
| descriptiveness Q4 | 0.147 [0.080, 0.240] n=75 | 0.063 [0.022, 0.115] n=75 |

### By title_richness quartile — `template_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| title_richness Q1 | 0.041 [0.000, 0.095] n=74 | 0.010 [0.000, 0.027] n=74 |
| title_richness Q2 | 0.054 [0.014, 0.108] n=74 | 0.023 [0.002, 0.056] n=74 |
| title_richness Q3 | 0.080 [0.027, 0.147] n=75 | 0.037 [0.006, 0.079] n=75 |
| title_richness Q4 | 0.147 [0.067, 0.227] n=75 | 0.067 [0.025, 0.123] n=75 |

### By jargon quartile — `template_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| jargon Q1 | 0.108 [0.041, 0.176] n=74 | 0.046 [0.012, 0.090] n=74 |
| jargon Q2 | 0.068 [0.014, 0.122] n=74 | 0.035 [0.003, 0.077] n=74 |
| jargon Q3 | 0.068 [0.014, 0.137] n=73 | 0.023 [0.003, 0.055] n=73 |
| jargon Q4 | 0.078 [0.026, 0.143] n=77 | 0.033 [0.006, 0.069] n=77 |

### By bucket_size quartile — `template_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| bucket_size Q1 | 0.108 [0.041, 0.176] n=74 | 0.057 [0.014, 0.108] n=74 |
| bucket_size Q2 | 0.070 [0.014, 0.127] n=71 | 0.044 [0.007, 0.092] n=71 |
| bucket_size Q3 | 0.130 [0.065, 0.208] n=77 | 0.037 [0.012, 0.074] n=77 |
| bucket_size Q4 | 0.013 [0.000, 0.039] n=76 | 0.001 [0.000, 0.004] n=76 |

### By popularity quartile — `template_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| popularity Q1 | 0.070 [0.018, 0.140] n=57 | 0.022 [0.002, 0.051] n=57 |
| popularity Q2 | 0.055 [0.011, 0.110] n=91 | 0.014 [0.002, 0.030] n=91 |
| popularity Q3 | 0.120 [0.053, 0.200] n=75 | 0.077 [0.023, 0.139] n=75 |
| popularity Q4 | 0.080 [0.027, 0.147] n=75 | 0.026 [0.005, 0.058] n=75 |

### By silent_on_material — `template_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| silent_on_material=False | 0.053 [0.023, 0.091] n=132 | 0.024 [0.005, 0.049] n=132 |
| silent_on_material=True | 0.102 [0.060, 0.151] n=166 | 0.042 [0.019, 0.070] n=166 |

### By has_near_duplicate — `template_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| has_near_duplicate=False | 0.091 [0.055, 0.130] n=254 | 0.040 [0.021, 0.062] n=254 |
| has_near_duplicate=True | 0.023 [0.000, 0.068] n=44 | 0.005 [0.000, 0.014] n=44 |

### By has_model_code — `template_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| has_model_code=False | 0.081 [0.049, 0.114] n=246 | 0.030 [0.014, 0.048] n=246 |
| has_model_code=True | 0.077 [0.019, 0.154] n=52 | 0.054 [0.006, 0.122] n=52 |

### By compat_eligible — `template_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| compat_eligible=False | 0.064 [0.036, 0.096] n=251 | 0.023 [0.010, 0.039] n=251 |
| compat_eligible=True | 0.170 [0.064, 0.298] n=47 | 0.097 [0.029, 0.183] n=47 |

### By promo_bucket — `template_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| promo_bucket=False | 0.079 [0.047, 0.111] n=279 | 0.035 [0.018, 0.056] n=279 |
| promo_bucket=True | 0.105 [0.000, 0.263] n=19 * | 0.023 [0.000, 0.064] n=19 * |

### By price_present — `template_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| price_present=False | 0.064 [0.034, 0.098] n=235 | 0.026 [0.011, 0.046] n=235 |
| price_present=True | 0.143 [0.063, 0.238] n=63 | 0.065 [0.017, 0.126] n=63 |

## lexical_rank

### Overall — `lexical_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| all | 0.161 [0.121, 0.201] n=298 | 0.079 [0.053, 0.106] n=298 |

### By style — `lexical_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| compatibility | 0.429 [0.143, 0.857] n=7 * | 0.314 [0.029, 0.657] n=7 * |
| exact | 0.333 [0.000, 0.667] n=6 * | 0.333 [0.000, 0.667] n=6 * |
| feature | 0.300 [0.175, 0.450] n=40 | 0.186 [0.086, 0.300] n=40 |
| lay | 0.000 [0.000, 0.000] n=48 | 0.000 [0.000, 0.000] n=48 |
| plain | 0.259 [0.148, 0.370] n=54 | 0.108 [0.048, 0.179] n=54 |
| product_type | 0.320 [0.200, 0.440] n=50 | 0.114 [0.053, 0.189] n=50 |
| symptom | 0.000 [0.000, 0.000] n=43 | 0.000 [0.000, 0.000] n=43 |
| use_case | 0.020 [0.000, 0.060] n=50 | 0.004 [0.000, 0.012] n=50 |

### By intent label (style prior) — `lexical_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| browsing | 0.158 [0.116, 0.202] n=292 | 0.073 [0.049, 0.101] n=292 |
| buying | 0.333 [0.000, 0.667] n=6 * | 0.333 [0.000, 0.667] n=6 * |

### By card hard-constraints voiced — `lexical_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| card_hard_said=0 | 0.055 [0.011, 0.110] n=91 | 0.021 [0.002, 0.050] n=91 |
| card_hard_said=1 | 0.126 [0.063, 0.200] n=95 | 0.061 [0.022, 0.110] n=95 |
| card_hard_said=2+ | 0.277 [0.196, 0.366] n=112 | 0.140 [0.086, 0.197] n=112 |

### By parsed hard slots — `lexical_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| n_hard=0 | 0.131 [0.086, 0.177] n=198 | 0.068 [0.038, 0.101] n=198 |
| n_hard=1 | 0.233 [0.151, 0.314] n=86 | 0.108 [0.058, 0.166] n=86 |
| n_hard=2+ | 0.143 [0.000, 0.357] n=14 * | 0.050 [0.000, 0.136] n=14 * |

### By resolver confidence — `lexical_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| conf <0.2 | 0.138 [0.093, 0.182] n=225 | 0.059 [0.035, 0.086] n=225 |
| conf >=0.2 | 0.233 [0.137, 0.329] n=73 | 0.138 [0.069, 0.212] n=73 |

### By generator — `lexical_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| gemma2:9b | 0.153 [0.111, 0.200] n=235 | 0.069 [0.045, 0.099] n=235 |
| llama3.1:8b | 0.190 [0.095, 0.302] n=63 | 0.114 [0.050, 0.193] n=63 |

### By modifier negation — `lexical_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| negation=False | 0.169 [0.123, 0.216] n=236 | 0.088 [0.057, 0.121] n=236 |
| negation=True | 0.129 [0.048, 0.226] n=62 | 0.043 [0.012, 0.085] n=62 |

### By modifier for_other — `lexical_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| for_other=False | 0.175 [0.122, 0.223] n=229 | 0.083 [0.053, 0.114] n=229 |
| for_other=True | 0.116 [0.043, 0.188] n=69 | 0.065 [0.020, 0.120] n=69 |

### By modifier vague_budget — `lexical_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| vague_budget=False | 0.173 [0.129, 0.221] n=249 | 0.086 [0.058, 0.117] n=249 |
| vague_budget=True | 0.102 [0.020, 0.184] n=49 | 0.038 [0.005, 0.088] n=49 |

### By modifier format_noise — `lexical_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| format_noise=False | 0.160 [0.119, 0.201] n=294 | 0.076 [0.051, 0.102] n=294 |
| format_noise=True | 0.250 [0.000, 0.750] n=4 * | 0.250 [0.000, 0.750] n=4 * |

### By listing overlap quartile — `lexical_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| overlap Q1 | 0.000 [0.000, 0.000] n=74 | 0.000 [0.000, 0.000] n=74 |
| overlap Q2 | 0.000 [0.000, 0.000] n=75 | 0.000 [0.000, 0.000] n=75 |
| overlap Q3 | 0.130 [0.058, 0.217] n=69 | 0.047 [0.012, 0.096] n=69 |
| overlap Q4 | 0.487 [0.375, 0.600] n=80 | 0.252 [0.171, 0.334] n=80 |

### By descriptiveness quartile — `lexical_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| descriptiveness Q1 | 0.068 [0.014, 0.123] n=73 | 0.024 [0.003, 0.057] n=73 |
| descriptiveness Q2 | 0.147 [0.067, 0.227] n=75 | 0.054 [0.020, 0.100] n=75 |
| descriptiveness Q3 | 0.133 [0.067, 0.213] n=75 | 0.086 [0.034, 0.149] n=75 |
| descriptiveness Q4 | 0.293 [0.200, 0.400] n=75 | 0.148 [0.083, 0.222] n=75 |

### By title_richness quartile — `lexical_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| title_richness Q1 | 0.122 [0.054, 0.203] n=74 | 0.058 [0.017, 0.109] n=74 |
| title_richness Q2 | 0.122 [0.054, 0.203] n=74 | 0.049 [0.016, 0.093] n=74 |
| title_richness Q3 | 0.147 [0.067, 0.227] n=75 | 0.050 [0.017, 0.095] n=75 |
| title_richness Q4 | 0.253 [0.160, 0.347] n=75 | 0.156 [0.084, 0.233] n=75 |

### By jargon quartile — `lexical_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| jargon Q1 | 0.162 [0.081, 0.257] n=74 | 0.080 [0.030, 0.141] n=74 |
| jargon Q2 | 0.189 [0.108, 0.284] n=74 | 0.102 [0.046, 0.173] n=74 |
| jargon Q3 | 0.164 [0.082, 0.260] n=73 | 0.063 [0.023, 0.117] n=73 |
| jargon Q4 | 0.130 [0.052, 0.208] n=77 | 0.068 [0.025, 0.123] n=77 |

### By bucket_size quartile — `lexical_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| bucket_size Q1 | 0.189 [0.108, 0.284] n=74 | 0.118 [0.052, 0.188] n=74 |
| bucket_size Q2 | 0.197 [0.113, 0.296] n=71 | 0.097 [0.043, 0.160] n=71 |
| bucket_size Q3 | 0.182 [0.104, 0.273] n=77 | 0.078 [0.033, 0.133] n=77 |
| bucket_size Q4 | 0.079 [0.026, 0.145] n=76 | 0.024 [0.005, 0.055] n=76 |

### By popularity quartile — `lexical_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| popularity Q1 | 0.211 [0.105, 0.333] n=57 | 0.078 [0.029, 0.136] n=57 |
| popularity Q2 | 0.121 [0.055, 0.198] n=91 | 0.046 [0.016, 0.086] n=91 |
| popularity Q3 | 0.200 [0.107, 0.293] n=75 | 0.128 [0.064, 0.201] n=75 |
| popularity Q4 | 0.133 [0.067, 0.213] n=75 | 0.069 [0.023, 0.123] n=75 |

### By silent_on_material — `lexical_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| silent_on_material=False | 0.121 [0.068, 0.182] n=132 | 0.064 [0.031, 0.104] n=132 |
| silent_on_material=True | 0.193 [0.133, 0.253] n=166 | 0.090 [0.054, 0.130] n=166 |

### By has_near_duplicate — `lexical_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| has_near_duplicate=False | 0.181 [0.138, 0.228] n=254 | 0.091 [0.062, 0.123] n=254 |
| has_near_duplicate=True | 0.045 [0.000, 0.114] n=44 | 0.007 [0.000, 0.018] n=44 |

### By has_model_code — `lexical_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| has_model_code=False | 0.163 [0.118, 0.211] n=246 | 0.078 [0.051, 0.109] n=246 |
| has_model_code=True | 0.154 [0.058, 0.269] n=52 | 0.083 [0.023, 0.159] n=52 |

### By compat_eligible — `lexical_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| compat_eligible=False | 0.131 [0.088, 0.175] n=251 | 0.058 [0.035, 0.084] n=251 |
| compat_eligible=True | 0.319 [0.191, 0.447] n=47 | 0.185 [0.098, 0.296] n=47 |

### By promo_bucket — `lexical_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| promo_bucket=False | 0.158 [0.115, 0.204] n=279 | 0.081 [0.052, 0.113] n=279 |
| promo_bucket=True | 0.211 [0.053, 0.421] n=19 * | 0.047 [0.011, 0.096] n=19 * |

### By price_present — `lexical_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| price_present=False | 0.136 [0.098, 0.183] n=235 | 0.061 [0.036, 0.090] n=235 |
| price_present=True | 0.254 [0.143, 0.365] n=63 | 0.144 [0.071, 0.230] n=63 |

## parsed_rank

### Overall — `parsed_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| all | 0.225 [0.178, 0.272] n=298 | 0.124 [0.093, 0.158] n=298 |

### By style — `parsed_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| compatibility | 0.286 [0.000, 0.571] n=7 * | 0.163 [0.000, 0.449] n=7 * |
| exact | 0.667 [0.333, 1.000] n=6 * | 0.413 [0.079, 0.778] n=6 * |
| feature | 0.500 [0.350, 0.650] n=40 | 0.261 [0.156, 0.374] n=40 |
| lay | 0.021 [0.000, 0.062] n=48 | 0.007 [0.000, 0.021] n=48 |
| plain | 0.315 [0.204, 0.444] n=54 | 0.210 [0.115, 0.315] n=54 |
| product_type | 0.380 [0.240, 0.520] n=50 | 0.176 [0.098, 0.268] n=50 |
| symptom | 0.047 [0.000, 0.116] n=43 | 0.026 [0.000, 0.076] n=43 |
| use_case | 0.040 [0.000, 0.100] n=50 | 0.025 [0.000, 0.080] n=50 |

### By intent label (style prior) — `parsed_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| browsing | 0.216 [0.168, 0.264] n=292 | 0.118 [0.087, 0.152] n=292 |
| buying | 0.667 [0.333, 1.000] n=6 * | 0.413 [0.079, 0.778] n=6 * |

### By card hard-constraints voiced — `parsed_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| card_hard_said=0 | 0.055 [0.011, 0.099] n=91 | 0.020 [0.002, 0.046] n=91 |
| card_hard_said=1 | 0.221 [0.137, 0.305] n=95 | 0.107 [0.058, 0.167] n=95 |
| card_hard_said=2+ | 0.366 [0.277, 0.455] n=112 | 0.223 [0.158, 0.292] n=112 |

### By parsed hard slots — `parsed_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| n_hard=0 | 0.177 [0.126, 0.232] n=198 | 0.093 [0.060, 0.130] n=198 |
| n_hard=1 | 0.326 [0.233, 0.430] n=86 | 0.181 [0.113, 0.251] n=86 |
| n_hard=2+ | 0.286 [0.071, 0.571] n=14 * | 0.214 [0.036, 0.429] n=14 * |

### By resolver confidence — `parsed_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| conf <0.2 | 0.187 [0.138, 0.236] n=225 | 0.101 [0.068, 0.135] n=225 |
| conf >=0.2 | 0.342 [0.233, 0.452] n=73 | 0.194 [0.121, 0.270] n=73 |

### By generator — `parsed_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| gemma2:9b | 0.213 [0.162, 0.264] n=235 | 0.111 [0.077, 0.147] n=235 |
| llama3.1:8b | 0.270 [0.159, 0.381] n=63 | 0.172 [0.092, 0.257] n=63 |

### By modifier negation — `parsed_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| negation=False | 0.233 [0.182, 0.284] n=236 | 0.133 [0.095, 0.173] n=236 |
| negation=True | 0.194 [0.097, 0.290] n=62 | 0.090 [0.036, 0.154] n=62 |

### By modifier for_other — `parsed_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| for_other=False | 0.218 [0.166, 0.271] n=229 | 0.133 [0.093, 0.169] n=229 |
| for_other=True | 0.246 [0.145, 0.348] n=69 | 0.095 [0.045, 0.159] n=69 |

### By modifier vague_budget — `parsed_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| vague_budget=False | 0.225 [0.177, 0.281] n=249 | 0.125 [0.090, 0.164] n=249 |
| vague_budget=True | 0.224 [0.102, 0.347] n=49 | 0.117 [0.046, 0.201] n=49 |

### By modifier format_noise — `parsed_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| format_noise=False | 0.221 [0.173, 0.269] n=294 | 0.122 [0.091, 0.156] n=294 |
| format_noise=True | 0.500 [0.000, 1.000] n=4 * | 0.286 [0.000, 0.750] n=4 * |

### By listing overlap quartile — `parsed_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| overlap Q1 | 0.014 [0.000, 0.041] n=74 | 0.014 [0.000, 0.041] n=74 |
| overlap Q2 | 0.053 [0.013, 0.107] n=75 | 0.012 [0.001, 0.026] n=75 |
| overlap Q3 | 0.232 [0.130, 0.333] n=69 | 0.102 [0.049, 0.165] n=69 |
| overlap Q4 | 0.575 [0.463, 0.688] n=80 | 0.351 [0.261, 0.446] n=80 |

### By descriptiveness quartile — `parsed_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| descriptiveness Q1 | 0.123 [0.055, 0.205] n=73 | 0.052 [0.016, 0.096] n=73 |
| descriptiveness Q2 | 0.213 [0.120, 0.307] n=75 | 0.106 [0.051, 0.171] n=75 |
| descriptiveness Q3 | 0.213 [0.133, 0.307] n=75 | 0.140 [0.074, 0.221] n=75 |
| descriptiveness Q4 | 0.347 [0.240, 0.453] n=75 | 0.196 [0.121, 0.275] n=75 |

### By title_richness quartile — `parsed_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| title_richness Q1 | 0.149 [0.068, 0.230] n=74 | 0.050 [0.016, 0.091] n=74 |
| title_richness Q2 | 0.203 [0.122, 0.297] n=74 | 0.098 [0.047, 0.156] n=74 |
| title_richness Q3 | 0.253 [0.160, 0.347] n=75 | 0.159 [0.088, 0.238] n=75 |
| title_richness Q4 | 0.293 [0.187, 0.400] n=75 | 0.187 [0.112, 0.273] n=75 |

### By jargon quartile — `parsed_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| jargon Q1 | 0.203 [0.108, 0.297] n=74 | 0.083 [0.035, 0.142] n=74 |
| jargon Q2 | 0.311 [0.216, 0.419] n=74 | 0.204 [0.129, 0.294] n=74 |
| jargon Q3 | 0.192 [0.110, 0.288] n=73 | 0.097 [0.045, 0.161] n=73 |
| jargon Q4 | 0.195 [0.104, 0.286] n=77 | 0.112 [0.055, 0.177] n=77 |

### By bucket_size quartile — `parsed_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| bucket_size Q1 | 0.297 [0.189, 0.405] n=74 | 0.189 [0.115, 0.269] n=74 |
| bucket_size Q2 | 0.239 [0.141, 0.352] n=71 | 0.121 [0.061, 0.194] n=71 |
| bucket_size Q3 | 0.234 [0.143, 0.325] n=77 | 0.124 [0.063, 0.197] n=77 |
| bucket_size Q4 | 0.132 [0.066, 0.211] n=76 | 0.064 [0.022, 0.117] n=76 |

### By popularity quartile — `parsed_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| popularity Q1 | 0.246 [0.140, 0.368] n=57 | 0.114 [0.050, 0.192] n=57 |
| popularity Q2 | 0.176 [0.099, 0.253] n=91 | 0.089 [0.044, 0.142] n=91 |
| popularity Q3 | 0.267 [0.173, 0.373] n=75 | 0.179 [0.104, 0.264] n=75 |
| popularity Q4 | 0.227 [0.133, 0.320] n=75 | 0.120 [0.062, 0.187] n=75 |

### By silent_on_material — `parsed_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| silent_on_material=False | 0.167 [0.106, 0.227] n=132 | 0.092 [0.052, 0.134] n=132 |
| silent_on_material=True | 0.271 [0.205, 0.337] n=166 | 0.149 [0.104, 0.199] n=166 |

### By has_near_duplicate — `parsed_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| has_near_duplicate=False | 0.252 [0.201, 0.303] n=254 | 0.141 [0.105, 0.180] n=254 |
| has_near_duplicate=True | 0.068 [0.000, 0.159] n=44 | 0.025 [0.000, 0.057] n=44 |

### By has_model_code — `parsed_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| has_model_code=False | 0.220 [0.171, 0.268] n=246 | 0.121 [0.086, 0.160] n=246 |
| has_model_code=True | 0.250 [0.135, 0.385] n=52 | 0.138 [0.064, 0.231] n=52 |

### By compat_eligible — `parsed_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| compat_eligible=False | 0.191 [0.143, 0.243] n=251 | 0.113 [0.080, 0.147] n=251 |
| compat_eligible=True | 0.404 [0.255, 0.532] n=47 | 0.183 [0.094, 0.284] n=47 |

### By promo_bucket — `parsed_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| promo_bucket=False | 0.229 [0.179, 0.283] n=279 | 0.124 [0.091, 0.159] n=279 |
| promo_bucket=True | 0.158 [0.000, 0.316] n=19 * | 0.123 [0.000, 0.281] n=19 * |

### By price_present — `parsed_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| price_present=False | 0.183 [0.136, 0.238] n=235 | 0.101 [0.068, 0.139] n=235 |
| price_present=True | 0.381 [0.254, 0.492] n=63 | 0.208 [0.131, 0.294] n=63 |

## bucket_rank

### Overall — `bucket_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| all | 0.513 [0.456, 0.570] n=298 | 0.286 [0.246, 0.333] n=298 |

### By style — `bucket_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| compatibility | 0.857 [0.571, 1.000] n=7 * | 0.735 [0.367, 1.000] n=7 * |
| exact | 0.833 [0.500, 1.000] n=6 * | 0.263 [0.112, 0.417] n=6 * |
| feature | 0.700 [0.550, 0.850] n=40 | 0.522 [0.372, 0.661] n=40 |
| lay | 0.312 [0.188, 0.438] n=48 | 0.142 [0.073, 0.223] n=48 |
| plain | 0.667 [0.537, 0.796] n=54 | 0.337 [0.238, 0.446] n=54 |
| product_type | 0.660 [0.520, 0.780] n=50 | 0.376 [0.270, 0.490] n=50 |
| symptom | 0.326 [0.186, 0.465] n=43 | 0.147 [0.067, 0.244] n=43 |
| use_case | 0.320 [0.180, 0.440] n=50 | 0.152 [0.073, 0.242] n=50 |

### By intent label (style prior) — `bucket_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| browsing | 0.507 [0.449, 0.562] n=292 | 0.287 [0.243, 0.333] n=292 |
| buying | 0.833 [0.500, 1.000] n=6 * | 0.263 [0.112, 0.417] n=6 * |

### By card hard-constraints voiced — `bucket_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| card_hard_said=0 | 0.385 [0.286, 0.484] n=91 | 0.159 [0.105, 0.223] n=91 |
| card_hard_said=1 | 0.484 [0.389, 0.579] n=95 | 0.263 [0.184, 0.342] n=95 |
| card_hard_said=2+ | 0.643 [0.554, 0.732] n=112 | 0.410 [0.334, 0.489] n=112 |

### By parsed hard slots — `bucket_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| n_hard=0 | 0.439 [0.369, 0.505] n=198 | 0.224 [0.176, 0.273] n=198 |
| n_hard=1 | 0.674 [0.570, 0.767] n=86 | 0.407 [0.320, 0.501] n=86 |
| n_hard=2+ | 0.571 [0.286, 0.786] n=14 * | 0.421 [0.189, 0.653] n=14 * |

### By resolver confidence — `bucket_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| conf <0.2 | 0.507 [0.440, 0.573] n=225 | 0.238 [0.194, 0.282] n=225 |
| conf >=0.2 | 0.534 [0.411, 0.644] n=73 | 0.435 [0.322, 0.547] n=73 |

### By generator — `bucket_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| gemma2:9b | 0.528 [0.464, 0.587] n=235 | 0.271 [0.225, 0.319] n=235 |
| llama3.1:8b | 0.460 [0.349, 0.587] n=63 | 0.343 [0.238, 0.459] n=63 |

### By modifier negation — `bucket_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| negation=False | 0.534 [0.470, 0.597] n=236 | 0.298 [0.251, 0.349] n=236 |
| negation=True | 0.435 [0.323, 0.548] n=62 | 0.240 [0.152, 0.332] n=62 |

### By modifier for_other — `bucket_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| for_other=False | 0.511 [0.450, 0.576] n=229 | 0.281 [0.234, 0.335] n=229 |
| for_other=True | 0.522 [0.406, 0.638] n=69 | 0.304 [0.211, 0.402] n=69 |

### By modifier vague_budget — `bucket_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| vague_budget=False | 0.502 [0.442, 0.566] n=249 | 0.286 [0.240, 0.339] n=249 |
| vague_budget=True | 0.571 [0.429, 0.714] n=49 | 0.286 [0.183, 0.393] n=49 |

### By modifier format_noise — `bucket_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| format_noise=False | 0.507 [0.452, 0.565] n=294 | 0.285 [0.241, 0.333] n=294 |
| format_noise=True | 1.000 [1.000, 1.000] n=4 * | 0.369 [0.232, 0.500] n=4 * |

### By listing overlap quartile — `bucket_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| overlap Q1 | 0.203 [0.108, 0.297] n=74 | 0.080 [0.034, 0.135] n=74 |
| overlap Q2 | 0.533 [0.413, 0.640] n=75 | 0.264 [0.187, 0.349] n=75 |
| overlap Q3 | 0.594 [0.478, 0.710] n=69 | 0.335 [0.242, 0.430] n=69 |
| overlap Q4 | 0.713 [0.613, 0.812] n=80 | 0.456 [0.363, 0.550] n=80 |

### By descriptiveness quartile — `bucket_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| descriptiveness Q1 | 0.411 [0.301, 0.521] n=73 | 0.242 [0.156, 0.333] n=73 |
| descriptiveness Q2 | 0.600 [0.480, 0.707] n=75 | 0.312 [0.229, 0.402] n=75 |
| descriptiveness Q3 | 0.453 [0.347, 0.560] n=75 | 0.243 [0.161, 0.327] n=75 |
| descriptiveness Q4 | 0.587 [0.480, 0.693] n=75 | 0.347 [0.263, 0.442] n=75 |

### By title_richness quartile — `bucket_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| title_richness Q1 | 0.514 [0.392, 0.622] n=74 | 0.315 [0.226, 0.410] n=74 |
| title_richness Q2 | 0.351 [0.243, 0.459] n=74 | 0.176 [0.108, 0.254] n=74 |
| title_richness Q3 | 0.560 [0.440, 0.667] n=75 | 0.271 [0.193, 0.357] n=75 |
| title_richness Q4 | 0.627 [0.520, 0.720] n=75 | 0.383 [0.286, 0.476] n=75 |

### By jargon quartile — `bucket_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| jargon Q1 | 0.649 [0.541, 0.757] n=74 | 0.388 [0.296, 0.483] n=74 |
| jargon Q2 | 0.554 [0.432, 0.662] n=74 | 0.280 [0.200, 0.364] n=74 |
| jargon Q3 | 0.438 [0.329, 0.562] n=73 | 0.224 [0.149, 0.310] n=73 |
| jargon Q4 | 0.416 [0.312, 0.519] n=77 | 0.254 [0.171, 0.349] n=77 |

### By bucket_size quartile — `bucket_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| bucket_size Q1 | 0.514 [0.392, 0.635] n=74 | 0.301 [0.214, 0.391] n=74 |
| bucket_size Q2 | 0.549 [0.437, 0.662] n=71 | 0.382 [0.280, 0.482] n=71 |
| bucket_size Q3 | 0.545 [0.429, 0.662] n=77 | 0.305 [0.216, 0.398] n=77 |
| bucket_size Q4 | 0.447 [0.329, 0.566] n=76 | 0.163 [0.108, 0.229] n=76 |

### By popularity quartile — `bucket_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| popularity Q1 | 0.632 [0.509, 0.754] n=57 | 0.376 [0.270, 0.488] n=57 |
| popularity Q2 | 0.440 [0.341, 0.538] n=91 | 0.233 [0.163, 0.306] n=91 |
| popularity Q3 | 0.440 [0.333, 0.547] n=75 | 0.226 [0.148, 0.311] n=75 |
| popularity Q4 | 0.587 [0.480, 0.707] n=75 | 0.344 [0.255, 0.436] n=75 |

### By silent_on_material — `bucket_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| silent_on_material=False | 0.462 [0.379, 0.545] n=132 | 0.280 [0.214, 0.349] n=132 |
| silent_on_material=True | 0.554 [0.476, 0.633] n=166 | 0.292 [0.234, 0.351] n=166 |

### By has_near_duplicate — `bucket_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| has_near_duplicate=False | 0.516 [0.457, 0.579] n=254 | 0.299 [0.251, 0.347] n=254 |
| has_near_duplicate=True | 0.500 [0.364, 0.636] n=44 | 0.215 [0.129, 0.315] n=44 |

### By has_model_code — `bucket_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| has_model_code=False | 0.480 [0.415, 0.541] n=246 | 0.288 [0.240, 0.338] n=246 |
| has_model_code=True | 0.673 [0.538, 0.808] n=52 | 0.280 [0.192, 0.372] n=52 |

### By compat_eligible — `bucket_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| compat_eligible=False | 0.470 [0.406, 0.526] n=251 | 0.233 [0.190, 0.278] n=251 |
| compat_eligible=True | 0.745 [0.617, 0.872] n=47 | 0.574 [0.445, 0.705] n=47 |

### By promo_bucket — `bucket_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| promo_bucket=False | 0.548 [0.491, 0.606] n=279 | 0.306 [0.260, 0.354] n=279 |
| promo_bucket=True | 0.000 [0.000, 0.000] n=19 * | 0.000 [0.000, 0.000] n=19 * |

### By price_present — `bucket_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| price_present=False | 0.489 [0.426, 0.553] n=235 | 0.272 [0.225, 0.325] n=235 |
| price_present=True | 0.603 [0.476, 0.714] n=63 | 0.340 [0.237, 0.442] n=63 |
