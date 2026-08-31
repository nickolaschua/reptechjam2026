# Messy benchmark report

1685 scored cases. `*` = n < 20.

## First-question quality (`question_hit`)

| slice | hit rate |
|---|---|
| all=all | 0.553 [0.529, 0.576] n=1685 |
| style=compatibility | 0.667 [0.515, 0.818] n=33 |
| style=exact | 0.711 [0.553, 0.842] n=38 |
| style=feature | 0.546 [0.487, 0.606] n=269 |
| style=lay | 0.546 [0.491, 0.602] n=269 |
| style=plain | 0.546 [0.487, 0.606] n=269 |
| style=product_type | 0.546 [0.483, 0.606] n=269 |
| style=symptom | 0.546 [0.483, 0.606] n=269 |
| style=use_case | 0.546 [0.483, 0.610] n=269 |
| intent=browsing | 0.549 [0.525, 0.572] n=1647 |
| intent=buying | 0.711 [0.553, 0.842] n=38 |

## template_rank

### Overall — `template_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| all | 0.055 [0.044, 0.065] n=1685 | 0.026 [0.020, 0.033] n=1685 |

### By style — `template_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| compatibility | 0.121 [0.030, 0.242] n=33 | 0.056 [0.005, 0.126] n=33 |
| exact | 0.105 [0.026, 0.211] n=38 | 0.075 [0.009, 0.158] n=38 |
| feature | 0.104 [0.071, 0.141] n=269 | 0.061 [0.037, 0.087] n=269 |
| lay | 0.004 [0.000, 0.011] n=269 | 0.004 [0.000, 0.011] n=269 |
| plain | 0.108 [0.074, 0.149] n=269 | 0.054 [0.033, 0.079] n=269 |
| product_type | 0.086 [0.056, 0.119] n=269 | 0.023 [0.013, 0.035] n=269 |
| symptom | 0.000 [0.000, 0.000] n=269 | 0.000 [0.000, 0.000] n=269 |
| use_case | 0.011 [0.000, 0.026] n=269 | 0.003 [0.000, 0.008] n=269 |

### By intent label (style prior) — `template_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| browsing | 0.053 [0.043, 0.064] n=1647 | 0.025 [0.019, 0.031] n=1647 |
| buying | 0.105 [0.026, 0.211] n=38 | 0.075 [0.009, 0.158] n=38 |

### By card hard-constraints voiced — `template_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| card_hard_said=0 | 0.018 [0.007, 0.029] n=553 | 0.007 [0.002, 0.013] n=553 |
| card_hard_said=1 | 0.060 [0.039, 0.081] n=532 | 0.021 [0.013, 0.032] n=532 |
| card_hard_said=2+ | 0.083 [0.062, 0.107] n=600 | 0.048 [0.033, 0.064] n=600 |

### By generator — `template_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| gemma2:9b | 0.102 [0.082, 0.124] n=842 | 0.048 [0.036, 0.061] n=842 |
| llama3.1:8b | 0.007 [0.002, 0.013] n=843 | 0.004 [0.001, 0.008] n=843 |

### By modifier negation — `template_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| negation=False | 0.058 [0.046, 0.070] n=1329 | 0.029 [0.021, 0.037] n=1329 |
| negation=True | 0.042 [0.022, 0.065] n=356 | 0.016 [0.006, 0.027] n=356 |

### By modifier for_other — `template_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| for_other=False | 0.060 [0.048, 0.073] n=1316 | 0.029 [0.022, 0.038] n=1316 |
| for_other=True | 0.035 [0.016, 0.054] n=369 | 0.013 [0.005, 0.024] n=369 |

### By modifier vague_budget — `template_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| vague_budget=False | 0.057 [0.046, 0.070] n=1374 | 0.027 [0.021, 0.035] n=1374 |
| vague_budget=True | 0.042 [0.023, 0.064] n=311 | 0.020 [0.008, 0.035] n=311 |

### By modifier format_noise — `template_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| format_noise=False | 0.055 [0.044, 0.066] n=1675 | 0.026 [0.020, 0.033] n=1675 |
| format_noise=True | 0.000 [0.000, 0.000] n=10 * | 0.000 [0.000, 0.000] n=10 * |

### By listing overlap quartile — `template_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| overlap Q1 | 0.000 [0.000, 0.000] n=410 | 0.000 [0.000, 0.000] n=410 |
| overlap Q2 | 0.005 [0.000, 0.012] n=420 | 0.001 [0.000, 0.001] n=420 |
| overlap Q3 | 0.034 [0.020, 0.054] n=407 | 0.010 [0.004, 0.017] n=407 |
| overlap Q4 | 0.170 [0.138, 0.205] n=448 | 0.088 [0.067, 0.111] n=448 |

### By descriptiveness quartile — `template_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| descriptiveness Q1 | 0.034 [0.017, 0.053] n=416 | 0.008 [0.004, 0.013] n=416 |
| descriptiveness Q2 | 0.045 [0.026, 0.063] n=426 | 0.023 [0.012, 0.035] n=426 |
| descriptiveness Q3 | 0.041 [0.021, 0.060] n=419 | 0.023 [0.011, 0.036] n=419 |
| descriptiveness Q4 | 0.099 [0.071, 0.130] n=424 | 0.050 [0.032, 0.069] n=424 |

### By title_richness quartile — `template_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| title_richness Q1 | 0.019 [0.007, 0.034] n=416 | 0.005 [0.002, 0.010] n=416 |
| title_richness Q2 | 0.043 [0.024, 0.064] n=422 | 0.015 [0.007, 0.026] n=422 |
| title_richness Q3 | 0.062 [0.040, 0.086] n=420 | 0.029 [0.017, 0.044] n=420 |
| title_richness Q4 | 0.094 [0.068, 0.122] n=427 | 0.053 [0.035, 0.074] n=427 |

### By jargon quartile — `template_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| jargon Q1 | 0.061 [0.039, 0.085] n=413 | 0.028 [0.016, 0.042] n=413 |
| jargon Q2 | 0.064 [0.043, 0.088] n=419 | 0.035 [0.020, 0.052] n=419 |
| jargon Q3 | 0.056 [0.035, 0.079] n=428 | 0.022 [0.012, 0.034] n=428 |
| jargon Q4 | 0.038 [0.021, 0.056] n=425 | 0.019 [0.009, 0.031] n=425 |

### By bucket_size quartile — `template_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| bucket_size Q1 | 0.062 [0.040, 0.086] n=421 | 0.030 [0.016, 0.044] n=421 |
| bucket_size Q2 | 0.062 [0.041, 0.086] n=417 | 0.041 [0.025, 0.059] n=417 |
| bucket_size Q3 | 0.063 [0.040, 0.088] n=396 | 0.025 [0.013, 0.037] n=396 |
| bucket_size Q4 | 0.033 [0.018, 0.051] n=451 | 0.010 [0.004, 0.016] n=451 |

### By popularity quartile — `template_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| popularity Q1 | 0.047 [0.026, 0.073] n=274 | 0.017 [0.007, 0.030] n=274 |
| popularity Q2 | 0.046 [0.029, 0.064] n=548 | 0.019 [0.011, 0.029] n=548 |
| popularity Q3 | 0.060 [0.039, 0.083] n=435 | 0.036 [0.021, 0.052] n=435 |
| popularity Q4 | 0.065 [0.044, 0.091] n=428 | 0.030 [0.017, 0.044] n=428 |

### By silent_on_material — `template_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| silent_on_material=False | 0.033 [0.020, 0.046] n=738 | 0.018 [0.010, 0.027] n=738 |
| silent_on_material=True | 0.072 [0.056, 0.089] n=947 | 0.032 [0.023, 0.042] n=947 |

### By has_near_duplicate — `template_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| has_near_duplicate=False | 0.062 [0.050, 0.074] n=1429 | 0.030 [0.022, 0.037] n=1429 |
| has_near_duplicate=True | 0.016 [0.004, 0.031] n=256 | 0.004 [0.001, 0.008] n=256 |

### By has_model_code — `template_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| has_model_code=False | 0.054 [0.043, 0.066] n=1417 | 0.024 [0.018, 0.031] n=1417 |
| has_model_code=True | 0.056 [0.030, 0.086] n=268 | 0.036 [0.017, 0.059] n=268 |

### By compat_eligible — `template_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| compat_eligible=False | 0.044 [0.034, 0.055] n=1452 | 0.018 [0.013, 0.024] n=1452 |
| compat_eligible=True | 0.120 [0.082, 0.163] n=233 | 0.076 [0.049, 0.110] n=233 |

### By promo_bucket — `template_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| promo_bucket=False | 0.056 [0.045, 0.068] n=1545 | 0.027 [0.021, 0.035] n=1545 |
| promo_bucket=True | 0.036 [0.007, 0.071] n=140 | 0.010 [0.002, 0.021] n=140 |

### By price_present — `template_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| price_present=False | 0.047 [0.037, 0.059] n=1315 | 0.023 [0.016, 0.030] n=1315 |
| price_present=True | 0.081 [0.054, 0.108] n=370 | 0.038 [0.022, 0.055] n=370 |

## lexical_rank

### Overall — `lexical_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| all | 0.143 [0.127, 0.160] n=1685 | 0.070 [0.060, 0.081] n=1685 |

### By style — `lexical_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| compatibility | 0.364 [0.212, 0.515] n=33 | 0.209 [0.101, 0.337] n=33 |
| exact | 0.395 [0.237, 0.553] n=38 | 0.223 [0.113, 0.344] n=38 |
| feature | 0.294 [0.242, 0.353] n=269 | 0.141 [0.108, 0.177] n=269 |
| lay | 0.007 [0.000, 0.019] n=269 | 0.006 [0.000, 0.015] n=269 |
| plain | 0.242 [0.190, 0.297] n=269 | 0.132 [0.097, 0.169] n=269 |
| product_type | 0.230 [0.182, 0.283] n=269 | 0.097 [0.069, 0.128] n=269 |
| symptom | 0.011 [0.000, 0.026] n=269 | 0.006 [0.000, 0.015] n=269 |
| use_case | 0.011 [0.000, 0.026] n=269 | 0.003 [0.000, 0.008] n=269 |

### By intent label (style prior) — `lexical_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| browsing | 0.137 [0.121, 0.154] n=1647 | 0.067 [0.057, 0.078] n=1647 |
| buying | 0.395 [0.237, 0.553] n=38 | 0.223 [0.113, 0.344] n=38 |

### By card hard-constraints voiced — `lexical_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| card_hard_said=0 | 0.031 [0.018, 0.047] n=553 | 0.011 [0.005, 0.018] n=553 |
| card_hard_said=1 | 0.139 [0.109, 0.167] n=532 | 0.061 [0.045, 0.078] n=532 |
| card_hard_said=2+ | 0.250 [0.217, 0.285] n=600 | 0.134 [0.111, 0.157] n=600 |

### By generator — `lexical_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| gemma2:9b | 0.132 [0.110, 0.156] n=842 | 0.061 [0.047, 0.074] n=842 |
| llama3.1:8b | 0.154 [0.132, 0.179] n=843 | 0.080 [0.065, 0.096] n=843 |

### By modifier negation — `lexical_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| negation=False | 0.155 [0.135, 0.174] n=1329 | 0.078 [0.066, 0.091] n=1329 |
| negation=True | 0.098 [0.070, 0.129] n=356 | 0.040 [0.025, 0.058] n=356 |

### By modifier for_other — `lexical_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| for_other=False | 0.148 [0.129, 0.167] n=1316 | 0.073 [0.062, 0.085] n=1316 |
| for_other=True | 0.125 [0.092, 0.160] n=369 | 0.060 [0.040, 0.081] n=369 |

### By modifier vague_budget — `lexical_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| vague_budget=False | 0.154 [0.135, 0.174] n=1374 | 0.075 [0.064, 0.087] n=1374 |
| vague_budget=True | 0.093 [0.064, 0.129] n=311 | 0.049 [0.029, 0.071] n=311 |

### By modifier format_noise — `lexical_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| format_noise=False | 0.142 [0.125, 0.159] n=1675 | 0.069 [0.059, 0.080] n=1675 |
| format_noise=True | 0.300 [0.000, 0.600] n=10 * | 0.225 [0.000, 0.500] n=10 * |

### By listing overlap quartile — `lexical_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| overlap Q1 | 0.000 [0.000, 0.000] n=410 | 0.000 [0.000, 0.000] n=410 |
| overlap Q2 | 0.010 [0.002, 0.019] n=420 | 0.002 [0.000, 0.005] n=420 |
| overlap Q3 | 0.103 [0.074, 0.133] n=407 | 0.034 [0.022, 0.048] n=407 |
| overlap Q4 | 0.435 [0.391, 0.480] n=448 | 0.232 [0.198, 0.264] n=448 |

### By descriptiveness quartile — `lexical_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| descriptiveness Q1 | 0.094 [0.065, 0.125] n=416 | 0.030 [0.018, 0.044] n=416 |
| descriptiveness Q2 | 0.101 [0.073, 0.129] n=426 | 0.048 [0.032, 0.065] n=426 |
| descriptiveness Q3 | 0.141 [0.110, 0.177] n=419 | 0.072 [0.053, 0.094] n=419 |
| descriptiveness Q4 | 0.236 [0.193, 0.276] n=424 | 0.131 [0.102, 0.160] n=424 |

### By title_richness quartile — `lexical_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| title_richness Q1 | 0.082 [0.055, 0.111] n=416 | 0.033 [0.020, 0.047] n=416 |
| title_richness Q2 | 0.102 [0.073, 0.133] n=422 | 0.043 [0.028, 0.059] n=422 |
| title_richness Q3 | 0.150 [0.117, 0.186] n=420 | 0.069 [0.050, 0.088] n=420 |
| title_richness Q4 | 0.237 [0.197, 0.279] n=427 | 0.136 [0.108, 0.166] n=427 |

### By jargon quartile — `lexical_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| jargon Q1 | 0.160 [0.126, 0.196] n=413 | 0.075 [0.055, 0.099] n=413 |
| jargon Q2 | 0.172 [0.136, 0.208] n=419 | 0.094 [0.070, 0.119] n=419 |
| jargon Q3 | 0.140 [0.107, 0.175] n=428 | 0.062 [0.044, 0.082] n=428 |
| jargon Q4 | 0.101 [0.073, 0.132] n=425 | 0.051 [0.033, 0.070] n=425 |

### By bucket_size quartile — `lexical_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| bucket_size Q1 | 0.181 [0.143, 0.219] n=421 | 0.098 [0.074, 0.124] n=421 |
| bucket_size Q2 | 0.161 [0.127, 0.197] n=417 | 0.097 [0.072, 0.124] n=417 |
| bucket_size Q3 | 0.144 [0.109, 0.182] n=396 | 0.057 [0.040, 0.077] n=396 |
| bucket_size Q4 | 0.091 [0.067, 0.120] n=451 | 0.031 [0.019, 0.045] n=451 |

### By popularity quartile — `lexical_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| popularity Q1 | 0.142 [0.102, 0.186] n=274 | 0.057 [0.037, 0.080] n=274 |
| popularity Q2 | 0.137 [0.109, 0.166] n=548 | 0.058 [0.044, 0.075] n=548 |
| popularity Q3 | 0.147 [0.115, 0.179] n=435 | 0.084 [0.061, 0.108] n=435 |
| popularity Q4 | 0.147 [0.117, 0.182] n=428 | 0.081 [0.060, 0.104] n=428 |

### By silent_on_material — `lexical_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| silent_on_material=False | 0.106 [0.083, 0.129] n=738 | 0.056 [0.042, 0.071] n=738 |
| silent_on_material=True | 0.172 [0.148, 0.199] n=947 | 0.082 [0.067, 0.097] n=947 |

### By has_near_duplicate — `lexical_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| has_near_duplicate=False | 0.160 [0.141, 0.178] n=1429 | 0.080 [0.068, 0.092] n=1429 |
| has_near_duplicate=True | 0.047 [0.023, 0.074] n=256 | 0.016 [0.006, 0.029] n=256 |

### By has_model_code — `lexical_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| has_model_code=False | 0.138 [0.121, 0.157] n=1417 | 0.068 [0.056, 0.079] n=1417 |
| has_model_code=True | 0.168 [0.127, 0.213] n=268 | 0.085 [0.057, 0.116] n=268 |

### By compat_eligible — `lexical_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| compat_eligible=False | 0.123 [0.105, 0.139] n=1452 | 0.054 [0.045, 0.064] n=1452 |
| compat_eligible=True | 0.270 [0.215, 0.326] n=233 | 0.172 [0.131, 0.218] n=233 |

### By promo_bucket — `lexical_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| promo_bucket=False | 0.146 [0.129, 0.164] n=1545 | 0.073 [0.062, 0.085] n=1545 |
| promo_bucket=True | 0.107 [0.057, 0.164] n=140 | 0.042 [0.018, 0.071] n=140 |

### By price_present — `lexical_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| price_present=False | 0.124 [0.107, 0.142] n=1315 | 0.059 [0.049, 0.070] n=1315 |
| price_present=True | 0.211 [0.168, 0.254] n=370 | 0.110 [0.082, 0.138] n=370 |
