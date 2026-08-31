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

### By parsed hard slots — `template_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| n_hard=0 | 0.046 [0.034, 0.059] n=1169 | 0.021 [0.014, 0.028] n=1169 |
| n_hard=1 | 0.070 [0.047, 0.092] n=446 | 0.033 [0.020, 0.048] n=446 |
| n_hard=2+ | 0.100 [0.029, 0.171] n=70 | 0.061 [0.018, 0.119] n=70 |

### By resolver confidence — `template_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| conf <0.2 | 0.050 [0.038, 0.061] n=1271 | 0.020 [0.014, 0.026] n=1271 |
| conf >=0.2 | 0.070 [0.046, 0.097] n=414 | 0.045 [0.027, 0.064] n=414 |

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

### By parsed hard slots — `lexical_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| n_hard=0 | 0.122 [0.104, 0.142] n=1169 | 0.060 [0.048, 0.071] n=1169 |
| n_hard=1 | 0.186 [0.150, 0.222] n=446 | 0.091 [0.069, 0.115] n=446 |
| n_hard=2+ | 0.214 [0.129, 0.314] n=70 | 0.119 [0.059, 0.196] n=70 |

### By resolver confidence — `lexical_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| conf <0.2 | 0.133 [0.114, 0.152] n=1271 | 0.061 [0.050, 0.072] n=1271 |
| conf >=0.2 | 0.174 [0.138, 0.210] n=414 | 0.100 [0.074, 0.128] n=414 |

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

## parsed_rank

### Overall — `parsed_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| all | 0.233 [0.211, 0.253] n=1685 | 0.134 [0.119, 0.148] n=1685 |

### By style — `parsed_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| compatibility | 0.455 [0.273, 0.636] n=33 | 0.346 [0.194, 0.503] n=33 |
| exact | 0.711 [0.553, 0.842] n=38 | 0.503 [0.368, 0.637] n=38 |
| feature | 0.491 [0.435, 0.554] n=269 | 0.290 [0.246, 0.340] n=269 |
| lay | 0.052 [0.030, 0.078] n=269 | 0.024 [0.009, 0.041] n=269 |
| plain | 0.364 [0.312, 0.424] n=269 | 0.208 [0.169, 0.251] n=269 |
| product_type | 0.320 [0.264, 0.372] n=269 | 0.179 [0.140, 0.220] n=269 |
| symptom | 0.056 [0.030, 0.086] n=269 | 0.016 [0.007, 0.027] n=269 |
| use_case | 0.019 [0.004, 0.037] n=269 | 0.007 [0.002, 0.014] n=269 |

### By intent label (style prior) — `parsed_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| browsing | 0.222 [0.202, 0.243] n=1647 | 0.125 [0.111, 0.140] n=1647 |
| buying | 0.711 [0.553, 0.842] n=38 | 0.503 [0.368, 0.637] n=38 |

### By card hard-constraints voiced — `parsed_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| card_hard_said=0 | 0.074 [0.052, 0.096] n=553 | 0.031 [0.020, 0.044] n=553 |
| card_hard_said=1 | 0.212 [0.180, 0.248] n=532 | 0.116 [0.094, 0.140] n=532 |
| card_hard_said=2+ | 0.397 [0.357, 0.435] n=600 | 0.244 [0.212, 0.275] n=600 |

### By parsed hard slots — `parsed_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| n_hard=0 | 0.182 [0.161, 0.205] n=1169 | 0.104 [0.089, 0.121] n=1169 |
| n_hard=1 | 0.332 [0.289, 0.374] n=446 | 0.195 [0.163, 0.227] n=446 |
| n_hard=2+ | 0.443 [0.329, 0.557] n=70 | 0.231 [0.154, 0.322] n=70 |

### By resolver confidence — `parsed_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| conf <0.2 | 0.216 [0.194, 0.238] n=1271 | 0.121 [0.106, 0.137] n=1271 |
| conf >=0.2 | 0.285 [0.242, 0.331] n=414 | 0.172 [0.139, 0.205] n=414 |

### By generator — `parsed_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| gemma2:9b | 0.214 [0.185, 0.242] n=842 | 0.110 [0.093, 0.128] n=842 |
| llama3.1:8b | 0.251 [0.224, 0.280] n=843 | 0.157 [0.136, 0.180] n=843 |

### By modifier negation — `parsed_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| negation=False | 0.238 [0.216, 0.260] n=1329 | 0.137 [0.121, 0.153] n=1329 |
| negation=True | 0.213 [0.171, 0.256] n=356 | 0.120 [0.091, 0.151] n=356 |

### By modifier for_other — `parsed_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| for_other=False | 0.235 [0.211, 0.258] n=1316 | 0.135 [0.119, 0.151] n=1316 |
| for_other=True | 0.225 [0.184, 0.268] n=369 | 0.127 [0.099, 0.160] n=369 |

### By modifier vague_budget — `parsed_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| vague_budget=False | 0.233 [0.211, 0.255] n=1374 | 0.134 [0.119, 0.152] n=1374 |
| vague_budget=True | 0.232 [0.186, 0.280] n=311 | 0.130 [0.098, 0.165] n=311 |

### By modifier format_noise — `parsed_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| format_noise=False | 0.230 [0.211, 0.251] n=1675 | 0.132 [0.117, 0.146] n=1675 |
| format_noise=True | 0.600 [0.300, 0.900] n=10 * | 0.383 [0.133, 0.658] n=10 * |

### By listing overlap quartile — `parsed_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| overlap Q1 | 0.000 [0.000, 0.000] n=410 | 0.000 [0.000, 0.000] n=410 |
| overlap Q2 | 0.093 [0.064, 0.121] n=420 | 0.042 [0.026, 0.059] n=420 |
| overlap Q3 | 0.260 [0.219, 0.305] n=407 | 0.123 [0.098, 0.150] n=407 |
| overlap Q4 | 0.551 [0.504, 0.598] n=448 | 0.351 [0.315, 0.390] n=448 |

### By descriptiveness quartile — `parsed_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| descriptiveness Q1 | 0.207 [0.166, 0.245] n=416 | 0.103 [0.077, 0.127] n=416 |
| descriptiveness Q2 | 0.209 [0.169, 0.244] n=426 | 0.114 [0.087, 0.141] n=426 |
| descriptiveness Q3 | 0.212 [0.174, 0.251] n=419 | 0.142 [0.111, 0.174] n=419 |
| descriptiveness Q4 | 0.302 [0.259, 0.344] n=424 | 0.175 [0.145, 0.207] n=424 |

### By title_richness quartile — `parsed_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| title_richness Q1 | 0.147 [0.115, 0.183] n=416 | 0.076 [0.055, 0.100] n=416 |
| title_richness Q2 | 0.185 [0.147, 0.223] n=422 | 0.107 [0.081, 0.135] n=422 |
| title_richness Q3 | 0.279 [0.238, 0.319] n=420 | 0.151 [0.120, 0.182] n=420 |
| title_richness Q4 | 0.319 [0.274, 0.363] n=427 | 0.198 [0.164, 0.232] n=427 |

### By jargon quartile — `parsed_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| jargon Q1 | 0.213 [0.174, 0.254] n=413 | 0.112 [0.088, 0.139] n=413 |
| jargon Q2 | 0.260 [0.220, 0.301] n=419 | 0.158 [0.130, 0.191] n=419 |
| jargon Q3 | 0.250 [0.208, 0.292] n=428 | 0.136 [0.107, 0.165] n=428 |
| jargon Q4 | 0.207 [0.169, 0.245] n=425 | 0.127 [0.100, 0.157] n=425 |

### By bucket_size quartile — `parsed_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| bucket_size Q1 | 0.292 [0.247, 0.335] n=421 | 0.186 [0.153, 0.220] n=421 |
| bucket_size Q2 | 0.242 [0.199, 0.283] n=417 | 0.140 [0.110, 0.170] n=417 |
| bucket_size Q3 | 0.242 [0.199, 0.285] n=396 | 0.133 [0.106, 0.164] n=396 |
| bucket_size Q4 | 0.160 [0.126, 0.193] n=451 | 0.078 [0.058, 0.101] n=451 |

### By popularity quartile — `parsed_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| popularity Q1 | 0.215 [0.168, 0.263] n=274 | 0.114 [0.082, 0.147] n=274 |
| popularity Q2 | 0.235 [0.201, 0.272] n=548 | 0.131 [0.107, 0.156] n=548 |
| popularity Q3 | 0.248 [0.209, 0.287] n=435 | 0.155 [0.126, 0.185] n=435 |
| popularity Q4 | 0.224 [0.185, 0.264] n=428 | 0.128 [0.102, 0.156] n=428 |

### By silent_on_material — `parsed_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| silent_on_material=False | 0.187 [0.160, 0.215] n=738 | 0.106 [0.087, 0.126] n=738 |
| silent_on_material=True | 0.268 [0.241, 0.297] n=947 | 0.155 [0.134, 0.175] n=947 |

### By has_near_duplicate — `parsed_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| has_near_duplicate=False | 0.258 [0.237, 0.281] n=1429 | 0.148 [0.132, 0.164] n=1429 |
| has_near_duplicate=True | 0.090 [0.055, 0.125] n=256 | 0.051 [0.029, 0.076] n=256 |

### By has_model_code — `parsed_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| has_model_code=False | 0.219 [0.198, 0.242] n=1417 | 0.124 [0.109, 0.141] n=1417 |
| has_model_code=True | 0.302 [0.250, 0.358] n=268 | 0.182 [0.142, 0.225] n=268 |

### By compat_eligible — `parsed_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| compat_eligible=False | 0.209 [0.188, 0.230] n=1452 | 0.116 [0.102, 0.131] n=1452 |
| compat_eligible=True | 0.378 [0.318, 0.438] n=233 | 0.242 [0.195, 0.290] n=233 |

### By promo_bucket — `parsed_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| promo_bucket=False | 0.237 [0.216, 0.258] n=1545 | 0.137 [0.123, 0.152] n=1545 |
| promo_bucket=True | 0.186 [0.129, 0.250] n=140 | 0.092 [0.055, 0.134] n=140 |

### By price_present — `parsed_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| price_present=False | 0.208 [0.188, 0.231] n=1315 | 0.118 [0.103, 0.134] n=1315 |
| price_present=True | 0.319 [0.273, 0.365] n=370 | 0.189 [0.156, 0.224] n=370 |

## bucket_rank

### Overall — `bucket_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| all | 0.496 [0.471, 0.519] n=1685 | 0.254 [0.237, 0.272] n=1685 |

### By style — `bucket_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| compatibility | 0.818 [0.667, 0.939] n=33 | 0.506 [0.366, 0.651] n=33 |
| exact | 0.737 [0.605, 0.868] n=38 | 0.248 [0.171, 0.344] n=38 |
| feature | 0.662 [0.602, 0.717] n=269 | 0.366 [0.318, 0.414] n=269 |
| lay | 0.320 [0.260, 0.375] n=269 | 0.150 [0.115, 0.185] n=269 |
| plain | 0.665 [0.610, 0.721] n=269 | 0.377 [0.329, 0.425] n=269 |
| product_type | 0.658 [0.606, 0.714] n=269 | 0.351 [0.306, 0.397] n=269 |
| symptom | 0.327 [0.271, 0.383] n=269 | 0.137 [0.104, 0.170] n=269 |
| use_case | 0.268 [0.216, 0.320] n=269 | 0.116 [0.085, 0.147] n=269 |

### By intent label (style prior) — `bucket_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| browsing | 0.490 [0.466, 0.515] n=1647 | 0.255 [0.237, 0.272] n=1647 |
| buying | 0.737 [0.605, 0.868] n=38 | 0.248 [0.171, 0.344] n=38 |

### By card hard-constraints voiced — `bucket_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| card_hard_said=0 | 0.367 [0.327, 0.409] n=553 | 0.168 [0.143, 0.193] n=553 |
| card_hard_said=1 | 0.494 [0.453, 0.538] n=532 | 0.255 [0.225, 0.287] n=532 |
| card_hard_said=2+ | 0.615 [0.577, 0.655] n=600 | 0.334 [0.304, 0.365] n=600 |

### By parsed hard slots — `bucket_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| n_hard=0 | 0.447 [0.418, 0.476] n=1169 | 0.219 [0.200, 0.239] n=1169 |
| n_hard=1 | 0.587 [0.540, 0.632] n=446 | 0.316 [0.280, 0.354] n=446 |
| n_hard=2+ | 0.714 [0.600, 0.814] n=70 | 0.450 [0.355, 0.550] n=70 |

### By resolver confidence — `bucket_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| conf <0.2 | 0.487 [0.460, 0.515] n=1271 | 0.211 [0.193, 0.230] n=1271 |
| conf >=0.2 | 0.522 [0.473, 0.570] n=414 | 0.388 [0.344, 0.433] n=414 |

### By generator — `bucket_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| gemma2:9b | 0.529 [0.496, 0.561] n=842 | 0.265 [0.243, 0.290] n=842 |
| llama3.1:8b | 0.463 [0.431, 0.496] n=843 | 0.244 [0.220, 0.269] n=843 |

### By modifier negation — `bucket_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| negation=False | 0.512 [0.484, 0.537] n=1329 | 0.261 [0.241, 0.279] n=1329 |
| negation=True | 0.435 [0.385, 0.486] n=356 | 0.231 [0.195, 0.268] n=356 |

### By modifier for_other — `bucket_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| for_other=False | 0.498 [0.472, 0.524] n=1316 | 0.255 [0.237, 0.274] n=1316 |
| for_other=True | 0.488 [0.436, 0.539] n=369 | 0.251 [0.213, 0.288] n=369 |

### By modifier vague_budget — `bucket_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| vague_budget=False | 0.491 [0.464, 0.517] n=1374 | 0.252 [0.234, 0.272] n=1374 |
| vague_budget=True | 0.518 [0.460, 0.576] n=311 | 0.263 [0.222, 0.303] n=311 |

### By modifier format_noise — `bucket_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| format_noise=False | 0.493 [0.467, 0.518] n=1675 | 0.254 [0.236, 0.272] n=1675 |
| format_noise=True | 1.000 [1.000, 1.000] n=10 * | 0.270 [0.187, 0.367] n=10 * |

### By listing overlap quartile — `bucket_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| overlap Q1 | 0.193 [0.154, 0.232] n=410 | 0.083 [0.062, 0.107] n=410 |
| overlap Q2 | 0.498 [0.450, 0.543] n=420 | 0.236 [0.204, 0.269] n=420 |
| overlap Q3 | 0.587 [0.536, 0.634] n=407 | 0.319 [0.280, 0.357] n=407 |
| overlap Q4 | 0.688 [0.645, 0.730] n=448 | 0.369 [0.333, 0.405] n=448 |

### By descriptiveness quartile — `bucket_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| descriptiveness Q1 | 0.416 [0.370, 0.466] n=416 | 0.224 [0.190, 0.258] n=416 |
| descriptiveness Q2 | 0.486 [0.439, 0.531] n=426 | 0.236 [0.202, 0.272] n=426 |
| descriptiveness Q3 | 0.516 [0.463, 0.563] n=419 | 0.257 [0.222, 0.291] n=419 |
| descriptiveness Q4 | 0.564 [0.514, 0.608] n=424 | 0.301 [0.266, 0.337] n=424 |

### By title_richness quartile — `bucket_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| title_richness Q1 | 0.498 [0.447, 0.546] n=416 | 0.266 [0.230, 0.301] n=416 |
| title_richness Q2 | 0.393 [0.348, 0.438] n=422 | 0.199 [0.169, 0.232] n=422 |
| title_richness Q3 | 0.486 [0.438, 0.531] n=420 | 0.222 [0.191, 0.253] n=420 |
| title_richness Q4 | 0.604 [0.557, 0.649] n=427 | 0.330 [0.292, 0.368] n=427 |

### By jargon quartile — `bucket_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| jargon Q1 | 0.547 [0.501, 0.593] n=413 | 0.313 [0.275, 0.352] n=413 |
| jargon Q2 | 0.499 [0.449, 0.547] n=419 | 0.232 [0.202, 0.264] n=419 |
| jargon Q3 | 0.474 [0.425, 0.523] n=428 | 0.224 [0.192, 0.256] n=428 |
| jargon Q4 | 0.464 [0.416, 0.511] n=425 | 0.250 [0.213, 0.286] n=425 |

### By bucket_size quartile — `bucket_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| bucket_size Q1 | 0.473 [0.428, 0.518] n=421 | 0.250 [0.216, 0.287] n=421 |
| bucket_size Q2 | 0.576 [0.528, 0.624] n=417 | 0.323 [0.286, 0.361] n=417 |
| bucket_size Q3 | 0.528 [0.477, 0.578] n=396 | 0.279 [0.244, 0.319] n=396 |
| bucket_size Q4 | 0.415 [0.370, 0.461] n=451 | 0.174 [0.147, 0.202] n=451 |

### By popularity quartile — `bucket_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| popularity Q1 | 0.551 [0.496, 0.609] n=274 | 0.272 [0.230, 0.316] n=274 |
| popularity Q2 | 0.432 [0.392, 0.474] n=548 | 0.211 [0.183, 0.240] n=548 |
| popularity Q3 | 0.506 [0.460, 0.552] n=435 | 0.274 [0.241, 0.308] n=435 |
| popularity Q4 | 0.530 [0.481, 0.579] n=428 | 0.279 [0.245, 0.314] n=428 |

### By silent_on_material — `bucket_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| silent_on_material=False | 0.457 [0.423, 0.493] n=738 | 0.233 [0.209, 0.259] n=738 |
| silent_on_material=True | 0.526 [0.494, 0.555] n=947 | 0.271 [0.247, 0.294] n=947 |

### By has_near_duplicate — `bucket_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| has_near_duplicate=False | 0.500 [0.474, 0.527] n=1429 | 0.261 [0.242, 0.280] n=1429 |
| has_near_duplicate=True | 0.473 [0.410, 0.531] n=256 | 0.217 [0.176, 0.257] n=256 |

### By has_model_code — `bucket_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| has_model_code=False | 0.483 [0.457, 0.510] n=1417 | 0.261 [0.243, 0.281] n=1417 |
| has_model_code=True | 0.563 [0.504, 0.619] n=268 | 0.218 [0.184, 0.255] n=268 |

### By compat_eligible — `bucket_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| compat_eligible=False | 0.456 [0.430, 0.482] n=1452 | 0.218 [0.201, 0.237] n=1452 |
| compat_eligible=True | 0.742 [0.682, 0.798] n=233 | 0.478 [0.426, 0.535] n=233 |

### By promo_bucket — `bucket_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| promo_bucket=False | 0.535 [0.509, 0.559] n=1545 | 0.276 [0.258, 0.295] n=1545 |
| promo_bucket=True | 0.064 [0.029, 0.107] n=140 | 0.014 [0.005, 0.027] n=140 |

### By price_present — `bucket_rank`

| slice | HitRate@10 | MRR |
|---|---|---|
| price_present=False | 0.471 [0.445, 0.499] n=1315 | 0.244 [0.225, 0.264] n=1315 |
| price_present=True | 0.581 [0.532, 0.632] n=370 | 0.291 [0.256, 0.328] n=370 |
