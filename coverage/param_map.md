# Param Default Map

## 1. Scope

- This file groups params by the earliest public place they can be set.
- Shared decode params are listed under `Model.__init__` because that is the
  first public entrypoint that accepts them, even though `transcribe()` can
  override them later.
- Effective runtime default means the value used by a plain `Model()` call
  with no overrides, plus method-local defaults for later method calls.
- `params_sampling_strategy=0` by default, so runtime starts in greedy mode.

## 2. Earliest Set In `Model.__init__`

### 2.1 Shared Whisper Decode Params

| Param                      | Default                        | Source                      |
| -------------------------- | ------------------------------ | --------------------------- |
| n_threads                  | min(4, hardware_concurrency()) | whisper.cpp                 |
| n_max_text_ctx             | 16384                          | whisper.cpp                 |
| offset_ms                  | 0                              | whisper.cpp                 |
| duration_ms                | 0                              | whisper.cpp                 |
| translate                  | False                          | whisper.cpp                 |
| no_context                 | True                           | whisper.cpp                 |
| no_timestamps              | False                          | whisper.cpp                 |
| single_segment             | False                          | whisper.cpp                 |
| print_special              | False                          | whisper.cpp                 |
| print_progress             | True                           | whisper.cpp                 |
| print_realtime             | False                          | whisper.cpp                 |
| print_timestamps           | True                           | whisper.cpp                 |
| token_timestamps           | False                          | whisper.cpp                 |
| thold_pt                   | 0.01                           | whisper.cpp                 |
| thold_ptsum                | 0.01                           | whisper.cpp                 |
| max_len                    | 0                              | whisper.cpp                 |
| split_on_word              | False                          | whisper.cpp                 |
| max_tokens                 | 0                              | whisper.cpp                 |
| debug_mode                 | False                          | whisper.cpp                 |
| audio_ctx                  | 0                              | whisper.cpp                 |
| tdrz_enable                | False                          | whisper.cpp                 |
| initial_prompt             | None                           | whisper.cpp                 |
| grammar                    | None                           | whisper.cpp                 |
| grammar_rule               | root when grammar is set       | bindings                    |
| prompt_tokens              | None                           | whisper.cpp                 |
| prompt_n_tokens            | 0                              | whisper.cpp                 |
| carry_initial_prompt       | False                          | whisper.cpp                 |
| language                   | "en"                           | whisper.cpp                 |
| detect_language            | False                          | whisper.cpp                 |
| suppress_blank             | True                           | whisper.cpp                 |
| suppress_non_speech_tokens | False                          | python alias -> whisper.cpp |
| suppress_nst               | False                          | whisper.cpp                 |
| suppress_regex             | ''                             | whisper.cpp                 |
| temperature                | 0.0                            | whisper.cpp                 |
| max_initial_ts             | 1.0                            | whisper.cpp                 |
| length_penalty             | -1.0                           | whisper.cpp                 |
| temperature_inc            | 0.2                            | whisper.cpp                 |
| entropy_thold              | 2.4                            | whisper.cpp                 |
| logprob_thold              | -1.0                           | whisper.cpp                 |
| no_speech_thold            | 0.6                            | whisper.cpp                 |
| grammar_penalty            | 100.0                          | whisper.cpp                 |
| greedy                     | best_of=5                      | whisper.cpp                 |
| beam_search                | beam_size=-1, patience=-1.0    | whisper.cpp                 |
| vad                        | False                          | whisper.cpp                 |
| vad_model_path             | None                           | whisper.cpp                 |

### 2.2 `Model.__init__`-Only Params

| Param                       | Default                                                      | Source               |
| --------------------------- | ------------------------------------------------------------ | -------------------- |
| model                       | tiny                                                         | python               |
| models_dir                  | MODELS_DIR unless model is a direct file path                | python/utils         |
| params_sampling_strategy    | 0 (GREEDY)                                                   | python               |
| redirect_whispercpp_logs_to | False                                                        | python               |
| use_openvino                | False                                                        | python               |
| openvino_model_path         | None                                                         | python               |
| openvino_device             | CPU                                                          | python               |
| openvino_cache_dir          | None                                                         | python               |
| context_params              | None arg; omitted path uses whisper_context_default_params() | python + whisper.cpp |

### 2.3 Notes

- These shared decode params can also be passed again to `transcribe()`.
- Their earliest public set point is still `Model.__init__`.
- `grammar_rule` is only used when `grammar` is provided.
- `suppress_non_speech_tokens` is a Python alias for `suppress_nst`.

## 3. Earliest Set In `Model.transcribe()`

| Param                | Default  | Source |
| -------------------- | -------- | ------ |
| media                | required | python |
| n_processors         | None     | python |
| new_segment_callback | None     | python |
| abort_callback       | None     | python |
| extract_probability  | False    | python |

### 3.1 Notes

- `transcribe()` also accepts the shared decode params from section 2.1 as
  override kwargs.
- `extract_probability` is Python-only and is not part of
  `whisper_full_params`.

## 4. Earliest Set In `Model.auto_detect_language()`

| Param     | Default                               | Source |
| --------- | ------------------------------------- | ------ |
| media     | required                              | python |
| offset_ms | None -> uses `self._params.offset_ms` | python |
| n_threads | None -> uses `self._params.n_threads` | python |

### 4.1 Notes

- The current code no longer hardcodes `offset_ms=0` and `n_threads=4`.
- If omitted, `auto_detect_language()` inherits those values from the model's
  active decode params.

## 5. Earliest Set In `Model.__init__(context_params=...)`

| ContextParams field  | Default             | Source      |
| -------------------- | ------------------- | ----------- |
| use_gpu              | True                | whisper.cpp |
| flash_attn           | True                | whisper.cpp |
| gpu_device           | 0                   | whisper.cpp |
| dtw_token_timestamps | False               | whisper.cpp |
| dtw_aheads_preset    | WHISPER_AHEADS_NONE | whisper.cpp |
| dtw_n_top            | -1                  | whisper.cpp |
| dtw_mem_size         | 1024*1024*128       | whisper.cpp |

### 5.1 Notes

- These are loader/context params, not `whisper_full_params` decode params.
- They are applied through `whisper_context_default_params()` and
  `whisper_init_from_file_with_params(...)`.
- CLI flags like `--no-gpu`, `--device`, `--flash-attn`,
  `--no-flash-attn`, and parts of `--dtw` map here indirectly.

## 6. Effective Runtime Notes

- `params_sampling_strategy` defaults to 0 in `Model.__init__`, so the active
  runtime decoding defaults are greedy: `greedy={"best_of": 5}` and
  `beam_search={"beam_size": -1, "patience": -1.0}`.
- `grammar_rule` is not used unless `grammar` is provided. When `grammar` is
  provided and `grammar_rule` is omitted, the binding layer sets it to `root`.
- `suppress_non_speech_tokens` is a Python alias. The actual backing field is
  `suppress_nst`, and its runtime default comes from whisper.cpp.
- `extract_probability` is not part of `whisper_full_params`. Its runtime
  default is set in Python during `transcribe()`.
- `auto_detect_language()` now inherits `offset_ms` and `n_threads` from the
  model params when those arguments are omitted.

## 7. Schema vs Runtime Differences

### 7.1 `beam_search`

- Schema default: `{"beam_size": 5, "patience": -1.0}`
- Effective runtime default: `{"beam_size": -1, "patience": -1.0}`
- Why: `Model()` starts in greedy mode, so the beam block is not rewritten by
  the strategy switch.

### 7.2 `grammar_rule`

- Schema default: `"root"`
- Effective runtime default: unused unless `grammar` is set; then `"root"`
- Why: the default is introduced by `set_grammar()` in the binding layer, not
  by the base params struct.

### 7.3 `extract_probability`

- Schema default: `False`
- Effective runtime default: `False`
- Why: the schema matches runtime, but the default is owned by Python, not by
  whisper.cpp.

### 7.4 `suppress_non_speech_tokens`

- Schema default: `False`
- Effective runtime default: `False`
- Why: the schema matches runtime, but the param is only an alias to
  `suppress_nst`.
