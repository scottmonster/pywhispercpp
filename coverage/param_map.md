# Param Default Map

## 1. Scope

- This file groups params by the earliest public place they can be set.
- Shared decode params are listed under `Model.__init__` because that is the
  first public entrypoint that accepts them, even though `transcribe()` can
  override them later.
- Source entries point to the owning implementation line for the effective
  runtime default, not every call-site hop used to reach it.
- Effective runtime default means the value used by a plain `Model()` call
  with no overrides, plus method-local defaults for later method calls.
- `params_sampling_strategy=0` in `model.py  line: 90`, so runtime starts in
  greedy mode via `model.py  line: 163` and `whisper.cpp  line: 5995`.

## 2. Earliest Set In `Model.__init__`

### 2.1 Shared Whisper Decode Params

| Param                      | Default                        | Source                 |
| -------------------------- | ------------------------------ | ---------------------- |
| n_threads                  | min(4, hardware_concurrency()) | whisper.cpp line: 5910 |
| n_max_text_ctx             | 16384                          | whisper.cpp line: 5911 |
| offset_ms                  | 0                              | whisper.cpp line: 5912 |
| duration_ms                | 0                              | whisper.cpp line: 5913 |
| translate                  | False                          | whisper.cpp line: 5915 |
| no_context                 | True                           | whisper.cpp line: 5916 |
| no_timestamps              | False                          | whisper.cpp line: 5917 |
| single_segment             | False                          | whisper.cpp line: 5918 |
| print_special              | False                          | whisper.cpp line: 5919 |
| print_progress             | True                           | whisper.cpp line: 5920 |
| print_realtime             | False                          | whisper.cpp line: 5921 |
| print_timestamps           | True                           | whisper.cpp line: 5922 |
| token_timestamps           | False                          | whisper.cpp line: 5924 |
| thold_pt                   | 0.01                           | whisper.cpp line: 5925 |
| thold_ptsum                | 0.01                           | whisper.cpp line: 5926 |
| max_len                    | 0                              | whisper.cpp line: 5927 |
| split_on_word              | False                          | whisper.cpp line: 5928 |
| max_tokens                 | 0                              | whisper.cpp line: 5929 |
| debug_mode                 | False                          | whisper.cpp line: 5931 |
| audio_ctx                  | 0                              | whisper.cpp line: 5932 |
| tdrz_enable                | False                          | whisper.cpp line: 5934 |
| initial_prompt             | None                           | whisper.cpp line: 5938 |
| prompt_tokens              | None                           | whisper.cpp line: 5940 |
| prompt_n_tokens            | 0                              | whisper.cpp line: 5941 |
| carry_initial_prompt       | False                          | whisper.cpp line: 5939 |
| language                   | "en"                           | whisper.cpp line: 5943 |
| detect_language            | False                          | whisper.cpp line: 5944 |
| suppress_blank             | True                           | whisper.cpp line: 5946 |
| suppress_non_speech_tokens | False                          | whisper.cpp line: 5947 |
| suppress_nst               | False                          | whisper.cpp line: 5947 |
| suppress_regex             | ''                             | whisper.cpp line: 5936 |
| temperature                | 0.0                            | whisper.cpp line: 5949 |
| max_initial_ts             | 1.0                            | whisper.cpp line: 5950 |
| length_penalty             | -1.0                           | whisper.cpp line: 5951 |
| temperature_inc            | 0.2                            | whisper.cpp line: 5953 |
| entropy_thold              | 2.4                            | whisper.cpp line: 5954 |
| logprob_thold              | -1.0                           | whisper.cpp line: 5955 |
| no_speech_thold            | 0.6                            | whisper.cpp line: 5956 |
| greedy                     | best_of=5                      | whisper.cpp line: 5995 |
| beam_search                | beam_size=-1, patience=-1.0    | whisper.cpp line: 5906 |
| vad                        | False                          | whisper.cpp line: 5988 |
| vad_model_path             | None                           | whisper.cpp line: 5989 |

### 2.2 `__init__`-Only Params

| Param                       | Default                          | Source                 |
| --------------------------- | -------------------------------- | ---------------------- |
| model                       | tiny                             | model.py line: 88      |
| models_dir                  | MODELS_DIR                       | constants.py line: 24  |
| params_sampling_strategy    | 0 (GREEDY)                       | model.py line: 90      |
| redirect_whispercpp_logs_to | False                            | model.py line: 91      |
| use_openvino                | False                            | model.py line: 92      |
| openvino_model_path         | None                             | model.py line: 93      |
| openvino_device             | CPU                              | model.py line: 94      |
| openvino_cache_dir          | None                             | model.py line: 95      |
| context_params              | whisper_context_default_params() | whisper.cpp line: 3606 |

### 2.3 Notes

- These shared decode params can also be passed again to `transcribe()`.
- Their earliest public set point is still `Model.__init__`.
- `models_dir` is bypassed when `model` is already a direct file path; that
  branch is handled in `utils.py  line: 96`.
- `context_params=None` in Python delegates to the runtime default struct from
  `whisper.cpp  line: 3606`.
- `suppress_non_speech_tokens` is a Python alias for `suppress_nst`; see
  `model.py  line: 359` and `whisper.cpp  line: 5947`.

## 3. Earliest Set In `Model.transcribe()`

| Param                | Default  | Source             |
| -------------------- | -------- | ------------------ |
| media                | required | model.py line: 180 |
| n_processors         | None     | model.py line: 181 |
| new_segment_callback | None     | model.py line: 182 |
| abort_callback       | None     | model.py line: 183 |
| extract_probability  | False    | model.py line: 209 |

### 3.1 Notes

- `transcribe()` also accepts the shared decode params from section 2.1 as
  override kwargs.
- `extract_probability` is Python-only and is not part of
  `whisper_full_params`; see `model.py  line: 209`.

## 4. Earliest Set In `Model.auto_detect_language()`

| Param     | Default                               | Source             |
| --------- | ------------------------------------- | ------------------ |
| media     | required                              | model.py line: 497 |
| offset_ms | None -> uses `self._params.offset_ms` | model.py line: 513 |
| n_threads | None -> uses `self._params.n_threads` | model.py line: 516 |

### 4.1 Notes

- The current code no longer hardcodes `offset_ms=0` and `n_threads=4`.
- If omitted, `auto_detect_language()` inherits those values from the model's
  active decode params; see `model.py  line: 513` and `model.py  line: 516`.

## 5. Earliest Set In `Model.__init__(context_params=...)`

| ContextParams field  | Default             | Source                 |
| -------------------- | ------------------- | ---------------------- |
| use_gpu              | True                | whisper.cpp line: 3608 |
| flash_attn           | True                | whisper.cpp line: 3609 |
| gpu_device           | 0                   | whisper.cpp line: 3610 |
| dtw_token_timestamps | False               | whisper.cpp line: 3612 |
| dtw_aheads_preset    | WHISPER_AHEADS_NONE | whisper.cpp line: 3613 |
| dtw_n_top            | -1                  | whisper.cpp line: 3614 |
| dtw_mem_size         | 1024*1024*128       | whisper.cpp line: 3619 |

### 5.1 Notes

- These are loader/context params, not `whisper_full_params` decode params.
- They are applied through `whisper_context_default_params()` and
  `whisper_init_from_file_with_params(...)`; see
  `whisper.cpp  line: 3606` and `model.py  line: 349`.
- CLI flags like `--no-gpu`, `--device`, `--flash-attn`,
  `--no-flash-attn`, and parts of `--dtw` map here indirectly.

## 6. Effective Runtime Notes

- `params_sampling_strategy` defaults to 0 in `model.py  line: 90`, so the
  active runtime decoding defaults are greedy via `model.py  line: 163` and
  `whisper.cpp  line: 5995`. The untouched beam block remains at the base
  initializer from `whisper.cpp  line: 5906`.
- `suppress_non_speech_tokens` is a Python alias. The actual backing field is
  `suppress_nst`; see `model.py  line: 359` and `whisper.cpp  line: 5947`.
- `extract_probability` is not part of `whisper_full_params`. Its runtime
  default is set in Python during `transcribe()`; see `model.py  line: 209`.
- `auto_detect_language()` now inherits `offset_ms` and `n_threads` from the
  model params when those arguments are omitted; see `model.py  line: 513`
  and `model.py  line: 516`.

## 7. Schema vs Runtime Differences

### 7.1 `beam_search`

- Schema default: `{"beam_size": -1, "patience": -1.0}` in
  `constants.py  line: 303` and `model.pyi  line: 96`
- Effective runtime default: `{"beam_size": -1, "patience": -1.0}` from
  `whisper.cpp  line: 5906`
- Why: the current generated schema matches runtime. `Model()` still starts in
  greedy mode because `params_sampling_strategy=0` at `model.py  line: 90`.

### 7.2 `extract_probability`

- Schema default: `False` in `constants.py  line: 309` and
  `model.pyi  line: 149`
- Effective runtime default: `False` in `model.py  line: 209`
- Why: the schema matches runtime, but the default is owned by Python, not by
  whisper.cpp.

### 7.3 `suppress_non_speech_tokens`

- Schema default: `False` in `constants.py  line: 237` and
  `model.pyi  line: 85`
- Effective runtime default: `False` from `whisper.cpp  line: 5947`
- Why: the schema matches runtime, but the param is only an alias to
  `suppress_nst`; the alias rewrite happens in `model.py  line: 359`.
