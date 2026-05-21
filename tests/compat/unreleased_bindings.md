# pywhispercpp Unreleased Bindings Reference

## 1. Notes

- This reflects the current unreleased bindings in `src/main.cpp`.
- Compared with v1.4.1, `whisper_full_params` now exposes several previously missing fields, including `no_timestamps`, `debug_mode`, `tdrz_enable`, `carry_initial_prompt`, `detect_language`, and `suppress_nst`.
- Abort handling is now available through both `params.set_abort_callback(...)` / `params.clear_abort_callback()` and the module-level helpers `assign_abort_callback(...)` / `clear_abort_callback(...)`.
- `new_segment_callback`, `encoder_begin_callback`, and `logits_filter_callback` remain helper-only callback bindings, but now follow the same wrapper-managed Python callback and `*_user_data` storage model as the newer callback helpers.
- `prompt_tokens` is now wrapper-managed and available through both the compatibility attribute `params.prompt_tokens` and dedicated helpers (`params.set_prompt_tokens(...)` / `params.clear_prompt_tokens()`), while the underlying C pointer remains wrapper-managed.
- `progress_callback` is available through both the compatibility attribute `params.progress_callback` and dedicated helpers (`params.set_progress_callback(...)` / `params.clear_progress_callback()` and `assign_progress_callback(...)` / `clear_progress_callback(...)`), while the underlying C callback remains wrapper-managed.
- Grammar support is not exposed by the Python bindings because it would require the whisper examples grammar-parser surface.
- Callback helpers only pass a trailing `user_data` argument when a Python `*_callback_user_data` value was explicitly set.

## 2. `whisper_full_params` Coverage

### 2.1 Exposed Fields

- `strategy` via params.strategy
- `n_threads` via params.n_threads
- `n_max_text_ctx` via params.n_max_text_ctx
- `offset_ms` via params.offset_ms
- `duration_ms` via params.duration_ms
- `translate` via params.translate
- `no_context` via params.no_context
- `no_timestamps` via params.no_timestamps
- `single_segment` via params.single_segment
- `print_special` via params.print_special
- `print_progress` via params.print_progress
- `print_realtime` via params.print_realtime
- `print_timestamps` via params.print_timestamps
- `token_timestamps` via params.token_timestamps
- `thold_pt` via params.thold_pt
- `thold_ptsum` via params.thold_ptsum
- `max_len` via params.max_len
- `split_on_word` via params.split_on_word
- `max_tokens` via params.max_tokens
- `debug_mode` via params.debug_mode
- `audio_ctx` via params.audio_ctx
- `tdrz_enable` via params.tdrz_enable
- `suppress_regex` via params.suppress_regex
- `initial_prompt` via params.initial_prompt
- `prompt_tokens` via `params.prompt_tokens`, `params.set_prompt_tokens(tokens)`, and `params.clear_prompt_tokens()` **(Python-native sequence API backed by wrapper-managed storage)**
- `prompt_n_tokens` via params.prompt_n_tokens
- `carry_initial_prompt` via params.carry_initial_prompt
- `language` via params.language
- `detect_language` via params.detect_language
- `suppress_blank` via params.suppress_blank
- `suppress_nst` via params.suppress_nst
- `temperature` via params.temperature
- `max_initial_ts` via params.max_initial_ts
- `length_penalty` via params.length_penalty
- `temperature_inc` via params.temperature_inc
- `entropy_thold` via params.entropy_thold
- `logprob_thold` via params.logprob_thold
- `no_speech_thold` via params.no_speech_thold
- `greedy` via params.greedy (dict wrapper, for example `{"best_of": 2}`)
- `beam_search` via params.beam_search (dict wrapper, for example `{"beam_size": 2, "patience": 0.5}`)
- `vad` via params.vad
- `vad_model_path` via params.vad_model_path
- `vad_params` via params.vad_params
- `new_segment_callback` via `params.set_new_segment_callback(callback)`, `params.clear_new_segment_callback()`, `pw.assign_new_segment_callback(params, callback)`, and `pw.clear_new_segment_callback(params)` **(helper-only path; not a direct `whisper_full_params` field)**
- `new_segment_callback_user_data` via params.new_segment_callback_user_data **(Python-visible object storage; the internal C callback user_data remains wrapper-managed)**
- `encoder_begin_callback` via `params.set_encoder_begin_callback(callback)`, `params.clear_encoder_begin_callback()`, `pw.assign_encoder_begin_callback(params, callback)`, and `pw.clear_encoder_begin_callback(params)` **(helper-only path; not a direct `whisper_full_params` field)**
- `encoder_begin_callback_user_data` via params.encoder_begin_callback_user_data **(Python-visible object storage; the internal C callback user_data remains wrapper-managed)**
- `logits_filter_callback` via `params.set_logits_filter_callback(callback)`, `params.clear_logits_filter_callback()`, `pw.assign_logits_filter_callback(params, callback)`, and `pw.clear_logits_filter_callback(params)` **(helper-only path; not a direct `whisper_full_params` field)**
- `logits_filter_callback_user_data` via params.logits_filter_callback_user_data **(Python-visible object storage; the internal C callback user_data remains wrapper-managed)**
- `abort_callback` via `params.set_abort_callback(callback)`, `pw.assign_abort_callback(params, callback)`, and corresponding clear helpers
- `abort_callback_user_data` via params.abort_callback_user_data **(Python-visible object storage; the internal C callback user_data remains wrapper-managed)**
- `progress_callback` via `params.progress_callback`, `params.set_progress_callback(callback)`, `params.clear_progress_callback()`, `pw.assign_progress_callback(params, callback)`, and `pw.clear_progress_callback(params)`
- `progress_callback_user_data` via params.progress_callback_user_data **(Python-visible object storage; the internal C callback user_data remains wrapper-managed)**

### 2.2 Not exposed at all

- `new_segment_callback` as a direct writable field (helper-only path)
- `encoder_begin_callback` as a direct writable field (helper-only path)
- `logits_filter_callback` as a direct writable field (helper-only path)
- `grammar_rules`
- `n_grammar_rules`
- `i_start_rule`
- `grammar_penalty`
- `set_grammar(...)`
- `clear_grammar()`

## 3. Broader `_pywhispercpp` Low-Level API

### 3.1 Context and Model Lifecycle

- `whisper_alignment_heads_preset`
- `whisper_context`
- `whisper_context_params`
- `whisper_model_loader`
- `whisper_context_default_params`
- `whisper_init_from_file`
- `whisper_init_from_file_with_params`
- `whisper_init_from_buffer`
- `whisper_init_from_buffer_with_params`
- `whisper_init`
- `whisper_init_with_params`
- `whisper_free`
- `whisper_full_default_params`
- `whisper_full`
- `whisper_full_parallel`

### 3.2 Audio, Mel, and Decode Pipeline

- `whisper_pcm_to_mel`
- `whisper_set_mel`
- `whisper_encode`
- `whisper_decode`

### 3.3 Language Helpers

- `whisper_lang_max_id`
- `whisper_lang_id`
- `whisper_lang_str`
- `whisper_lang_auto_detect`

### 3.4 Model Metadata and Logits

- `whisper_n_len`
- `whisper_n_vocab`
- `whisper_n_text_ctx`
- `whisper_n_audio_ctx`
- `whisper_is_multilingual`
- `whisper_get_logits`
- `whisper_model_type_readable`
- `whisper_model_n_vocab`
- `whisper_model_n_audio_ctx`
- `whisper_model_n_audio_state`
- `whisper_model_n_audio_head`
- `whisper_model_n_audio_layer`
- `whisper_model_n_text_ctx`
- `whisper_model_n_text_state`
- `whisper_model_n_text_head`
- `whisper_model_n_text_layer`
- `whisper_model_n_mels`
- `whisper_model_ftype`

### 3.5 Token Helpers

- `whisper_tokenize`
- `whisper_token`
- `whisper_token_data`
- `whisper_token_to_str`
- `whisper_token_to_bytes`
- `whisper_token_eot`
- `whisper_token_sot`
- `whisper_token_prev`
- `whisper_token_solm`
- `whisper_token_not`
- `whisper_token_beg`
- `whisper_token_lang`
- `whisper_token_translate`
- `whisper_token_transcribe`

### 3.6 Timings, System Info, and Logging

- `whisper_print_timings`
- `whisper_reset_timings`
- `whisper_print_system_info`
- `whisper_log_set`

### 3.7 Full Result Accessors

- `whisper_full_n_segments`
- `whisper_full_lang_id`
- `whisper_full_get_segment_t0`
- `whisper_full_get_segment_t1`
- `whisper_full_get_segment_speaker_turn_next`
- `whisper_full_get_segment_text`
- `whisper_full_n_tokens`
- `whisper_full_get_token_text`
- `whisper_full_get_token_id`
- `whisper_full_get_token_data`
- `whisper_full_get_token_p`

### 3.8 OpenVINO Helpers

- `whisper_ctx_init_openvino_encoder`

### 3.9 Callback Helper Functions

- `assign_new_segment_callback`
- `clear_new_segment_callback`
- `assign_encoder_begin_callback`
- `clear_encoder_begin_callback`
- `assign_logits_filter_callback`
- `clear_logits_filter_callback`
- `assign_progress_callback`
- `assign_abort_callback`
- `clear_progress_callback`
- `clear_abort_callback`

### 3.10 VAD Types and Functions

- `whisper_vad_params`
- `whisper_vad_default_params`
- `whisper_vad_context_params`
- `whisper_vad_default_context_params`
- `whisper_vad_init_from_file_with_params`
- `whisper_vad_detect_speech`
- `whisper_vad_n_probs`
- `whisper_vad_probs`
- `whisper_vad_segments`
- `whisper_vad_segments_from_probs`
- `whisper_vad_segments_from_samples`
- `whisper_vad_segments_n_segments`
- `whisper_vad_segments_get_segment_t0`
- `whisper_vad_segments_get_segment_t1`
- `whisper_vad_free_segments`
- `whisper_vad_free`

### 3.11 Benchmark Helpers

- `whisper_bench_memcpy`
- `whisper_bench_ggml_mul_mat`
