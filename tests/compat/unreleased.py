#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Unreleased compatibility test for the public Python surface:
# covers _pywhispercpp plus pywhispercpp.model Model/Segment behavior.
# Not a constants.py contract test, schema-parity test, or private implementation test.


import gc
import math
import shutil
import unittest
from unittest import TestCase

import _pywhispercpp as pw

from pywhispercpp.model import Model, Segment
from tests.compat._support import FailureSummaryTestCase, WHISPER_CPP_DIR, cpu_context_params, maybe_run_isolated_case


def _assert_param_value(test_case: TestCase, name: str, actual, expected):
    if isinstance(expected, tuple):
        try:
            actual_cmp = tuple(actual)
        except Exception:
            actual_cmp = actual
    else:
        actual_cmp = actual

    if isinstance(expected, float) and isinstance(actual_cmp, float):
        test_case.assertTrue(
            math.isclose(actual_cmp, expected, rel_tol=1e-6, abs_tol=1e-6),
            msg=f'{name}: expected {expected!r}, got {actual_cmp!r}',
        )
        return

    test_case.assertEqual(actual_cmp, expected, msg=f'{name}: expected {expected!r}, got {actual_cmp!r}')


class TestBackwardsCompatibilityVUnreleased(FailureSummaryTestCase):
    module_name = 'tests.compat.unreleased'
    audio_file = WHISPER_CPP_DIR / 'samples/jfk.wav'
    mp3_audio_file = WHISPER_CPP_DIR / 'samples/jfk.mp3'
    models_dir = str(WHISPER_CPP_DIR / 'models')
    binding_missing_fields = {
        'new_segment_callback',
        'encoder_begin_callback',
        'logits_filter_callback',
        'suppress_non_speech_tokens',
    }
    binding_roundtrip_values = {
        'n_threads': 2,
        'n_max_text_ctx': 128,
        'offset_ms': 1,
        'duration_ms': 100,
        'translate': True,
        'no_context': True,
        'no_timestamps': True,
        'single_segment': True,
        'print_special': True,
        'print_progress': False,
        'print_realtime': True,
        'print_timestamps': False,
        'token_timestamps': True,
        'thold_pt': 0.02,
        'thold_ptsum': 0.03,
        'max_len': 10,
        'split_on_word': True,
        'max_tokens': 8,
        'debug_mode': True,
        'audio_ctx': 16,
        'tdrz_enable': True,
        'initial_prompt': 'test prompt',
        'carry_initial_prompt': True,
        'prompt_n_tokens': 3,
        'language': 'en',
        'detect_language': True,
        'suppress_blank': False,
        'suppress_nst': True,
        'temperature': 0.1,
        'max_initial_ts': 0.5,
        'length_penalty': 0.0,
        'temperature_inc': 0.1,
        'entropy_thold': 2.0,
        'logprob_thold': -0.5,
        'no_speech_thold': 0.5,
        'greedy': {'best_of': 2},
        'beam_search': {'beam_size': 2, 'patience': 0.5},
        'grammar_penalty': 100.0,
        'vad': True,
        'vad_model_path': 'vad.bin',
    }

    def tearDown(self):
        gc.collect()

    def _create_cpu_model(self):
        return Model(
            'tiny',
            models_dir=self.models_dir,
            context_params=cpu_context_params(),
        )

    def _compat_model_constructor_accepts_model_name(self):
        model = Model('tiny', models_dir=self.models_dir)
        self.assertIsInstance(model, Model)

    def _compat_binding_module_exposes_expected_constants(self):
        self.assertGreater(pw.WHISPER_N_FFT, 0)
        self.assertGreater(pw.WHISPER_HOP_LENGTH, 0)
        self.assertGreater(pw.WHISPER_CHUNK_SIZE, 0)

    def _compat_binding_module_exposes_expected_low_level_functions(self):
        for name in (
            'whisper_init_from_buffer',
            'whisper_print_system_info',
            'whisper_tokenize',
            'whisper_token_to_bytes',
            'whisper_bench_memcpy',
            'whisper_vad_default_params',
            'whisper_context_default_params',
            'assign_progress_callback',
            'clear_progress_callback',
            'assign_abort_callback',
            'clear_abort_callback',
            'whisper_log_set',
        ):
            self.assertTrue(hasattr(pw, name), msg=name)

    def _compat_model_constructor_accepts_extended_options(self):
        model = Model(
            'tiny',
            models_dir=self.models_dir,
            params_sampling_strategy=1,
            redirect_whispercpp_logs_to=None,
            use_openvino=False,
            openvino_model_path='openvino-model.xml',
            openvino_device='CPU',
            openvino_cache_dir='/tmp/pywhispercpp-openvino-cache',
        )
        self.assertEqual(
            model._sampling_strategy, # type: ignore
            pw.whisper_sampling_strategy.WHISPER_SAMPLING_BEAM_SEARCH,
        )
        self.assertIsNone(model.redirect_whispercpp_logs_to) # type: ignore
        self.assertFalse(model.use_openvino) # type: ignore
        self.assertEqual(model.openvino_model_path, 'openvino-model.xml') # type: ignore
        self.assertEqual(model.openvino_device, 'CPU') # type: ignore
        self.assertEqual(model.openvino_cache_dir, '/tmp/pywhispercpp-openvino-cache') # type: ignore

    def _compat_model_constructor_accepts_supported_kwargs(self):
        model = Model(
            'tiny',
            models_dir=self.models_dir,
            n_threads=2,
            language='en',
            single_segment=True,
            detect_language=True,
        )
        params = model.get_params()
        self.assertEqual(params['n_threads'], 2)
        self.assertEqual(params['language'], 'en')
        self.assertIs(params['single_segment'], True)
        self.assertIs(params['detect_language'], True)

    def _compat_model_constructor_accepts_context_params(self):
        model = self._create_cpu_model()
        self.assertIsInstance(model, Model)

    def _compat_model_constructor_accepts_partial_context_params(self):
        model = Model(
            'tiny',
            models_dir=self.models_dir,
            context_params={'use_gpu': False},
        )
        self.assertIsInstance(model, Model)

    def _compat_model_constructor_accepts_suppress_non_speech_tokens_alias(self):
        model = Model(
            'tiny',
            models_dir=self.models_dir,
            suppress_non_speech_tokens=True,
        )
        params = model.get_params()
        self.assertIs(params['suppress_nst'], True)
        self.assertNotIn('suppress_non_speech_tokens', params)

    def _compat_model_get_params_exposes_binding_subset(self):
        model = Model(
            'tiny',
            models_dir=self.models_dir,
            n_threads=2,
            language='en',
            initial_prompt='hello',
            suppress_non_speech_tokens=True,
        )
        params = model.get_params()
        self.assertEqual(params['n_threads'], 2)
        self.assertEqual(params['language'], 'en')
        self.assertEqual(params['initial_prompt'], 'hello')
        self.assertTrue(params['suppress_nst'])
        self.assertNotIn('extract_probability', params)
        self.assertNotIn('suppress_non_speech_tokens', params)
        self.assertNotIn('progress_callback', params)

    def _compat_context_params_reject_invalid_input(self):
        with self.assertRaises(TypeError):
            Model(
                'tiny',
                models_dir=self.models_dir,
                context_params=object(),  # type: ignore[arg-type]
            )

    def _compat_model_rejects_unknown_constructor_kwargs(self):
        with self.assertRaises(AttributeError):
            Model(
                'tiny',
                models_dir=self.models_dir,
                definitely_not_a_real_param=True,
            )

    def _compat_model_rejects_unknown_transcribe_kwargs(self):
        model = Model('tiny', models_dir=self.models_dir)
        with self.assertRaises(AttributeError):
            model.transcribe(
                str(self.audio_file),
                definitely_not_a_real_param=True,
            )

    def _compat_transcribe_returns_segments(self):
        model = Model('tiny', models_dir=self.models_dir)
        segments = model.transcribe(str(self.audio_file))
        self.assertIsInstance(segments, list)
        self.assertGreater(len(segments), 0)
        self.assertTrue(all(isinstance(segment, Segment) for segment in segments))

    def _compat_new_segment_callback_receives_segments(self):
        seen = []
        model = Model('tiny', models_dir=self.models_dir)

        def on_segment(segment):
            seen.append(segment)

        segments = model.transcribe(
            str(self.audio_file),
            new_segment_callback=on_segment,
        )
        self.assertIsInstance(segments, list)
        self.assertGreater(len(seen), 0)
        self.assertTrue(all(isinstance(segment, Segment) for segment in seen))

    def _compat_transcribe_accepts_numpy_audio(self):
        model = Model('tiny', models_dir=self.models_dir)
        audio = model._load_audio(str(self.audio_file))
        segments = model.transcribe(audio)
        self.assertIsInstance(segments, list)
        self.assertGreater(len(segments), 0)
        self.assertTrue(all(isinstance(segment, Segment) for segment in segments))

    def _compat_transcribe_accepts_parallel_processors(self):
        model = Model('tiny', models_dir=self.models_dir)
        segments = model.transcribe(
            str(self.audio_file),
            n_processors=2,
        )
        self.assertIsInstance(segments, list)
        self.assertGreater(len(segments), 0)

    def _compat_transcribe_accepts_non_wav_media_via_ffmpeg(self):
        if shutil.which('ffmpeg') is None:
            self.skipTest('ffmpeg is required for non-wav media compatibility coverage')

        model = Model('tiny', models_dir=self.models_dir)
        segments = model.transcribe(str(self.mp3_audio_file))
        self.assertIsInstance(segments, list)
        self.assertGreater(len(segments), 0)

    def _compat_transcribe_missing_file_raises_file_not_found(self):
        model = Model('tiny', models_dir=self.models_dir)
        with self.assertRaises(FileNotFoundError):
            model.transcribe(str(WHISPER_CPP_DIR / 'samples/missing.wav'))

    def _compat_segment_instances_expose_expected_attributes(self):
        model = Model('tiny', models_dir=self.models_dir)
        segment = model.transcribe(str(self.audio_file))[0]
        self.assertIsInstance(segment.t0, int)
        self.assertIsInstance(segment.t1, int)
        self.assertIsInstance(segment.text, str)
        self.assertIsInstance(float(segment.probability), float)

    def _compat_available_languages_include_english(self):
        languages = Model.available_languages()
        self.assertIsInstance(languages, list)
        self.assertIn('en', languages)

    def _compat_auto_detect_language_returns_language_and_probabilities(self):
        model = Model('tiny', models_dir=self.models_dir)
        detected_language, probabilities = model.auto_detect_language(
            str(self.audio_file)
        )
        self.assertEqual(detected_language[0], 'en')
        self.assertIsInstance(probabilities, dict)
        self.assertIn('en', probabilities)

    def _compat_lang_max_id_matches_available_languages(self):
        self.assertEqual(Model.lang_max_id() + 1, len(Model.available_languages()))

    def _compat_system_info_returns_text(self):
        info = Model.system_info()
        self.assertIsInstance(info, str)
        self.assertGreater(len(str(info)), 0)

    def _compat_print_timings_is_callable_after_transcribe(self):
        model = Model('tiny', models_dir=self.models_dir)
        model.transcribe(str(self.audio_file))
        self.assertIsNone(model.print_timings())

    def _compat_transcribe_accepts_extract_probability(self):
        model = Model('tiny', models_dir=self.models_dir)
        segments = model.transcribe(
            str(self.audio_file),
            extract_probability=True,
        )
        self.assertGreater(len(segments), 0)
        self.assertTrue(any(not math.isnan(float(segment.probability)) for segment in segments))

    def _compat_cpp_binding_missing_fields_match_contract(self):
        params = pw.whisper_full_default_params(
            pw.whisper_sampling_strategy.WHISPER_SAMPLING_GREEDY
        )

        missing = {
            name for name in self.binding_missing_fields
            if not hasattr(params, name)
        }

        self.assertEqual(missing, self.binding_missing_fields)

    def _compat_cpp_binding_round_trips_supported_params(self):
        params = pw.whisper_full_default_params(
            pw.whisper_sampling_strategy.WHISPER_SAMPLING_GREEDY
        )

        for name, value in self.binding_roundtrip_values.items():
            with self.subTest(param=name):
                setattr(params, name, value)
                actual = getattr(params, name)
                _assert_param_value(self, name, actual, value)

    def _compat_cpp_binding_prompt_tokens_round_trip(self):
        params = pw.whisper_full_default_params(
            pw.whisper_sampling_strategy.WHISPER_SAMPLING_GREEDY
        )

        self.assertTrue(hasattr(params, 'prompt_tokens'))
        self.assertTrue(hasattr(params, 'set_prompt_tokens'))

        params.set_prompt_tokens((1, 2, 3))
        self.assertEqual(tuple(params.prompt_tokens), (1, 2, 3))
        self.assertEqual(params.prompt_n_tokens, 3)

        params.prompt_tokens = (4, 5)
        self.assertEqual(tuple(params.prompt_tokens), (4, 5))
        self.assertEqual(params.prompt_n_tokens, 2)

        params.prompt_tokens = None
        self.assertEqual(tuple(params.prompt_tokens), ()) # type: ignore
        self.assertEqual(params.prompt_n_tokens, 0)

    def _compat_cpp_binding_string_properties_round_trip(self):
        params = pw.whisper_full_default_params(
            pw.whisper_sampling_strategy.WHISPER_SAMPLING_GREEDY
        )
        params.suppress_regex = 'foo.*bar'
        params.vad_model_path = 'vad.bin'
        self.assertEqual(params.suppress_regex, 'foo.*bar')
        self.assertEqual(params.vad_model_path, 'vad.bin')

    def _compat_cpp_binding_callback_user_data_round_trip(self):
        params = pw.whisper_full_default_params(
            pw.whisper_sampling_strategy.WHISPER_SAMPLING_GREEDY
        )

        new_segment_user_data = {'callback': 'new-segment'}
        encoder_begin_user_data = ('encoder-begin', 1)
        progress_user_data = {'callback': 'progress'}
        logits_filter_user_data = ['logits-filter']
        abort_user_data = ('abort', 1)

        params.new_segment_callback_user_data = new_segment_user_data
        params.encoder_begin_callback_user_data = encoder_begin_user_data
        params.progress_callback_user_data = progress_user_data
        params.logits_filter_callback_user_data = logits_filter_user_data
        params.abort_callback_user_data = abort_user_data

        self.assertEqual(params.new_segment_callback_user_data, new_segment_user_data)
        self.assertEqual(params.encoder_begin_callback_user_data, encoder_begin_user_data)
        self.assertEqual(params.progress_callback_user_data, progress_user_data)
        self.assertEqual(params.logits_filter_callback_user_data, logits_filter_user_data)
        self.assertEqual(params.abort_callback_user_data, abort_user_data)

    def _compat_cpp_binding_helper_only_callback_helpers_exist(self):
        params = pw.whisper_full_default_params(
            pw.whisper_sampling_strategy.WHISPER_SAMPLING_GREEDY
        )

        params.new_segment_callback_user_data = {'source': 'new-segment'}
        params.encoder_begin_callback_user_data = {'source': 'encoder-begin'}
        params.logits_filter_callback_user_data = {'source': 'logits-filter'}

        def on_new_segment(ctx, n_new, user_data=None):
            return None

        def on_encoder_begin(ctx, user_data=None):
            return True

        def on_logits_filter(ctx, n_tokens, logits, user_data=None):
            return None

        params.set_new_segment_callback(on_new_segment)
        params.set_encoder_begin_callback(on_encoder_begin)
        params.set_logits_filter_callback(on_logits_filter)

        pw.assign_new_segment_callback(params, on_new_segment)
        pw.assign_encoder_begin_callback(params, on_encoder_begin)
        pw.assign_logits_filter_callback(params, on_logits_filter)

        params.clear_new_segment_callback()
        params.clear_encoder_begin_callback()
        params.clear_logits_filter_callback()

        pw.clear_new_segment_callback(params)
        pw.clear_encoder_begin_callback(params)
        pw.clear_logits_filter_callback(params)

        self.assertEqual(params.new_segment_callback_user_data, {'source': 'new-segment'})
        self.assertEqual(params.encoder_begin_callback_user_data, {'source': 'encoder-begin'})
        self.assertEqual(params.logits_filter_callback_user_data, {'source': 'logits-filter'})

    def _compat_cpp_binding_progress_callback_helpers_exist(self):
        params = pw.whisper_full_default_params(
            pw.whisper_sampling_strategy.WHISPER_SAMPLING_GREEDY
        )

        seen = []

        def on_progress(progress, user_data=None):
            seen.append((progress, user_data))

        params.progress_callback_user_data = {'source': 'progress'}
        params.set_progress_callback(on_progress)
        self.assertTrue(callable(params.progress_callback))

        pw.assign_progress_callback(params, on_progress)
        self.assertTrue(callable(params.progress_callback))

        params.clear_progress_callback()
        pw.clear_progress_callback(params)
        self.assertEqual(params.progress_callback_user_data, {'source': 'progress'})

    def _compat_cpp_binding_callbacks_omit_user_data_when_unset(self):
        model = self._create_cpu_model()
        audio = model._load_audio(str(self.audio_file))
        params = pw.whisper_full_default_params(
            pw.whisper_sampling_strategy.WHISPER_SAMPLING_GREEDY
        )
        seen = {
            'new_segment': 0,
            'encoder_begin': 0,
            'logits_filter': 0,
            'progress': 0,
            'abort': 0,
        }

        def on_new_segment(ctx, n_new):
            self.assertIsInstance(n_new, int)
            seen['new_segment'] += 1

        def on_encoder_begin(ctx):
            seen['encoder_begin'] += 1
            return True

        def on_logits_filter(ctx, n_tokens, logits):
            self.assertIsInstance(n_tokens, int)
            seen['logits_filter'] += 1

        def on_progress(progress):
            self.assertIsInstance(progress, int)
            seen['progress'] += 1

        def on_abort():
            seen['abort'] += 1
            return False

        params.set_new_segment_callback(on_new_segment)
        params.set_encoder_begin_callback(on_encoder_begin)
        params.set_logits_filter_callback(on_logits_filter)
        params.set_progress_callback(on_progress)
        params.set_abort_callback(on_abort)

        result = pw.whisper_full(model._ctx, params, audio, len(audio)) # type: ignore

        self.assertEqual(result, 0)
        self.assertGreater(seen['new_segment'], 0)
        self.assertGreater(seen['encoder_begin'], 0)
        self.assertGreater(seen['logits_filter'], 0)
        self.assertGreater(seen['progress'], 0)
        self.assertGreater(seen['abort'], 0)

    def _compat_cpp_binding_callbacks_pass_user_data_when_set(self):
        model = self._create_cpu_model()
        audio = model._load_audio(str(self.audio_file))
        params = pw.whisper_full_default_params(
            pw.whisper_sampling_strategy.WHISPER_SAMPLING_GREEDY
        )

        new_segment_user_data = {'source': 'new-segment'}
        encoder_begin_user_data = {'source': 'encoder-begin'}
        logits_filter_user_data = {'source': 'logits-filter'}
        progress_user_data = {'source': 'progress'}
        abort_user_data = {'source': 'abort'}

        seen = {
            'new_segment': 0,
            'encoder_begin': 0,
            'logits_filter': 0,
            'progress': 0,
            'abort': 0,
        }

        def on_new_segment(ctx, n_new, user_data):
            self.assertEqual(user_data, new_segment_user_data)
            seen['new_segment'] += 1

        def on_encoder_begin(ctx, user_data):
            self.assertEqual(user_data, encoder_begin_user_data)
            seen['encoder_begin'] += 1
            return True

        def on_logits_filter(ctx, n_tokens, logits, user_data):
            self.assertEqual(user_data, logits_filter_user_data)
            seen['logits_filter'] += 1

        def on_progress(progress, user_data):
            self.assertEqual(user_data, progress_user_data)
            seen['progress'] += 1

        def on_abort(user_data):
            self.assertEqual(user_data, abort_user_data)
            seen['abort'] += 1
            return False

        params.new_segment_callback_user_data = new_segment_user_data
        params.encoder_begin_callback_user_data = encoder_begin_user_data
        params.logits_filter_callback_user_data = logits_filter_user_data
        params.progress_callback_user_data = progress_user_data
        params.abort_callback_user_data = abort_user_data
        params.set_new_segment_callback(on_new_segment)
        params.set_encoder_begin_callback(on_encoder_begin)
        params.set_logits_filter_callback(on_logits_filter)
        params.set_progress_callback(on_progress)
        params.set_abort_callback(on_abort)

        result = pw.whisper_full(model._ctx, params, audio, len(audio)) # type: ignore

        self.assertEqual(result, 0)
        self.assertGreater(seen['new_segment'], 0)
        self.assertGreater(seen['encoder_begin'], 0)
        self.assertGreater(seen['logits_filter'], 0)
        self.assertGreater(seen['progress'], 0)
        self.assertGreater(seen['abort'], 0)

    def _compat_cpp_binding_grammar_helpers_expose_metadata(self):
        params = pw.whisper_full_default_params(
            pw.whisper_sampling_strategy.WHISPER_SAMPLING_GREEDY
        )

        params.set_grammar('root ::= "yes" | "no"', 'root', 42.0)

        self.assertIsInstance(params.grammar_rules, list)
        self.assertGreater(len(params.grammar_rules), 0)
        self.assertGreater(params.n_grammar_rules, 0)
        self.assertIsInstance(params.i_start_rule, int)
        self.assertGreaterEqual(params.i_start_rule, 0)
        self.assertEqual(params.grammar_penalty, 42.0)

        params.clear_grammar()

        self.assertEqual(params.grammar_rules, [])
        self.assertEqual(params.n_grammar_rules, 0)
        self.assertEqual(params.i_start_rule, 0)

    def _compat_abort_callback_can_abort_and_then_clear(self):
        model = self._create_cpu_model()
        callback_calls = []

        def abort_immediately():
            callback_calls.append(True)
            return True

        aborted_segments = model.transcribe(
            str(self.audio_file),
            abort_callback=abort_immediately,
        )
        self.assertIsInstance(aborted_segments, list)
        self.assertGreater(len(callback_calls), 0)

        normal_segments = model.transcribe(str(self.audio_file))
        self.assertIsInstance(normal_segments, list)
        self.assertGreater(len(normal_segments), 0)

    def _compat_log_callback_can_be_set_and_cleared(self):
        pw.whisper_log_set(lambda level, text: None)
        pw.whisper_log_set(None)

    def _compat_alignment_preset_enum_is_available(self):
        preset = pw.whisper_alignment_heads_preset.WHISPER_AHEADS_TINY
        self.assertIsNotNone(preset)

    def test_unreleased_model_constructor_accepts_model_name(self):
        self._compat_model_constructor_accepts_model_name()

    def test_unreleased_binding_module_exposes_expected_constants(self):
        self._compat_binding_module_exposes_expected_constants()

    def test_unreleased_binding_module_exposes_expected_low_level_functions(self):
        self._compat_binding_module_exposes_expected_low_level_functions()

    def test_unreleased_cpp_binding_missing_fields_match_contract(self):
        self._compat_cpp_binding_missing_fields_match_contract()

    def test_unreleased_cpp_binding_round_trips_supported_params(self):
        self._compat_cpp_binding_round_trips_supported_params()

    def test_unreleased_cpp_binding_prompt_tokens_round_trip(self):
        self._compat_cpp_binding_prompt_tokens_round_trip()

    def test_unreleased_cpp_binding_string_properties_round_trip(self):
        self._compat_cpp_binding_string_properties_round_trip()

    def test_unreleased_cpp_binding_callback_user_data_round_trip(self):
        self._compat_cpp_binding_callback_user_data_round_trip()

    def test_unreleased_cpp_binding_helper_only_callback_helpers_exist(self):
        self._compat_cpp_binding_helper_only_callback_helpers_exist()

    def test_unreleased_cpp_binding_progress_callback_helpers_exist(self):
        self._compat_cpp_binding_progress_callback_helpers_exist()

    def test_unreleased_cpp_binding_callbacks_omit_user_data_when_unset(self):
        self._compat_cpp_binding_callbacks_omit_user_data_when_unset()

    def test_unreleased_cpp_binding_callbacks_pass_user_data_when_set(self):
        self._compat_cpp_binding_callbacks_pass_user_data_when_set()

    def test_unreleased_cpp_binding_grammar_helpers_expose_metadata(self):
        self._compat_cpp_binding_grammar_helpers_expose_metadata()

    def test_unreleased_model_constructor_accepts_extended_options(self):
        self._compat_model_constructor_accepts_extended_options()

    def test_unreleased_model_constructor_accepts_supported_kwargs(self):
        self._compat_model_constructor_accepts_supported_kwargs()

    def test_unreleased_model_constructor_accepts_context_params(self):
        self._compat_model_constructor_accepts_context_params()

    def test_unreleased_model_constructor_accepts_partial_context_params(self):
        self._compat_model_constructor_accepts_partial_context_params()

    def test_unreleased_model_constructor_accepts_suppress_non_speech_tokens_alias(self):
        self._compat_model_constructor_accepts_suppress_non_speech_tokens_alias()

    def test_unreleased_model_get_params_exposes_binding_subset(self):
        self._compat_model_get_params_exposes_binding_subset()

    def test_unreleased_context_params_reject_invalid_input(self):
        self._compat_context_params_reject_invalid_input()

    def test_unreleased_model_rejects_unknown_constructor_kwargs(self):
        self._compat_model_rejects_unknown_constructor_kwargs()

    def test_unreleased_model_rejects_unknown_transcribe_kwargs(self):
        self._run_case('_compat_model_rejects_unknown_transcribe_kwargs')

    def test_unreleased_transcribe_returns_segments(self):
        self._run_case('_compat_transcribe_returns_segments')

    def test_unreleased_new_segment_callback_receives_segments(self):
        self._run_case('_compat_new_segment_callback_receives_segments')

    def test_unreleased_transcribe_accepts_numpy_audio(self):
        self._run_case('_compat_transcribe_accepts_numpy_audio')

    def test_unreleased_transcribe_accepts_parallel_processors(self):
        self._run_case('_compat_transcribe_accepts_parallel_processors')

    def test_unreleased_transcribe_accepts_non_wav_media_via_ffmpeg(self):
        self._run_case('_compat_transcribe_accepts_non_wav_media_via_ffmpeg')

    def test_unreleased_transcribe_missing_file_raises_file_not_found(self):
        self._compat_transcribe_missing_file_raises_file_not_found()

    def test_unreleased_segment_instances_expose_expected_attributes(self):
        self._run_case('_compat_segment_instances_expose_expected_attributes')

    def test_unreleased_available_languages_include_english(self):
        self._compat_available_languages_include_english()

    def test_unreleased_lang_max_id_matches_available_languages(self):
        self._compat_lang_max_id_matches_available_languages()

    def test_unreleased_auto_detect_language_returns_language_and_probabilities(self):
        self._run_case('_compat_auto_detect_language_returns_language_and_probabilities')

    def test_unreleased_system_info_returns_text(self):
        self._compat_system_info_returns_text()

    def test_unreleased_print_timings_is_callable_after_transcribe(self):
        self._run_case('_compat_print_timings_is_callable_after_transcribe')

    def test_unreleased_transcribe_accepts_extract_probability(self):
        self._run_case('_compat_transcribe_accepts_extract_probability')

    def test_unreleased_abort_callback_can_abort_and_then_clear(self):
        self._run_case('_compat_abort_callback_can_abort_and_then_clear')

    def test_unreleased_log_callback_can_be_set_and_cleared(self):
        self._compat_log_callback_can_be_set_and_cleared()

    def test_unreleased_alignment_preset_enum_is_available(self):
        self._compat_alignment_preset_enum_is_available()


if __name__ == '__main__':
    maybe_run_isolated_case(globals())
    unittest.main()
