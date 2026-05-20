#!/usr/bin/env python
# -*- coding: utf-8 -*-

# v1.4.1 compatibility test for the public Python surface:
# covers _pywhispercpp plus pywhispercpp.model Model/Segment behavior.
# Not a constants.py contract test, schema-parity test, or private implementation test.

# whisper.cpp supports more whisper_full_params fields than pywhispercpp v1.4.1 currently
# exposes through _pywhispercpp.whisper_full_params.
#
# Present upstream, but not exposed as normal Python binding fields right now:
# - no_timestamps
# - debug_mode
# - tdrz_enable
# - detect_language
# - new_segment_callback
# - progress_callback_user_data
# - encoder_begin_callback
# - abort_callback
# - abort_callback_user_data
# - logits_filter_callback
# - grammar_rules
# - n_grammar_rules
# - i_start_rule
# - grammar_penalty
#
# Notes:
# - prompt_tokens is exposed, but not properly supported for Python tuple/list assignment.
# - progress_callback is exposed, but its behavior is not clean/fully Python-friendly.
# - new_segment_callback is not exposed as a direct whisper_full_params field, but the
#   high-level Model API wires segment callbacks indirectly.

# whisper_full_params fields currently exposed by pywhispercpp v1.4.1:
# - strategy
# - n_threads
# - n_max_text_ctx
# - offset_ms
# - duration_ms
# - translate
# - no_context
# - single_segment
# - print_special
# - print_progress
# - progress_callback
# - print_realtime
# - print_timestamps
# - token_timestamps
# - thold_pt
# - thold_ptsum
# - max_len
# - split_on_word
# - max_tokens
# - audio_ctx
# - suppress_regex
# - initial_prompt
# - prompt_tokens
# - prompt_n_tokens
# - language
# - suppress_blank
# - temperature
# - max_initial_ts
# - length_penalty
# - temperature_inc
# - entropy_thold
# - logprob_thold
# - no_speech_thold
# - greedy
# - beam_search
# - new_segment_callback_user_data
# - encoder_begin_callback_user_data
# - logits_filter_callback_user_data
# - vad
# - vad_model_path
# - vad_params

import gc
import math
import shutil
import unittest
from unittest import TestCase

import _pywhispercpp as pw

from pywhispercpp.model import Model, Segment
from tests.compat._support import FailureSummaryTestCase, WHISPER_CPP_DIR, maybe_run_isolated_case


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


class TestBackwardsCompatibilityV141(FailureSummaryTestCase):
    module_name = 'tests.compat.v1_4_1'
    audio_file = WHISPER_CPP_DIR / 'samples/jfk.wav'
    mp3_audio_file = WHISPER_CPP_DIR / 'samples/jfk.mp3'
    models_dir = str(WHISPER_CPP_DIR / 'models')
    binding_missing_fields = {'suppress_non_speech_tokens'}
    binding_known_roundtrip_bugs = {
        'prompt_tokens': {
            # TypeError was the original v1.4.1 error. Since we have changed the binding it will now throw a RunTime error.
            'exception': (TypeError, RuntimeError),
            'reason': (
                'Known v1.4.1 bug: prompt_tokens is bound as a raw pointer in src/main.cpp; '
                'tuple assignment is not supported in v1.4.1'
            ),
        },
    }
    binding_roundtrip_values = {
        'n_threads': 2,
        'n_max_text_ctx': 128,
        'offset_ms': 1,
        'duration_ms': 100,
        'translate': True,
        'no_context': True,
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
        'audio_ctx': 16,
        'initial_prompt': 'test prompt',
        'prompt_tokens': ("just some", " mf ", "tokens"),
        'prompt_n_tokens': 3,
        'language': 'en',
        'suppress_blank': False,
        'temperature': 0.1,
        'max_initial_ts': 0.5,
        'length_penalty': 0.0,
        'temperature_inc': 0.1,
        'entropy_thold': 2.0,
        'logprob_thold': -0.5,
        'no_speech_thold': 0.5,
        'greedy': {'best_of': 2},
        'beam_search': {'beam_size': 2, 'patience': 0.5},
        'vad': True,
        'vad_model_path': 'vad.bin',
    }

    def tearDown(self):
        gc.collect()

    def _compat_model_constructor_accepts_model_name(self):
        model = Model('tiny', models_dir=str(WHISPER_CPP_DIR / 'models'))
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
        ):
            self.assertTrue(hasattr(pw, name), msg=name)

    def _compat_model_constructor_accepts_extended_options(self):
        model = Model(
            'tiny',
            models_dir=str(WHISPER_CPP_DIR / 'models'),
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
            models_dir=str(WHISPER_CPP_DIR / 'models'),
            n_threads=2,
            language='en',
            single_segment=True,
        )
        params = model.get_params()
        self.assertEqual(params['n_threads'], 2)
        self.assertEqual(params['language'], 'en')
        self.assertIs(params['single_segment'], True)

    def _compat_model_get_params_exposes_binding_subset(self):
        model = Model(
            'tiny',
            models_dir=str(WHISPER_CPP_DIR / 'models'),
            n_threads=2,
            language='en',
            initial_prompt='hello',
        )
        params = model.get_params()
        self.assertEqual(params['n_threads'], 2)
        self.assertEqual(params['language'], 'en')
        self.assertEqual(params['initial_prompt'], 'hello')
        self.assertNotIn('extract_probability', params)
        self.assertNotIn('suppress_non_speech_tokens', params)
        self.assertNotIn('progress_callback', params)

    def _compat_model_rejects_unknown_constructor_kwargs(self):
        with self.assertRaises(AttributeError):
            Model(
                'tiny',
                models_dir=str(WHISPER_CPP_DIR / 'models'),
                definitely_not_a_real_param=True,
            )

    def _compat_model_rejects_unknown_transcribe_kwargs(self):
        model = Model('tiny', models_dir=str(WHISPER_CPP_DIR / 'models'))
        with self.assertRaises(AttributeError):
            model.transcribe(
                str(WHISPER_CPP_DIR / 'samples/jfk.wav'),
                definitely_not_a_real_param=True,
            )

    def _compat_transcribe_returns_segments(self):
        model = Model('tiny', models_dir=str(WHISPER_CPP_DIR / 'models'))
        segments = model.transcribe(str(WHISPER_CPP_DIR / 'samples/jfk.wav'))
        self.assertIsInstance(segments, list)
        self.assertGreater(len(segments), 0)
        self.assertTrue(all(isinstance(segment, Segment) for segment in segments))

    def _compat_new_segment_callback_receives_segments(self):
        seen = []
        model = Model('tiny', models_dir=str(WHISPER_CPP_DIR / 'models'))

        def on_segment(segment):
            seen.append(segment)

        segments = model.transcribe(
            str(WHISPER_CPP_DIR / 'samples/jfk.wav'),
            new_segment_callback=on_segment,
        )
        self.assertIsInstance(segments, list)
        self.assertGreater(len(seen), 0)
        self.assertTrue(all(isinstance(segment, Segment) for segment in seen))

    def _compat_transcribe_accepts_numpy_audio(self):
        model = Model('tiny', models_dir=str(WHISPER_CPP_DIR / 'models'))
        audio = model._load_audio(str(WHISPER_CPP_DIR / 'samples/jfk.wav'))
        segments = model.transcribe(audio)
        self.assertIsInstance(segments, list)
        self.assertGreater(len(segments), 0)
        self.assertTrue(all(isinstance(segment, Segment) for segment in segments))

    def _compat_transcribe_accepts_parallel_processors(self):
        model = Model('tiny', models_dir=str(WHISPER_CPP_DIR / 'models'))
        segments = model.transcribe(
            str(WHISPER_CPP_DIR / 'samples/jfk.wav'),
            n_processors=2,
        )
        self.assertIsInstance(segments, list)
        self.assertGreater(len(segments), 0)

    def _compat_transcribe_accepts_non_wav_media_via_ffmpeg(self):
        if shutil.which('ffmpeg') is None:
            self.skipTest('ffmpeg is required for non-wav media compatibility coverage')

        model = Model('tiny', models_dir=str(WHISPER_CPP_DIR / 'models'))
        segments = model.transcribe(str(self.mp3_audio_file))
        self.assertIsInstance(segments, list)
        self.assertGreater(len(segments), 0)

    def _compat_transcribe_missing_file_raises_file_not_found(self):
        model = Model('tiny', models_dir=str(WHISPER_CPP_DIR / 'models'))
        with self.assertRaises(FileNotFoundError):
            model.transcribe(str(WHISPER_CPP_DIR / 'samples/missing.wav'))

    def _compat_segment_instances_expose_expected_attributes(self):
        model = Model('tiny', models_dir=str(WHISPER_CPP_DIR / 'models'))
        segment = model.transcribe(str(WHISPER_CPP_DIR / 'samples/jfk.wav'))[0]
        self.assertIsInstance(segment.t0, int)
        self.assertIsInstance(segment.t1, int)
        self.assertIsInstance(segment.text, str)
        self.assertIsInstance(float(segment.probability), float)

    def _compat_available_languages_include_english(self):
        languages = Model.available_languages()
        self.assertIsInstance(languages, list)
        self.assertIn('en', languages)

    def _compat_auto_detect_language_returns_language_and_probabilities(self):
        model = Model('tiny', models_dir=str(WHISPER_CPP_DIR / 'models'))
        detected_language, probabilities = model.auto_detect_language(
            str(WHISPER_CPP_DIR / 'samples/jfk.wav')
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
        model = Model('tiny', models_dir=str(WHISPER_CPP_DIR / 'models'))
        model.transcribe(str(WHISPER_CPP_DIR / 'samples/jfk.wav'))
        self.assertIsNone(model.print_timings())

    def _compat_transcribe_accepts_extract_probability(self):
        model = Model('tiny', models_dir=str(WHISPER_CPP_DIR / 'models'))
        segments = model.transcribe(
            str(WHISPER_CPP_DIR / 'samples/jfk.wav'),
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
                bug = self.binding_known_roundtrip_bugs.get(name)
                try:
                    setattr(params, name, value)
                except Exception as exc:
                    if bug and isinstance(exc, bug['exception']):
                        self.skipTest(
                            f"{bug['reason']}. Original error: {exc}"
                        )
                    raise

                actual = getattr(params, name)
                _assert_param_value(self, name, actual, value)

    def _compat_cpp_binding_prompt_tokens_tuple_assignment_is_not_supported(self):
        params = pw.whisper_full_default_params(
            pw.whisper_sampling_strategy.WHISPER_SAMPLING_GREEDY
        )

        self.assertTrue(hasattr(params, 'prompt_tokens'))
        with self.assertRaises((TypeError, RuntimeError)):
            params.prompt_tokens = self.binding_roundtrip_values['prompt_tokens']

    def _compat_cpp_binding_string_properties_round_trip(self):
        params = pw.whisper_full_default_params(
            pw.whisper_sampling_strategy.WHISPER_SAMPLING_GREEDY
        )
        params.suppress_regex = 'foo.*bar'
        params.vad_model_path = 'vad.bin'
        self.assertEqual(params.suppress_regex, 'foo.*bar')
        self.assertEqual(params.vad_model_path, 'vad.bin')

    def test_v1_4_1_model_constructor_accepts_model_name(self):
        self._compat_model_constructor_accepts_model_name()

    def test_v1_4_1_binding_module_exposes_expected_constants(self):
        self._compat_binding_module_exposes_expected_constants()

    def test_v1_4_1_binding_module_exposes_expected_low_level_functions(self):
        self._compat_binding_module_exposes_expected_low_level_functions()

    def test_v1_4_1_cpp_binding_missing_fields_match_contract(self):
        self._compat_cpp_binding_missing_fields_match_contract()

    def test_v1_4_1_cpp_binding_round_trips_supported_params(self):
        self._compat_cpp_binding_round_trips_supported_params()

    # def test_v1_4_1_cpp_binding_prompt_tokens_tuple_assignment_is_not_supported(self):
    #     self._compat_cpp_binding_prompt_tokens_tuple_assignment_is_not_supported()

    def test_v1_4_1_cpp_binding_string_properties_round_trip(self):
        self._compat_cpp_binding_string_properties_round_trip()

    def test_v1_4_1_model_constructor_accepts_extended_options(self):
        self._compat_model_constructor_accepts_extended_options()

    def test_v1_4_1_model_constructor_accepts_supported_kwargs(self):
        self._compat_model_constructor_accepts_supported_kwargs()

    def test_v1_4_1_model_get_params_exposes_binding_subset(self):
        self._compat_model_get_params_exposes_binding_subset()

    def test_v1_4_1_model_rejects_unknown_transcribe_kwargs(self):
        self._run_case('_compat_model_rejects_unknown_transcribe_kwargs')

    def test_v1_4_1_transcribe_returns_segments(self):
        self._run_case('_compat_transcribe_returns_segments')

    def test_v1_4_1_new_segment_callback_receives_segments(self):
        self._run_case('_compat_new_segment_callback_receives_segments')

    def test_v1_4_1_transcribe_accepts_numpy_audio(self):
        self._run_case('_compat_transcribe_accepts_numpy_audio')

    def test_v1_4_1_transcribe_accepts_parallel_processors(self):
        self._run_case('_compat_transcribe_accepts_parallel_processors')

    def test_v1_4_1_transcribe_accepts_non_wav_media_via_ffmpeg(self):
        self._run_case('_compat_transcribe_accepts_non_wav_media_via_ffmpeg')

    def test_v1_4_1_transcribe_missing_file_raises_file_not_found(self):
        self._compat_transcribe_missing_file_raises_file_not_found()

    def test_v1_4_1_segment_instances_expose_expected_attributes(self):
        self._run_case('_compat_segment_instances_expose_expected_attributes')

    def test_v1_4_1_available_languages_include_english(self):
        self._compat_available_languages_include_english()

    def test_v1_4_1_lang_max_id_matches_available_languages(self):
        self._compat_lang_max_id_matches_available_languages()

    def test_v1_4_1_auto_detect_language_returns_language_and_probabilities(self):
        self._run_case('_compat_auto_detect_language_returns_language_and_probabilities')

    def test_v1_4_1_system_info_returns_text(self):
        self._compat_system_info_returns_text()

    def test_v1_4_1_print_timings_is_callable_after_transcribe(self):
        self._run_case('_compat_print_timings_is_callable_after_transcribe')

    def test_v1_4_1_transcribe_accepts_extract_probability(self):
        self._run_case('_compat_transcribe_accepts_extract_probability')


if __name__ == '__main__':
    maybe_run_isolated_case(globals())
    unittest.main()
