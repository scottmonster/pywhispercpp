"""

        Pywhispercpp: Python binding to whisper.cpp
        -----------------------

        .. currentmodule:: _whispercpp

        .. autosummary::
           :toctree: _generate

    
"""
from __future__ import annotations
import numpy
import numpy.typing
import types
import typing
__all__: list[str] = ['WHISPER_CHUNK_SIZE', 'WHISPER_HOP_LENGTH', 'WHISPER_N_FFT', 'WHISPER_SAMPLE_RATE', 'WHISPER_SAMPLING_BEAM_SEARCH', 'WHISPER_SAMPLING_GREEDY', 'assign_encoder_begin_callback', 'assign_logits_filter_callback', 'assign_new_segment_callback', 'whisper_bench_ggml_mul_mat', 'whisper_bench_memcpy', 'whisper_context', 'whisper_ctx_init_openvino_encoder', 'whisper_decode', 'whisper_encode', 'whisper_free', 'whisper_full', 'whisper_full_default_params', 'whisper_full_get_segment_t0', 'whisper_full_get_segment_t1', 'whisper_full_get_segment_text', 'whisper_full_get_token_data', 'whisper_full_get_token_id', 'whisper_full_get_token_p', 'whisper_full_get_token_text', 'whisper_full_lang_id', 'whisper_full_n_segments', 'whisper_full_n_tokens', 'whisper_full_parallel', 'whisper_full_params', 'whisper_get_logits', 'whisper_init', 'whisper_init_from_buffer', 'whisper_init_from_file', 'whisper_is_multilingual', 'whisper_lang_auto_detect', 'whisper_lang_id', 'whisper_lang_max_id', 'whisper_lang_str', 'whisper_model_loader', 'whisper_n_audio_ctx', 'whisper_n_len', 'whisper_n_text_ctx', 'whisper_n_vocab', 'whisper_pcm_to_mel', 'whisper_print_system_info', 'whisper_print_timings', 'whisper_reset_timings', 'whisper_sampling_strategy', 'whisper_set_mel', 'whisper_token', 'whisper_token_beg', 'whisper_token_data', 'whisper_token_eot', 'whisper_token_lang', 'whisper_token_not', 'whisper_token_prev', 'whisper_token_solm', 'whisper_token_sot', 'whisper_token_to_bytes', 'whisper_token_to_str', 'whisper_token_transcribe', 'whisper_token_translate', 'whisper_tokenize', 'whisper_vad_context_params', 'whisper_vad_default_context_params', 'whisper_vad_default_params', 'whisper_vad_detect_speech', 'whisper_vad_free', 'whisper_vad_free_segments', 'whisper_vad_init_from_file_with_params', 'whisper_vad_n_probs', 'whisper_vad_params', 'whisper_vad_probs', 'whisper_vad_segments', 'whisper_vad_segments_from_probs', 'whisper_vad_segments_from_samples', 'whisper_vad_segments_get_segment_t0', 'whisper_vad_segments_get_segment_t1', 'whisper_vad_segments_n_segments']
class __whisper_full_params__internal:
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
class whisper_context:
    pass
class whisper_full_params(__whisper_full_params__internal):
    beam_search: dict
    encoder_begin_callback_user_data: types.CapsuleType
    greedy: dict
    initial_prompt: str
    language: str
    logits_filter_callback_user_data: types.CapsuleType
    new_segment_callback_user_data: types.CapsuleType
    no_context: bool
    print_progress: bool
    print_realtime: bool
    print_special: bool
    print_timestamps: bool
    progress_callback: typing.Callable
    single_segment: bool
    split_on_word: bool
    strategy: whisper_sampling_strategy
    suppress_blank: bool
    suppress_regex: str
    token_timestamps: bool
    translate: bool
    vad: bool
    vad_model_path: str
    vad_params: whisper_vad_params
    def __init__(self) -> None:
        ...
    @property
    def audio_ctx(self) -> int:
        ...
    @audio_ctx.setter
    def audio_ctx(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def duration_ms(self) -> int:
        ...
    @duration_ms.setter
    def duration_ms(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def entropy_thold(self) -> float:
        ...
    @entropy_thold.setter
    def entropy_thold(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def length_penalty(self) -> float:
        ...
    @length_penalty.setter
    def length_penalty(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def logprob_thold(self) -> float:
        ...
    @logprob_thold.setter
    def logprob_thold(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def max_initial_ts(self) -> float:
        ...
    @max_initial_ts.setter
    def max_initial_ts(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def max_len(self) -> int:
        ...
    @max_len.setter
    def max_len(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def max_tokens(self) -> int:
        ...
    @max_tokens.setter
    def max_tokens(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def n_max_text_ctx(self) -> int:
        ...
    @n_max_text_ctx.setter
    def n_max_text_ctx(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def n_threads(self) -> int:
        ...
    @n_threads.setter
    def n_threads(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def no_speech_thold(self) -> float:
        ...
    @no_speech_thold.setter
    def no_speech_thold(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def offset_ms(self) -> int:
        ...
    @offset_ms.setter
    def offset_ms(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def prompt_n_tokens(self) -> int:
        ...
    @prompt_n_tokens.setter
    def prompt_n_tokens(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def prompt_tokens(self) -> int:
        ...
    @prompt_tokens.setter
    def prompt_tokens(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def temperature(self) -> float:
        ...
    @temperature.setter
    def temperature(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def temperature_inc(self) -> float:
        ...
    @temperature_inc.setter
    def temperature_inc(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def thold_pt(self) -> float:
        ...
    @thold_pt.setter
    def thold_pt(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def thold_ptsum(self) -> float:
        ...
    @thold_ptsum.setter
    def thold_ptsum(self, arg0: typing.SupportsFloat) -> None:
        ...
class whisper_model_loader:
    def __init__(self) -> None:
        ...
class whisper_sampling_strategy:
    """
    Members:
    
      WHISPER_SAMPLING_GREEDY
    
      WHISPER_SAMPLING_BEAM_SEARCH
    """
    WHISPER_SAMPLING_BEAM_SEARCH: typing.ClassVar[whisper_sampling_strategy]  # value = <whisper_sampling_strategy.WHISPER_SAMPLING_BEAM_SEARCH: 1>
    WHISPER_SAMPLING_GREEDY: typing.ClassVar[whisper_sampling_strategy]  # value = <whisper_sampling_strategy.WHISPER_SAMPLING_GREEDY: 0>
    __members__: typing.ClassVar[dict[str, whisper_sampling_strategy]]  # value = {'WHISPER_SAMPLING_GREEDY': <whisper_sampling_strategy.WHISPER_SAMPLING_GREEDY: 0>, 'WHISPER_SAMPLING_BEAM_SEARCH': <whisper_sampling_strategy.WHISPER_SAMPLING_BEAM_SEARCH: 1>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class whisper_token:
    def __init__(self) -> None:
        ...
class whisper_token_data:
    def __init__(self) -> None:
        ...
    @property
    def id(self) -> int:
        ...
    @id.setter
    def id(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def p(self) -> float:
        ...
    @p.setter
    def p(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def plog(self) -> float:
        ...
    @plog.setter
    def plog(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def pt(self) -> float:
        ...
    @pt.setter
    def pt(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def ptsum(self) -> float:
        ...
    @ptsum.setter
    def ptsum(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def t0(self) -> int:
        ...
    @t0.setter
    def t0(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def t1(self) -> int:
        ...
    @t1.setter
    def t1(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def tid(self) -> int:
        ...
    @tid.setter
    def tid(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def vlen(self) -> float:
        ...
    @vlen.setter
    def vlen(self, arg0: typing.SupportsFloat) -> None:
        ...
class whisper_vad_context_params:
    use_gpu: bool
    def __init__(self) -> None:
        ...
    @property
    def gpu_device(self) -> int:
        ...
    @gpu_device.setter
    def gpu_device(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def n_threads(self) -> int:
        ...
    @n_threads.setter
    def n_threads(self, arg0: typing.SupportsInt) -> None:
        ...
class whisper_vad_params:
    def __init__(self) -> None:
        ...
    @property
    def max_speech_duration_s(self) -> float:
        ...
    @max_speech_duration_s.setter
    def max_speech_duration_s(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def min_silence_duration_ms(self) -> int:
        ...
    @min_silence_duration_ms.setter
    def min_silence_duration_ms(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def min_speech_duration_ms(self) -> int:
        ...
    @min_speech_duration_ms.setter
    def min_speech_duration_ms(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def samples_overlap(self) -> float:
        ...
    @samples_overlap.setter
    def samples_overlap(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def speech_pad_ms(self) -> int:
        ...
    @speech_pad_ms.setter
    def speech_pad_ms(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def threshold(self) -> float:
        ...
    @threshold.setter
    def threshold(self, arg0: typing.SupportsFloat) -> None:
        ...
class whisper_vad_segments:
    pass
def assign_encoder_begin_callback(params: __whisper_full_params__internal, callback: typing.Callable) -> None:
    """
    Assigns an encoder_begin_callback, takes <whisper_full_params> instance and a callable function with the same parameters which are defined in the interface
    """
def assign_logits_filter_callback(params: __whisper_full_params__internal, callback: typing.Callable) -> None:
    """
    Assigns a logits_filter_callback, takes <whisper_full_params> instance and a callable function with the same parameters which are defined in the interface
    """
def assign_new_segment_callback(params: __whisper_full_params__internal, callback: typing.Callable) -> None:
    """
    Assigns a new_segment_callback, takes <whisper_full_params> instance and a callable function with the same parameters which are defined in the interface
    """
def whisper_bench_ggml_mul_mat(arg0: typing.SupportsInt) -> int:
    """
    Temporary helpers needed for exposing ggml interface
    """
def whisper_bench_memcpy(arg0: typing.SupportsInt) -> int:
    """
    Temporary helpers needed for exposing ggml interface
    """
def whisper_ctx_init_openvino_encoder(arg0: whisper_context, arg1: str, arg2: str, arg3: str) -> int:
    """
    Given a context, enable use of OpenVINO for encode inference.
    """
def whisper_decode(arg0: whisper_context, arg1: typing.SupportsInt, arg2: typing.SupportsInt, arg3: typing.SupportsInt, arg4: typing.SupportsInt) -> int:
    """
    Run the Whisper decoder to obtain the logits and probabilities for the next token.
    Make sure to call whisper_encode() first.
    tokens + n_tokens is the provided context for the decoder.
    n_past is the number of tokens to use from previous decoder calls.
    Returns 0 on success
    TODO: add support for multiple decoders
    """
def whisper_encode(arg0: whisper_context, arg1: typing.SupportsInt, arg2: typing.SupportsInt) -> int:
    """
    Run the Whisper encoder on the log mel spectrogram stored inside the provided whisper context.
    Make sure to call whisper_pcm_to_mel() or whisper_set_mel() first.
    offset can be used to specify the offset of the first frame in the spectrogram.
    Returns 0 on success
    """
def whisper_free(arg0: whisper_context) -> None:
    """
    Frees all memory allocated by the model.
    """
def whisper_full(arg0: whisper_context, arg1: __whisper_full_params__internal, arg2: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], arg3: typing.SupportsInt) -> int:
    """
    Run the entire model: PCM -> log mel spectrogram -> encoder -> decoder -> text
    Uses the specified decoding strategy to obtain the text.
    """
def whisper_full_default_params(arg0: whisper_sampling_strategy) -> whisper_full_params:
    ...
def whisper_full_get_segment_t0(arg0: whisper_context, arg1: typing.SupportsInt) -> int:
    """
    Get the start time of the specified segment
    """
def whisper_full_get_segment_t1(arg0: whisper_context, arg1: typing.SupportsInt) -> int:
    """
    Get the end time of the specified segment
    """
def whisper_full_get_segment_text(arg0: whisper_context, arg1: typing.SupportsInt) -> bytes:
    """
    Get the text of the specified segment
    """
def whisper_full_get_token_data(arg0: whisper_context, arg1: typing.SupportsInt, arg2: typing.SupportsInt) -> whisper_token_data:
    """
    Get token data for the specified token in the specified segment.
    This contains probabilities, timestamps, etc.
    """
def whisper_full_get_token_id(arg0: whisper_context, arg1: typing.SupportsInt, arg2: typing.SupportsInt) -> int:
    """
    Get the token text of the specified token in the specified segment.
    """
def whisper_full_get_token_p(arg0: whisper_context, arg1: typing.SupportsInt, arg2: typing.SupportsInt) -> float:
    """
    Get the probability of the specified token in the specified segment.
    """
def whisper_full_get_token_text(arg0: whisper_context, arg1: typing.SupportsInt, arg2: typing.SupportsInt) -> str:
    """
    Get the token text of the specified token in the specified segment.
    """
def whisper_full_lang_id(arg0: whisper_context) -> int:
    """
    Language id associated with the current context
    """
def whisper_full_n_segments(arg0: whisper_context) -> int:
    """
    Number of generated text segments.
    A segment can be a few words, a sentence, or even a paragraph.
    """
def whisper_full_n_tokens(arg0: whisper_context, arg1: typing.SupportsInt) -> int:
    """
    Get number of tokens in the specified segment.
    """
def whisper_full_parallel(arg0: whisper_context, arg1: __whisper_full_params__internal, arg2: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], arg3: typing.SupportsInt, arg4: typing.SupportsInt) -> int:
    """
    Split the input audio in chunks and process each chunk separately using whisper_full()
    It seems this approach can offer some speedup in some cases.
    However, the transcription accuracy can be worse at the beginning and end of each chunk.
    """
def whisper_get_logits(arg0: whisper_context) -> float:
    """
    Token logits obtained from the last call to whisper_decode()
    The logits for the last token are stored in the last row
    Rows: n_tokens
    Cols: n_vocab
    """
def whisper_init(arg0: whisper_model_loader) -> whisper_context:
    """
    Various functions for loading a ggml whisper model.
    Allocate (almost) all memory needed for the model.
    Return NULL on failure
    """
def whisper_init_from_buffer(arg0: types.CapsuleType, arg1: typing.SupportsInt) -> whisper_context:
    """
    Various functions for loading a ggml whisper model.
    Allocate (almost) all memory needed for the model.
    Return NULL on failure
    """
def whisper_init_from_file(arg0: str) -> whisper_context:
    """
    Various functions for loading a ggml whisper model.
    Allocate (almost) all memory needed for the model.
    Return NULL on failure
    """
def whisper_is_multilingual(arg0: whisper_context) -> int:
    """
    whisper_is_multilingual
    """
def whisper_lang_auto_detect(arg0: whisper_context, arg1: typing.SupportsInt, arg2: typing.SupportsInt, arg3: typing.Annotated[numpy.typing.ArrayLike, numpy.float32]) -> int:
    """
    Use mel data at offset_ms to try and auto-detect the spoken language
    Make sure to call whisper_pcm_to_mel() or whisper_set_mel() first
    Returns the top language id or negative on failure
    If not null, fills the lang_probs array with the probabilities of all languages
    The array must be whispe_lang_max_id() + 1 in size
    ref: https://github.com/openai/whisper/blob/main/whisper/decoding.py#L18-L69
    """
def whisper_lang_id(arg0: str) -> int:
    """
    Return the id of the specified language, returns -1 if not found
    Examples:
    "de" -> 2
    "german" -> 2
    """
def whisper_lang_max_id() -> int:
    """
    Largest language id (i.e. number of available languages - 1)
    """
def whisper_lang_str(arg0: typing.SupportsInt) -> str:
    """
    Return the short string of the specified language id (e.g. 2 -> "de"), returns nullptr if not found
    """
def whisper_n_audio_ctx(arg0: whisper_context) -> int:
    """
    whisper_n_audio_ctx
    """
def whisper_n_len(arg0: whisper_context) -> int:
    """
    whisper_n_len
    """
def whisper_n_text_ctx(arg0: whisper_context) -> int:
    """
    whisper_n_text_ctx
    """
def whisper_n_vocab(arg0: whisper_context) -> int:
    """
    wrapper_whisper_n_vocab
    """
def whisper_pcm_to_mel(arg0: whisper_context, arg1: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], arg2: typing.SupportsInt, arg3: typing.SupportsInt) -> int:
    """
    Convert RAW PCM audio to log mel spectrogram.
    The resulting spectrogram is stored inside the provided whisper context.
    Returns 0 on success
    """
def whisper_print_system_info() -> str:
    ...
def whisper_print_timings(arg0: whisper_context) -> None:
    ...
def whisper_reset_timings(arg0: whisper_context) -> None:
    ...
def whisper_set_mel(arg0: whisper_context, arg1: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], arg2: typing.SupportsInt, arg3: typing.SupportsInt) -> int:
    """
     This can be used to set a custom log mel spectrogram inside the provided whisper context.
    Use this instead of whisper_pcm_to_mel() if you want to provide your own log mel spectrogram.
    n_mel must be 80
    Returns 0 on success
    """
def whisper_token_beg(arg0: whisper_context) -> int:
    ...
def whisper_token_eot(arg0: whisper_context) -> int:
    """
    whisper_token_eot
    """
def whisper_token_lang(arg0: whisper_context, arg1: typing.SupportsInt) -> int:
    ...
def whisper_token_not(arg0: whisper_context) -> int:
    ...
def whisper_token_prev(arg0: whisper_context) -> int:
    ...
def whisper_token_solm(arg0: whisper_context) -> int:
    ...
def whisper_token_sot(arg0: whisper_context) -> int:
    """
    whisper_token_sot
    """
def whisper_token_to_bytes(arg0: whisper_context, arg1: typing.SupportsInt) -> bytes:
    """
    whisper_token_to_bytes
    """
def whisper_token_to_str(arg0: whisper_context, arg1: typing.SupportsInt) -> str:
    """
    whisper_token_to_str
    """
def whisper_token_transcribe(arg0: whisper_context) -> int:
    ...
def whisper_token_translate(arg0: whisper_context) -> int:
    ...
def whisper_tokenize(arg0: whisper_context, arg1: str, arg2: typing.SupportsInt, arg3: typing.SupportsInt) -> int:
    """
    Convert the provided text into tokens.
    The tokens pointer must be large enough to hold the resulting tokens.
    Returns the number of tokens on success, no more than n_max_tokens
    Returns -1 on failure
    TODO: not sure if correct
    """
def whisper_vad_default_context_params() -> whisper_vad_context_params:
    ...
def whisper_vad_default_params() -> whisper_vad_params:
    ...
def whisper_vad_detect_speech(arg0: whisper_vad_context_wrapper, arg1: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], arg2: typing.SupportsInt) -> bool:
    ...
def whisper_vad_free(arg0: whisper_vad_context_wrapper) -> None:
    ...
def whisper_vad_free_segments(arg0: whisper_vad_segments) -> None:
    ...
def whisper_vad_init_from_file_with_params(arg0: str, arg1: whisper_vad_context_params) -> whisper_vad_context_wrapper:
    ...
def whisper_vad_n_probs(arg0: whisper_vad_context_wrapper) -> int:
    ...
def whisper_vad_probs(arg0: whisper_vad_context_wrapper) -> numpy.typing.NDArray[numpy.float32]:
    ...
def whisper_vad_segments_from_probs(arg0: whisper_vad_context_wrapper, arg1: whisper_vad_params) -> whisper_vad_segments:
    ...
def whisper_vad_segments_from_samples(arg0: whisper_vad_context_wrapper, arg1: whisper_vad_params, arg2: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], arg3: typing.SupportsInt) -> whisper_vad_segments:
    ...
def whisper_vad_segments_get_segment_t0(arg0: whisper_vad_segments, arg1: typing.SupportsInt) -> float:
    ...
def whisper_vad_segments_get_segment_t1(arg0: whisper_vad_segments, arg1: typing.SupportsInt) -> float:
    ...
def whisper_vad_segments_n_segments(arg0: whisper_vad_segments) -> int:
    ...
WHISPER_CHUNK_SIZE: int = 30
WHISPER_HOP_LENGTH: int = 160
WHISPER_N_FFT: int = 400
WHISPER_SAMPLE_RATE: int = 16000
WHISPER_SAMPLING_BEAM_SEARCH: whisper_sampling_strategy  # value = <whisper_sampling_strategy.WHISPER_SAMPLING_BEAM_SEARCH: 1>
WHISPER_SAMPLING_GREEDY: whisper_sampling_strategy  # value = <whisper_sampling_strategy.WHISPER_SAMPLING_GREEDY: 0>
__version__: str = 'dev'
