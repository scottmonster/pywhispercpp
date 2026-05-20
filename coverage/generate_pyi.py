from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONSTANTS_PATH = PROJECT_ROOT / "pywhispercpp" / "constants.py"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "pywhispercpp" / "model.py"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "pywhispercpp" / "model.pyi"


@dataclass(frozen=True)
class ParamSpec:
	name: str
	type_name: str
	default: Any


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Generate pywhispercpp/model.pyi from model.py and constants.py."
	)
	parser.add_argument(
		"--constants-path",
		type=Path,
		default=DEFAULT_CONSTANTS_PATH,
		help="Path to pywhispercpp/constants.py",
	)
	parser.add_argument(
		"--model-path",
		type=Path,
		default=DEFAULT_MODEL_PATH,
		help="Path to pywhispercpp/model.py",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=DEFAULT_OUTPUT_PATH,
		help="Where to write the generated model.pyi file",
	)
	parser.add_argument(
		"--check",
		action="store_true",
		help="Exit with a non-zero status if the generated output differs from the existing file.",
	)
	return parser.parse_args()


def load_module_ast(path: Path) -> ast.Module:
	return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def find_assignment(module: ast.Module, name: str) -> ast.Assign:
	for node in module.body:
		if not isinstance(node, ast.Assign):
			continue
		for target in node.targets:
			if isinstance(target, ast.Name) and target.id == name:
				return node
	raise SystemExit(f"Could not find assignment to {name!r}")


def dict_items(node: ast.Dict) -> Iterable[tuple[ast.expr, ast.expr]]:
	items: list[tuple[ast.expr, ast.expr]] = []
	for key, value in zip(node.keys, node.values):
		if key is None:
			raise SystemExit("Unexpected dictionary unpacking while parsing source")
		items.append((key, value))
	return items


def get_string_literal(node: ast.expr | None) -> str:
	if isinstance(node, ast.Constant) and isinstance(node.value, str):
		return node.value
	raise SystemExit("Expected a string literal while parsing PARAMS_SCHEMA")


def get_type_name(node: ast.expr) -> str:
	if isinstance(node, ast.Name):
		return node.id
	if isinstance(node, ast.Attribute):
		return node.attr
	raise SystemExit("Unsupported type declaration in PARAMS_SCHEMA")


def collect_param_specs(constants_path: Path) -> list[ParamSpec]:
	constants_ast = load_module_ast(constants_path)
	schema_assign = find_assignment(constants_ast, "PARAMS_SCHEMA")
	if not isinstance(schema_assign.value, ast.Dict):
		raise SystemExit("PARAMS_SCHEMA must be a dictionary literal")

	specs: list[ParamSpec] = []
	for key_node, value_node in dict_items(schema_assign.value):
		name = get_string_literal(key_node)
		if not isinstance(value_node, ast.Dict):
			raise SystemExit(f"PARAMS_SCHEMA[{name!r}] must be a dictionary literal")

		type_name: str | None = None
		default: Any = ...
		for field_key_node, field_value_node in dict_items(value_node):
			field_name = get_string_literal(field_key_node)
			if field_name == "type":
				type_name = get_type_name(field_value_node)
			elif field_name == "default":
				default = ast.literal_eval(field_value_node)

		if type_name is None:
			raise SystemExit(f"PARAMS_SCHEMA[{name!r}] is missing a 'type' entry")

		specs.append(ParamSpec(name=name, type_name=type_name, default=default))

	return specs


def collect_model_defaults(model_path: Path) -> dict[str, Any]:
	model_ast = load_module_ast(model_path)
	defaults: dict[str, Any] = {}
	for node in model_ast.body:
		if not isinstance(node, ast.ClassDef) or node.name != "Model":
			continue
		for child in node.body:
			if not isinstance(child, ast.FunctionDef):
				continue
			if child.name not in {"__init__", "transcribe"}:
				continue

			positional_args = child.args.args[1:]
			positional_defaults = child.args.defaults
			if positional_defaults:
				default_offset = len(positional_args) - len(positional_defaults)
				for index, default_node in enumerate(positional_defaults):
					arg_name = positional_args[default_offset + index].arg
					defaults[arg_name] = ast.literal_eval(default_node)

	return defaults


def param_annotation(spec: ParamSpec) -> str:
	if spec.name == "greedy":
		return "GreedyParams"
	if spec.name == "beam_search":
		return "BeamSearchParams"
	if spec.type_name == "Tuple":
		return "Optional[Tuple[Any, ...]]" if spec.default is None else "Tuple[Any, ...]"
	if spec.type_name == "dict":
		return "Dict[str, Any]"
	if spec.type_name == "str" and spec.default is None:
		return "Optional[str]"
	if spec.type_name == "int" and spec.default is None:
		return "Optional[int]"
	if spec.type_name == "float" and spec.default is None:
		return "Optional[float]"
	if spec.type_name == "bool" and spec.default is None:
		return "Optional[bool]"
	return spec.type_name


def render_default(value: Any) -> str:
	if isinstance(value, str):
		return repr(value)
	return repr(value)


def render_kwargs(specs: Iterable[ParamSpec], indent: str = "        ") -> list[str]:
	lines: list[str] = []
	for spec in specs:
		lines.append(
			f"{indent}{spec.name}: {param_annotation(spec)} = {render_default(spec.default)},"
		)
	return lines


def generate_stub(constants_path: Path, model_path: Path) -> str:
	param_specs = collect_param_specs(constants_path)
	model_defaults = collect_model_defaults(model_path)

	init_param_specs = [spec for spec in param_specs if spec.name != "extract_probability"]
	transcribe_param_specs = [
		replace(spec, default=model_defaults.get("extract_probability", False))
		if spec.name == "extract_probability"
		else spec
		for spec in param_specs
	]

	lines = [
		"from __future__ import annotations",
		"",
		"# Generated by coverage/generate_pyi.py. Do not edit by hand.",
		"",
		"from typing import Any, Callable, Dict, List, Optional, TextIO, Tuple, TypedDict, Union",
		"",
		"import numpy as np",
		"import numpy.typing as npt",
		"",
		"AudioArray = npt.NDArray[np.float32]",
		"AudioInput = Union[str, AudioArray]",
		"",
		"",
		"class GreedyParams(TypedDict):",
		"    best_of: int",
		"",
		"",
		"class BeamSearchParams(TypedDict):",
		"    beam_size: int",
		"    patience: float",
		"",
		"",
		"class Segment:",
		"    t0: int",
		"    t1: int",
		"    text: str",
		"    probability: float",
		"",
		"    def __init__(self, t0: int, t1: int, text: str, probability: float = np.nan)->None: ...",
		"    def __str__(self)->str: ...",
		"    def __repr__(self)->str: ...",
		"",
		"",
		"class Model:",
		"    _new_segment_callback: Optional[Callable[[Segment], None]]",
		"",
		"    def __init__(",
		"        self,",
		f"        model: str = {render_default(model_defaults['model'])},",
		f"        models_dir: Optional[str] = {render_default(model_defaults['models_dir'])},",
		f"        params_sampling_strategy: int = {render_default(model_defaults['params_sampling_strategy'])},",
		"        redirect_whispercpp_logs_to: Union[bool, TextIO, str, None] = False,",
		f"        use_openvino: bool = {render_default(model_defaults['use_openvino'])},",
		f"        openvino_model_path: Optional[str] = {render_default(model_defaults['openvino_model_path'])},",
		f"        openvino_device: str = {render_default(model_defaults['openvino_device'])},",
		f"        openvino_cache_dir: Optional[str] = {render_default(model_defaults['openvino_cache_dir'])},",
		"        *,",
		*render_kwargs(init_param_specs),
		"    )->None: ...",
		"",
		"    def transcribe(",
		"        self,",
		"        media: AudioInput,",
		f"        n_processors: Optional[int] = {render_default(model_defaults['n_processors'])},",
		f"        new_segment_callback: Optional[Callable[[Segment], None]] = {render_default(model_defaults['new_segment_callback'])},",
		"        *,",
		*render_kwargs(transcribe_param_specs),
		"    ) -> List[Segment]: ...",
		"",
		"    def get_params(self) -> Dict[str, Any]: ...",
		"    @staticmethod",
		"    def get_params_schema() -> Dict[str, Dict[str, Any]]: ...",
		"    @staticmethod",
		"    def lang_max_id() -> int: ...",
		"    def print_timings(self) -> None: ...",
		"    @staticmethod",
		"    def system_info() -> Any: ...",
		"    @staticmethod",
		"    def available_languages() -> List[str]: ...",
		"    @staticmethod",
		"    def _load_audio(media_file_path: str) -> AudioArray: ...",
		"    def auto_detect_language(",
		"        self,",
		"        media: AudioInput,",
		"        offset_ms: int = 0,",
		"        n_threads: int = 4,",
		"    ) -> Tuple[Tuple[str, np.float32], Dict[str, np.float32]]: ...",
		"    def __del__(self) -> None: ...",
		"",
	]
	return "\n".join(lines)


def main() -> None:
	args = parse_args()
	output_text = generate_stub(args.constants_path, args.model_path) + "\n"

	if args.check:
		existing = ""
		if args.output.exists():
			existing = args.output.read_text(encoding="utf-8")
		if existing != output_text:
			raise SystemExit(f"Stub is out of date: {args.output}")
		print(f"Stub is up to date: {args.output}")
		return

	args.output.parent.mkdir(parents=True, exist_ok=True)
	args.output.write_text(output_text, encoding="utf-8")
	print(f"Wrote {args.output}")


if __name__ == "__main__":
	main()













