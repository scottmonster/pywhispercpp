from __future__ import annotations

import argparse
import ast
import re
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

from pybind11_stubgen import main as pybind11_stubgen_main
from pygccxml import declarations
from pygccxml import parser
from pygccxml import utils


def add(mapping: dict[str, set[str]], key: str, value: str) -> None:
    mapping.setdefault(key, set()).add(value)


def is_property_accessor(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Name) and decorator.id == "property":
            return True
        if (
            isinstance(decorator, ast.Attribute)
            and isinstance(decorator.value, ast.Name)
            and decorator.value.id == node.name
            and decorator.attr == "setter"
        ):
            return True
    return False


def parse_args() -> argparse.Namespace:
    parser_obj = argparse.ArgumentParser(description="Generate a binding coverage report.")
    parser_obj.add_argument("--report-path", type=Path, required=True)
    parser_obj.add_argument("--header-path", type=Path, required=True)
    parser_obj.add_argument(
        "-s",
        "--stubs",
        nargs="?",
        const=Path("stubs"),
        type=Path,
        dest="stub_dir",
        help="Optionally save generated stubs. Defaults to ./stubs when no path is provided.",
    )
    parser_obj.add_argument("--project-root", type=Path, required=True)
    parser_obj.add_argument("--module-name", default="_pywhispercpp")
    return parser_obj.parse_args()


def generate_stubs(stub_dir: Path, module_name: str) -> None:
    stub_dir.mkdir(parents=True, exist_ok=True)
    pybind11_stubgen_main(
        [
            "--numpy-array-use-type-var",
            "--ignore-unresolved-names",
            r"^(types\.CapsuleType|whisper_vad_context_wrapper)$",
            module_name,
            "-o",
            str(stub_dir),
        ]
    )


@contextmanager
def prepared_stub_dir(stub_dir: Path | None, module_name: str) -> Iterator[Path]:
    if stub_dir is not None:
        generate_stubs(stub_dir, module_name)
        yield stub_dir
        return

    with tempfile.TemporaryDirectory(prefix="pywhispercpp-stubs-") as temp_dir:
        temp_path = Path(temp_dir)
        generate_stubs(temp_path, module_name)
        yield temp_path


def collect_header_symbols(
    header_path: Path,
) -> tuple[dict[str, set[str]], set[str]]:
    generator_path, generator_name = utils.find_xml_generator()
    root = header_path.parents[2]

    config = parser.xml_generator_configuration_t(
        xml_generator_path=generator_path,
        xml_generator=generator_name,
        include_paths=[
            str(root / "whisper.cpp/include"),
            str(root / "whisper.cpp/ggml/include"),
        ],
        cflags="-std=c++11",
    )

    decls = parser.parse([str(header_path)], config)
    global_ns = declarations.get_global_namespace(decls)

    categories: dict[str, set[str]] = {
        "functions": set(),
        "types": set(),
        "enum_types": set(),
        "enum_values": set(),
        "struct_fields": set(),
    }

    for fn in global_ns.free_functions(recursive=True, allow_empty=True):
        if fn.name.startswith("whisper_"):
            add(categories, "functions", fn.name)

    for cls in global_ns.classes(recursive=True, allow_empty=True):
        if cls.name.startswith("whisper_"):
            add(categories, "types", cls.name)
            for var in cls.variables(allow_empty=True):
                add(categories, "struct_fields", f"{cls.name}.{var.name}")

    for enum in global_ns.enumerations(recursive=True, allow_empty=True):
        if enum.name and enum.name.startswith("whisper_"):
            add(categories, "enum_types", enum.name)
            for value_name, _ in enum.values:
                add(categories, "enum_values", f"{enum.name}.{value_name}")

    combined = set().union(*categories.values())
    return categories, combined


def collect_stub_symbols(
    stub_dir: Path, module_name: str
) -> tuple[dict[str, set[str]], set[str]]:
    categories: dict[str, set[str]] = {
        "functions": set(),
        "types": set(),
        "enum_types": set(),
        "enum_values": set(),
        "struct_fields": set(),
    }

    stub_files = sorted(stub_dir.rglob(f"{module_name}*.pyi"))
    if not stub_files:
        raise SystemExit(f"No {module_name} stub file found under {stub_dir}")

    for stub_path in stub_files:
        tree = ast.parse(stub_path.read_text(encoding="utf-8"))

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                add(categories, "functions", node.name)
                continue

            if not isinstance(node, ast.ClassDef):
                continue

            class_name = node.name
            add(categories, "types", class_name)

            is_enum_like = any(
                isinstance(child, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "__members__"
                    for target in child.targets
                )
                for child in node.body
            )
            if is_enum_like:
                add(categories, "enum_types", class_name)

            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if is_property_accessor(child):
                        add(categories, "struct_fields", f"{class_name}.{child.name}")
                    continue

                if isinstance(child, ast.Assign):
                    for target in child.targets:
                        if not isinstance(target, ast.Name):
                            continue
                        name = target.id
                        if name.startswith("__"):
                            continue
                        qualified = f"{class_name}.{name}"
                        if is_enum_like and name.isupper():
                            add(categories, "enum_values", qualified)
                        else:
                            add(categories, "struct_fields", qualified)

                elif isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
                    name = child.target.id
                    if name.startswith("__"):
                        continue
                    qualified = f"{class_name}.{name}"
                    if is_enum_like and name.isupper():
                        add(categories, "enum_values", qualified)
                    else:
                        add(categories, "struct_fields", qualified)

    combined = set().union(*categories.values())
    return categories, combined


def collect_macro_constants(header_path: Path) -> set[str]:
    text = header_path.read_text(encoding="utf-8")
    constants: set[str] = set()
    pattern = re.compile(r"^#define\s+(WHISPER_[A-Z0-9_]+)\s+(.+)$", re.M)
    for name, value in pattern.findall(text):
        if name in {"WHISPER_H", "WHISPER_API", "WHISPER_DEPRECATED"}:
            continue
        if "(" in name or value.strip().startswith("__"):
            continue
        constants.add(name)
    return constants


def render_section(lines: list[str], title: str, items: list[str]) -> None:
    render_heading(lines, f"{title} ({len(items)})", level=3)
    for item in items:
        lines.append(f"- {item}")
    if not items:
        lines.append("- None")
    lines.append("")


def render_heading(lines: list[str], title: str, level: int = 2) -> None:
    lines.append(f"{'#' * level} {title}")
    lines.append("")


def render_counts_table(
    lines: list[str],
    header_categories: dict[str, set[str]],
    stub_categories: dict[str, set[str]],
) -> None:
    lines.append("| Category | Header | Stub |")
    lines.append("| --- | ---: | ---: |")
    lines.append(f"| Functions | {len(header_categories['functions'])} | {len(stub_categories['functions'])} |")
    lines.append(f"| Types | {len(header_categories['types'])} | {len(stub_categories['types'])} |")
    lines.append(f"| Enum types | {len(header_categories['enum_types'])} | {len(stub_categories['enum_types'])} |")
    lines.append(f"| Enum values | {len(header_categories['enum_values'])} | {len(stub_categories['enum_values'])} |")
    lines.append(f"| Struct fields | {len(header_categories['struct_fields'])} | {len(stub_categories['struct_fields'])} |")
    lines.append("")


def collect_pywhispercpp_version(project_root: Path) -> str:
    version_file = project_root / "_version.py"
    text = version_file.read_text(encoding="utf-8")
    match = re.search(r"^__version__\s*=\s*version\s*=\s*['\"]([^'\"]+)['\"]", text, re.M)
    if not match:
        raise SystemExit(f"Could not determine pywhispercpp version from {version_file}")
    return match.group(1)


def collect_whispercpp_version(project_root: Path) -> str:
    cmake_file = project_root / "whisper.cpp" / "CMakeLists.txt"
    text = cmake_file.read_text(encoding="utf-8")
    match = re.search(r'^project\("whisper\.cpp"\s+VERSION\s+([^\)\s]+)\)', text, re.M)
    if not match:
        raise SystemExit(f"Could not determine whisper.cpp version from {cmake_file}")
    return match.group(1)


def main() -> None:
    args = parse_args()
    stub_output_dir = (
        None
        if args.stub_dir is None
        else args.stub_dir
        if args.stub_dir.is_absolute()
        else args.report_path.parent / args.stub_dir
    )

    header_categories, header_all = collect_header_symbols(args.header_path)
    header_macros = collect_macro_constants(args.header_path)
    pywhispercpp_version = collect_pywhispercpp_version(args.project_root)
    whispercpp_version = collect_whispercpp_version(args.project_root)

    with prepared_stub_dir(stub_output_dir, args.module_name) as stub_dir:
        stub_categories, stub_all = collect_stub_symbols(stub_dir, args.module_name)

        included_overall = sorted(header_all & stub_all)
        missing_overall = sorted(header_all - stub_all)
        extra_overall = sorted(stub_all - header_all)

        lines: list[str] = []
        lines.append("# Binding Report")
        lines.append("")
        render_heading(lines, "1. Summary")
        lines.append(f"- Generated: `{datetime.now().astimezone().isoformat(timespec='seconds')}`")
        lines.append(f"- pywhispercpp: `{pywhispercpp_version}`")
        lines.append(f"- whisper.cpp: `{whispercpp_version}`")
        lines.append(f"- Header: `{args.header_path}`")
        lines.append(f"- Stubs: `{stub_dir if stub_output_dir is not None else '(temporary, not saved)'}`")
        lines.append("")

        render_heading(lines, "2. Counts")
        render_counts_table(lines, header_categories, stub_categories)

        render_heading(lines, "3. Included")
        for key, title in (
            ("functions", "Included functions"),
            ("types", "Included types"),
            ("enum_types", "Included enum types"),
            ("enum_values", "Included enum values"),
            ("struct_fields", "Included struct fields"),
        ):
            render_section(lines, title, sorted(header_categories[key] & stub_categories[key]))

        render_section(lines, "Included overall", included_overall)

        render_heading(lines, "4. Missing")

        for key, title in (
            ("functions", "Missing functions"),
            ("types", "Missing types"),
            ("enum_types", "Missing enum types"),
            ("enum_values", "Missing enum values"),
            ("struct_fields", "Missing struct fields"),
        ):
            render_section(lines, title, sorted(header_categories[key] - stub_categories[key]))

        render_section(lines, "Missing macros", sorted(header_macros))
        render_section(lines, "Extra/python-only symbols", extra_overall)
        render_section(lines, "Missing overall", missing_overall)

        args.report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    print(f"Header symbols: {len(header_all)}")
    print(f"Stub symbols:   {len(stub_all)}")
    print(f"Missing overall: {len(missing_overall)}")
    print(f"Extra/python-only symbols: {len(extra_overall)}")
    print(f"Missing macros: {len(header_macros)}")
    if stub_output_dir is not None:
        print(f"Saved stubs to: {stub_output_dir}")
    print(f"Report written to: {args.report_path}")


if __name__ == "__main__":
    main()