from __future__ import annotations

import ast
import os
import re
from datetime import datetime
from pathlib import Path

from pygccxml import declarations
from pygccxml import parser
from pygccxml import utils


REPORT_PATH = Path(os.environ["REPORT_PATH"])
HEADER_PATH = Path(os.environ["HEADER_PATH"])
STUB_DIR = Path(os.environ["STUB_DIR"])
PROJECT_ROOT = HEADER_PATH.parents[2]


def add(mapping: dict[str, set[str]], key: str, value: str) -> None:
    mapping.setdefault(key, set()).add(value)


def collect_header_symbols(header_path: Path) -> tuple[dict[str, set[str]], set[str]]:
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


def collect_stub_symbols(stub_dir: Path) -> tuple[dict[str, set[str]], set[str]]:
    categories: dict[str, set[str]] = {
        "functions": set(),
        "types": set(),
        "enum_types": set(),
        "enum_values": set(),
        "struct_fields": set(),
    }

    stub_files = sorted(stub_dir.rglob("_pywhispercpp*.pyi"))
    if not stub_files:
        raise SystemExit(f"No _pywhispercpp stub file found under {stub_dir}")

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
    lines.append(f"{title}: {len(items)}")
    for item in items:
        lines.append(f"- {item}")
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
    header_categories, header_all = collect_header_symbols(HEADER_PATH)
    stub_categories, stub_all = collect_stub_symbols(STUB_DIR)
    header_macros = collect_macro_constants(HEADER_PATH)
    pywhispercpp_version = collect_pywhispercpp_version(PROJECT_ROOT)
    whispercpp_version = collect_whispercpp_version(PROJECT_ROOT)

    missing_overall = sorted(header_all - stub_all)
    extra_overall = sorted(stub_all - header_all)

    lines: list[str] = []
    lines.append(f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}")
    lines.append(f"pywhispercpp: {pywhispercpp_version}")
    lines.append(f"whisper.cpp: {whispercpp_version}")
    lines.append(f"Header: {HEADER_PATH}")
    lines.append(f"Stubs:  {STUB_DIR}")
    lines.append("")
    lines.append("Counts")
    lines.append(f"- header functions: {len(header_categories['functions'])}")
    lines.append(f"- header types: {len(header_categories['types'])}")
    lines.append(f"- header enum types: {len(header_categories['enum_types'])}")
    lines.append(f"- header enum values: {len(header_categories['enum_values'])}")
    lines.append(f"- header struct fields: {len(header_categories['struct_fields'])}")
    lines.append(f"- stub functions: {len(stub_categories['functions'])}")
    lines.append(f"- stub types: {len(stub_categories['types'])}")
    lines.append(f"- stub enum types: {len(stub_categories['enum_types'])}")
    lines.append(f"- stub enum values: {len(stub_categories['enum_values'])}")
    lines.append(f"- stub struct fields: {len(stub_categories['struct_fields'])}")
    lines.append("")

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

    REPORT_PATH.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    print(f"Header symbols: {len(header_all)}")
    print(f"Stub symbols:   {len(stub_all)}")
    print(f"Missing overall: {len(missing_overall)}")
    print(f"Extra/python-only symbols: {len(extra_overall)}")
    print(f"Missing macros: {len(header_macros)}")
    print(f"Report written to: {REPORT_PATH}")


if __name__ == "__main__":
    main()