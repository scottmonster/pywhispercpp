#!/usr/bin/env python
# -*- coding: utf-8 -*-

import importlib
import runpy
import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = REPO_ROOT / 'tests'
COMPAT_DIR = TESTS_DIR / 'compat'


def _parse_contract_name(argv: list[str]) -> str | None:
	for index, value in enumerate(argv):
		if value in ('-r', '--run') and index + 1 < len(argv):
			return argv[index + 1]
	return None


def _ensure_namespace_package(name: str, package_path: Path):
	module = sys.modules.get(name)
	if module is None:
		module = types.ModuleType(name)
		sys.modules[name] = module
	module.__path__ = [str(package_path)]
	return module


def main():
	sys.path.insert(0, str(REPO_ROOT))
	tests_module = _ensure_namespace_package('tests', TESTS_DIR)
	compat_module = _ensure_namespace_package('tests.compat', COMPAT_DIR)
	setattr(tests_module, 'compat', compat_module)

	if '--run-case' in sys.argv:
		contract_name = _parse_contract_name(sys.argv)
		if not contract_name:
			raise SystemExit('run-compat.py requires -r/--run when used with --run-case')
		run_case_index = sys.argv.index('--run-case')
		sys.argv = [sys.argv[0], *sys.argv[run_case_index:run_case_index + 2]]
		runpy.run_module(f'tests.compat.{contract_name}', run_name='__main__')
		return

	module = importlib.import_module('tests.compat.test')
	unittest.main(module=module)


if __name__ == '__main__':
	main()
