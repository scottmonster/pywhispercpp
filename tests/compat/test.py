#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import importlib
import sys
import unittest
from pathlib import Path


COMPAT_DIR = Path(__file__).resolve().parent


def _parse_args():
	parser = argparse.ArgumentParser(add_help=False)
	parser.add_argument('-r', '--run', action='append', dest='run')
	args, remaining = parser.parse_known_args()
	sys.argv = [sys.argv[0], *remaining]
	return args


def _configured_tests():
	selected = sorted(path.stem for path in COMPAT_DIR.glob('v*.py'))
	if (COMPAT_DIR / 'unreleased.py').exists():
		selected.append('unreleased')
	args = _parse_args()
	if args.run:
		selected = args.run
	return selected


def _load_contract_modules():
	return [
		importlib.import_module(f'tests.compat.{name}')
		for name in _configured_tests()
	]


def _expose_test_cases(module):
	for name, value in vars(module).items():
		if isinstance(value, type) and issubclass(value, unittest.TestCase):
			globals()[name] = value


for contract_module in _load_contract_modules():
	_expose_test_cases(contract_module)


if __name__ == '__main__':
	unittest.main()




