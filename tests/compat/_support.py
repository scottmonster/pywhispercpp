#!/usr/bin/env python
# -*- coding: utf-8 -*-

import atexit
import subprocess
import sys
from pathlib import Path
from unittest import TestCase


REPO_ROOT = Path(__file__).resolve().parents[2]
WHISPER_CPP_DIR = REPO_ROOT / 'whisper.cpp'
COMPAT_RUNNER = REPO_ROOT / 'tests' / 'run-compat.py'
FAILURE_SUMMARIES = []


def _print_failure_summary_at_exit():
    if not FAILURE_SUMMARIES:
        return

    print('\nFailure summary:', file=sys.stderr)
    for test_name, details in FAILURE_SUMMARIES:
        print(f'- {test_name}', file=sys.stderr)
        for detail in details:
            print(f'  - {detail}', file=sys.stderr)


atexit.register(_print_failure_summary_at_exit)


def cpu_context_params():
    return {'use_gpu': False, 'flash_attn': False}


def run_isolated_case(module_globals: dict, case_qualname: str):
    class_name, method_name = case_qualname.split('.', 1)
    case_class = module_globals[class_name]
    case_instance = case_class(methodName='runTest')
    getattr(case_instance, method_name)()


def maybe_run_isolated_case(module_globals: dict):
    if len(sys.argv) == 3 and sys.argv[1] == '--run-case':
        run_isolated_case(module_globals, sys.argv[2])
        raise SystemExit(0)


class FailureSummaryTestCase(TestCase):
    failure_summaries = []
    module_name = ''
    repo_root = REPO_ROOT

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.failure_summaries = []

    def run(self, result=None):
        if result is None:
            result = self.defaultTestResult()

        failures_before = len(result.failures)
        errors_before = len(result.errors)
        skipped_before = len(getattr(result, 'skipped', []))
        super().run(result)

        new_details = []

        for failed_test, traceback in result.failures[failures_before:]:
            if failed_test is self:
                new_details.append(f'failure:\n{traceback}')

        for errored_test, traceback in result.errors[errors_before:]:
            if errored_test is self:
                new_details.append(f'error:\n{traceback}')

        for skipped_test, reason in getattr(result, 'skipped', [])[skipped_before:]:
            skipped_id = skipped_test.id() if hasattr(skipped_test, 'id') else ''
            if skipped_test is self or skipped_id.startswith(self.id()):
                new_details.append(f'skipped:\n{reason}')

        if new_details:
            self.__class__.failure_summaries.append((self.id(), new_details))

        return result

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        if not cls.failure_summaries:
            return

        FAILURE_SUMMARIES.extend(cls.failure_summaries)

    def _run_case(self, helper_name: str, expected_returncode: int = 0):
        contract_name = self.module_name.rsplit('.', 1)[-1]
        result = subprocess.run(
            [
                sys.executable,
                str(COMPAT_RUNNER),
                '-r',
                contract_name,
                '--run-case',
                f'{self.__class__.__name__}.{helper_name}',
            ],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.returncode,
            expected_returncode,
            msg=f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}",
        )
