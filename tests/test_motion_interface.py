"""Tests for MotionInterface's new SDK-level output config args."""

import pytest
from omotion.MotionInterface import MotionInterface


def test_motion_interface_accepts_data_dir():
    motion = MotionInterface(demo_mode=True, data_dir="C:/tmp/scans")
    assert motion.data_dir == "C:/tmp/scans"


def test_motion_interface_accepts_scan_db_path():
    motion = MotionInterface(demo_mode=True, scan_db_path="C:/tmp/scans/scans.db")
    assert motion.scan_db_path == "C:/tmp/scans/scans.db"


def test_motion_interface_accepts_operator_id():
    motion = MotionInterface(demo_mode=True, operator_id="bloodflow-app")
    assert motion.operator_id == "bloodflow-app"


def test_motion_interface_defaults_when_args_omitted():
    motion = MotionInterface(demo_mode=True)
    assert motion.data_dir is None
    assert motion.scan_db_path is None
    assert motion.operator_id is None
