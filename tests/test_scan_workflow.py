"""Tests for ScanWorkflow and ScanRequest."""

import dataclasses
from omotion.ScanWorkflow import ScanRequest


def test_scan_request_carries_sinks_field():
    """ScanRequest should carry sinks, skip_default_storage, raw_save_max_duration_s fields."""
    field_names = {f.name for f in dataclasses.fields(ScanRequest)}
    assert "sinks" in field_names
    assert "skip_default_storage" in field_names
    assert "raw_save_max_duration_s" in field_names


def test_scan_request_drops_legacy_fields():
    """ScanRequest should not carry legacy callback/csv/operator fields."""
    field_names = {f.name for f in dataclasses.fields(ScanRequest)}
    assert "on_uncorrected_fn" not in field_names
    assert "on_corrected_batch_fn" not in field_names
    assert "on_dark_frame_fn" not in field_names
    assert "on_rolling_avg_fn" not in field_names
    assert "on_realtime_corrected_fn" not in field_names
    assert "on_raw_frame_fn" not in field_names
    assert "write_raw_csv" not in field_names
    assert "raw_csv_duration_sec" not in field_names
    assert "operator" not in field_names


def test_scan_request_sinks_default_empty_list():
    """ScanRequest should default to empty sinks list and allow construction without these fields."""
    req = ScanRequest(
        subject_id="x", duration_sec=60,
        left_camera_mask=0xFF, right_camera_mask=0xFF,
        disable_laser=False,
    )
    assert req.sinks == []
    assert req.skip_default_storage is False
    assert req.raw_save_max_duration_s is None


# ---------------------------------------------------------------------------
# Phase E tests — new pipeline-based start_scan
# ---------------------------------------------------------------------------

def _build_motion_with_data_dir(data_dir, scan_db_path=None):
    """Helper: construct a demo-mode MotionInterface with the given output paths."""
    from omotion.MotionInterface import MotionInterface
    return MotionInterface(
        demo_mode=True,
        data_dir=data_dir,
        scan_db_path=scan_db_path,
        operator_id="test",
    )


def test_start_scan_uses_new_runner():
    """After Phase E, ScanWorkflow.start_scan attaches a ScanRunner instance
    to self._runner instead of a SciencePipeline."""
    from omotion.pipeline.runner import ScanRunner
    motion = _build_motion_with_data_dir(None)
    request = ScanRequest(
        subject_id="x", duration_sec=1,
        left_camera_mask=0xFF, right_camera_mask=0, reduced_mode=False,
    )
    motion.scan_workflow.start_scan(request)
    assert isinstance(motion.scan_workflow._runner, ScanRunner)


def test_start_scan_auto_injects_csv_sink_when_data_dir_set(tmp_path):
    from omotion.pipeline.sinks import CsvSink
    motion = _build_motion_with_data_dir(str(tmp_path))
    request = ScanRequest(
        subject_id="x", duration_sec=1,
        left_camera_mask=0xFF, right_camera_mask=0, reduced_mode=False,
    )
    motion.scan_workflow.start_scan(request)
    csv_sinks = [s for s in motion.scan_workflow._runner.sinks if isinstance(s, CsvSink)]
    assert len(csv_sinks) == 1


def test_start_scan_skips_csv_sink_when_data_dir_none():
    from omotion.pipeline.sinks import CsvSink
    motion = _build_motion_with_data_dir(None)
    request = ScanRequest(
        subject_id="x", duration_sec=1,
        left_camera_mask=0xFF, right_camera_mask=0, reduced_mode=False,
    )
    motion.scan_workflow.start_scan(request)
    csv_sinks = [s for s in motion.scan_workflow._runner.sinks if isinstance(s, CsvSink)]
    assert csv_sinks == []


def test_start_scan_skips_default_storage_when_request_opts_out(tmp_path):
    from omotion.pipeline.sinks import CsvSink
    motion = _build_motion_with_data_dir(str(tmp_path))
    request = ScanRequest(
        subject_id="x", duration_sec=1,
        left_camera_mask=0xFF, right_camera_mask=0, reduced_mode=False,
        skip_default_storage=True,
    )
    motion.scan_workflow.start_scan(request)
    assert all(not isinstance(s, CsvSink) for s in motion.scan_workflow._runner.sinks)


def test_start_scan_auto_wires_telemetry_source_when_telemetry_sink_present():
    from omotion.pipeline.sinks import TelemetrySink
    from omotion.pipeline.sources import ConsoleTelemetrySource
    motion = _build_motion_with_data_dir(None)
    sink = TelemetrySink(output_path="/tmp/cq_telemetry.csv")
    request = ScanRequest(
        subject_id="x", duration_sec=1,
        left_camera_mask=0xFF, right_camera_mask=0, reduced_mode=False,
        sinks=[sink],
    )
    motion.scan_workflow.start_scan(request)
    assert isinstance(motion.scan_workflow._runner.telemetry_source, ConsoleTelemetrySource)


def test_start_scan_no_telemetry_source_when_no_telemetry_sink():
    motion = _build_motion_with_data_dir(None)
    request = ScanRequest(
        subject_id="x", duration_sec=1,
        left_camera_mask=0xFF, right_camera_mask=0, reduced_mode=False,
    )
    motion.scan_workflow.start_scan(request)
    assert motion.scan_workflow._runner.telemetry_source is None


def test_start_scan_passes_raw_save_max_duration_s_to_pipeline():
    motion = _build_motion_with_data_dir(None)
    request = ScanRequest(
        subject_id="x", duration_sec=1,
        left_camera_mask=0xFF, right_camera_mask=0, reduced_mode=False,
        raw_save_max_duration_s=60.0,
    )
    motion.scan_workflow.start_scan(request)
    raw_tee = next(s for s in motion.scan_workflow._runner.pipeline.stages
                   if s.name == "tee:raw")
    assert raw_tee.max_duration_s == 60.0
