"""Tests for bounded memory use when resetting camera capture."""

import subprocess as sp
import unittest
from collections import deque
from unittest.mock import MagicMock, call

from frigate.video.ffmpeg import CameraWatchdog


class TestCameraWatchdogReset(unittest.TestCase):
    """Verify reset never collects raw frames in the watchdog process."""

    def setUp(self):
        """Isolate the reset lifecycle from camera hardware and IPC."""
        self.watchdog = CameraWatchdog.__new__(CameraWatchdog)
        self.watchdog.ffmpeg_detect_process = MagicMock()
        self.watchdog.capture_thread = MagicMock()
        self.watchdog.capture_thread.is_alive.return_value = False
        self.watchdog.logger = MagicMock()
        self.watchdog.logpipe = MagicMock()
        self.watchdog.reconnect_timestamps = deque()
        self.watchdog.reconnects = MagicMock()
        self.watchdog.start_ffmpeg_detect = MagicMock()
        self.watchdog.detect_process_records_sub = False

    def test_reset_does_not_buffer_raw_frames(self):
        """A graceful reset waits for exit without collecting stdout."""
        process = self.watchdog.ffmpeg_detect_process

        self.watchdog.reset_capture_thread()

        self.assertEqual(process.mock_calls, [call.terminate(), call.wait(timeout=30)])
        self.watchdog.start_ffmpeg_detect.assert_called_once()

    def test_timed_out_reset_kills_and_reaps_without_reading_stdout(self):
        """An unresponsive child is killed and reaped after the timeout."""
        process = self.watchdog.ffmpeg_detect_process
        process.wait.side_effect = [sp.TimeoutExpired("ffmpeg", 30), 0]

        self.watchdog.reset_capture_thread()

        self.assertEqual(
            process.mock_calls,
            [call.terminate(), call.wait(timeout=30), call.kill(), call.wait()],
        )
        self.watchdog.start_ffmpeg_detect.assert_called_once()

    def test_exited_child_is_not_terminated_again(self):
        """The crashed-child path still joins capture before restarting."""
        self.watchdog.capture_thread.is_alive.side_effect = [True, False]

        self.watchdog.reset_capture_thread(terminate=False)

        self.assertEqual(self.watchdog.ffmpeg_detect_process.mock_calls, [])
        self.watchdog.capture_thread.join.assert_called_once_with(timeout=5)
        self.watchdog.start_ffmpeg_detect.assert_called_once()
        self.assertEqual(self.watchdog.reconnects.value, 1)
