"""Unit tests for whisperx.log_utils logging configuration."""

from __future__ import annotations

import logging
import sys

from whisperx.log_utils import _DATE_FORMAT, _LOG_FORMAT, get_logger, setup_logging


class TestSetupLogging:
    def test_sets_level_from_string(self, reset_logger_handlers):
        setup_logging(level="debug")
        logger = logging.getLogger("whisperx")
        assert logger.level == logging.DEBUG

    def test_invalid_level_falls_back_to_warning(self, reset_logger_handlers):
        setup_logging(level="bogus")
        logger = logging.getLogger("whisperx")
        assert logger.level == logging.WARNING

    def test_clears_existing_handlers(self, reset_logger_handlers):
        logger = logging.getLogger("whisperx")
        logger.addHandler(logging.StreamHandler())
        assert len(logger.handlers) == 1
        setup_logging(level="info")
        # one console handler is added by setup_logging
        assert len(logger.handlers) == 1

    def test_console_handler_attached(self, reset_logger_handlers):
        setup_logging(level="info")
        logger = logging.getLogger("whisperx")
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.StreamHandler)
        assert logger.handlers[0].stream is sys.stdout

    def test_propagate_false(self, reset_logger_handlers):
        setup_logging(level="info")
        assert logging.getLogger("whisperx").propagate is False

    def test_file_handler_added_when_path_given(self, tmp_path, reset_logger_handlers):
        log_file = tmp_path / "out.log"
        setup_logging(level="info", log_file=str(log_file))
        logger = logging.getLogger("whisperx")
        # console + file
        assert len(logger.handlers) == 2
        assert any(isinstance(h, logging.FileHandler) for h in logger.handlers)
        # The file handler must have a formatter set (kills setFormatter->None).
        file_h = next(h for h in logger.handlers if isinstance(h, logging.FileHandler))
        assert isinstance(file_h.formatter, logging.Formatter)
        assert file_h.formatter._fmt == _LOG_FORMAT
        logger.info("a message")
        for h in logger.handlers:
            h.flush()
        assert log_file.exists()

    def test_file_handler_failure_warns_and_continues(self, reset_logger_handlers, caplog):
        # A path inside a nonexistent directory triggers OSError on FileHandler.
        # Enable propagation so caplog (root handler) captures the warnings
        # emitted during setup_logging.
        logging.getLogger("whisperx").propagate = True
        with caplog.at_level(logging.WARNING, logger="whisperx"):
            setup_logging(level="info", log_file="/nonexistent_dir_xyz/abc/out.log")
        logger = logging.getLogger("whisperx")
        # console handler still present despite file handler failure
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.StreamHandler)
        # Assert the warning messages (kills string-literal mutants).
        assert any("Failed to create log file" in r.getMessage() for r in caplog.records)
        assert any("Continuing with console logging only" in r.getMessage() for r in caplog.records)

    def test_default_level_is_info(self, reset_logger_handlers):
        # setup_logging() with no level -> defaults to "info" -> INFO level.
        setup_logging()
        logger = logging.getLogger("whisperx")
        assert logger.level == logging.INFO

    def test_formatter_uses_config(self, reset_logger_handlers):
        setup_logging(level="info")
        logger = logging.getLogger("whisperx")
        handler = logger.handlers[0]
        assert isinstance(handler.formatter, logging.Formatter)
        assert handler.formatter._fmt == _LOG_FORMAT
        assert handler.formatter.datefmt == _DATE_FORMAT


class TestGetLogger:
    def test_main_name_resolves_to_whisperx(self, reset_logger_handlers):
        logger = get_logger("__main__")
        assert logger.name == "whisperx"

    def test_module_name_preserved(self, reset_logger_handlers):
        logger = get_logger("whisperx.alignment")
        assert logger.name == "whisperx.alignment"

    def test_initializes_handlers_if_missing(self, reset_logger_handlers):
        logger = logging.getLogger("whisperx")
        logger.handlers.clear()
        assert not logger.handlers
        get_logger("whisperx.utils")
        assert logging.getLogger("whisperx").handlers

    def test_does_not_reinit_when_handlers_exist(self, reset_logger_handlers):
        setup_logging(level="error")
        before = list(logging.getLogger("whisperx").handlers)
        get_logger("whisperx.alignment")
        after = logging.getLogger("whisperx").handlers
        # Same handler objects, no duplication
        assert before == after
