#!/usr/bin/env python3
"""
Observability and Metrics Collection Module

Provides structured logging, metrics collection, performance tracing,
and audit trail capabilities for enterprise observability.
"""

import logging
import json
import time
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from functools import wraps
from contextlib import contextmanager
import threading
from dataclasses import dataclass, asdict, field


# Configure structured logging
class StructuredLogger:
    """Structured logging with JSON output and contextual information."""

    def __init__(self, name: str, level: int = logging.INFO):
        """
        Initialize structured logger.

        Args:
            name: Logger name
            level: Logging level
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Create console handler with JSON formatter
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(JsonFormatter())
            self.logger.addHandler(handler)

        self.context: Dict[str, Any] = {}

    def add_context(self, **kwargs):
        """Add persistent context to all log messages."""
        self.context.update(kwargs)

    def clear_context(self):
        """Clear persistent context."""
        self.context.clear()

    def _log(self, level: int, message: str, **kwargs):
        """Internal log method with context."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": logging.getLevelName(level),
            "message": message,
            **self.context,
            **kwargs,
        }
        self.logger.log(level, json.dumps(log_data))

    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._log(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error message."""
        self._log(logging.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self._log(logging.CRITICAL, message, **kwargs)


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Message is already JSON from StructuredLogger
        if isinstance(record.msg, str) and record.msg.startswith("{"):
            return record.msg

        # Fallback for regular messages
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        return json.dumps(log_data)


@dataclass
class MetricPoint:
    """A single metric measurement."""

    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)
    metric_type: str = "gauge"  # gauge, counter, histogram, timer

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class MetricsCollector:
    """Thread-safe metrics collection and aggregation."""

    def __init__(self):
        """Initialize metrics collector."""
        self.metrics: List[MetricPoint] = []
        self.counters: Dict[str, float] = {}
        self.gauges: Dict[str, float] = {}
        self.timers: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    def counter(self, name: str, value: float = 1, tags: Optional[Dict[str, str]] = None):
        """
        Increment a counter metric.

        Args:
            name: Counter name
            value: Increment value
            tags: Optional tags
        """
        with self._lock:
            self.counters[name] = self.counters.get(name, 0) + value
            self.metrics.append(
                MetricPoint(name=name, value=value, tags=tags or {}, metric_type="counter")
            )

    def gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """
        Set a gauge metric.

        Args:
            name: Gauge name
            value: Gauge value
            tags: Optional tags
        """
        with self._lock:
            self.gauges[name] = value
            self.metrics.append(
                MetricPoint(name=name, value=value, tags=tags or {}, metric_type="gauge")
            )

    def timer(self, name: str, duration: float, tags: Optional[Dict[str, str]] = None):
        """
        Record a timer metric.

        Args:
            name: Timer name
            duration: Duration in seconds
            tags: Optional tags
        """
        with self._lock:
            if name not in self.timers:
                self.timers[name] = []
            self.timers[name].append(duration)
            self.metrics.append(
                MetricPoint(name=name, value=duration, tags=tags or {}, metric_type="timer")
            )

    def get_counter(self, name: str) -> float:
        """Get counter value."""
        with self._lock:
            return self.counters.get(name, 0)

    def get_gauge(self, name: str) -> Optional[float]:
        """Get gauge value."""
        with self._lock:
            return self.gauges.get(name)

    def get_timer_stats(self, name: str) -> Optional[Dict[str, float]]:
        """Get timer statistics."""
        with self._lock:
            if name not in self.timers or not self.timers[name]:
                return None

            values = self.timers[name]
            return {
                "count": len(values),
                "sum": sum(values),
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "p50": sorted(values)[len(values) // 2],
                "p95": sorted(values)[int(len(values) * 0.95)],
                "p99": sorted(values)[int(len(values) * 0.99)],
            }

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics snapshot."""
        with self._lock:
            return {
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "timers": {name: self.get_timer_stats(name) for name in self.timers},
                "timestamp": time.time(),
            }

    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self.metrics.clear()
            self.counters.clear()
            self.gauges.clear()
            self.timers.clear()


# Global metrics collector instance
_metrics_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance."""
    return _metrics_collector


@contextmanager
def trace_operation(operation_name: str, logger: Optional[StructuredLogger] = None):
    """
    Context manager for tracing operations with timing.

    Args:
        operation_name: Name of operation
        logger: Optional logger for trace output

    Yields:
        Dictionary with operation metadata
    """
    start_time = time.time()
    metrics = get_metrics_collector()

    trace_data = {
        "operation": operation_name,
        "start_time": start_time,
        "trace_id": f"{operation_name}_{int(start_time * 1000)}",
    }

    if logger:
        logger.info(f"Operation started: {operation_name}", **trace_data)

    try:
        yield trace_data
        duration = time.time() - start_time
        trace_data["duration"] = duration
        trace_data["status"] = "success"

        metrics.timer(f"operation.{operation_name}.duration", duration)
        metrics.counter(f"operation.{operation_name}.success")

        if logger:
            logger.info(f"Operation completed: {operation_name}", **trace_data)
    except Exception as e:
        duration = time.time() - start_time
        trace_data["duration"] = duration
        trace_data["status"] = "error"
        trace_data["error"] = str(e)

        metrics.timer(f"operation.{operation_name}.duration", duration)
        metrics.counter(f"operation.{operation_name}.error")

        if logger:
            logger.error(f"Operation failed: {operation_name}", **trace_data)
        raise


def instrumented(operation_name: Optional[str] = None):
    """
    Decorator for instrumenting functions with metrics and tracing.

    Args:
        operation_name: Optional custom operation name

    Returns:
        Decorated function
    """

    def decorator(func: Callable) -> Callable:
        op_name = operation_name or f"{func.__module__}.{func.__name__}"

        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = StructuredLogger(func.__module__)
            with trace_operation(op_name, logger):
                return func(*args, **kwargs)

        return wrapper

    return decorator


class AuditTrail:
    """Audit trail for tracking significant events and decisions."""

    def __init__(self, filepath: str = "audit_trail.jsonl"):
        """
        Initialize audit trail.

        Args:
            filepath: Path to audit log file
        """
        self.filepath = filepath
        self._lock = threading.Lock()

    def log_event(
        self,
        event_type: str,
        description: str,
        actor: str = "system",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Log an audit event.

        Args:
            event_type: Type of event
            description: Event description
            actor: Who performed the action
            metadata: Additional metadata
        """
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "description": description,
            "actor": actor,
            "metadata": metadata or {},
        }

        with self._lock:
            with open(self.filepath, "a") as f:
                f.write(json.dumps(event) + "\n")

    def get_events(
        self, event_type: Optional[str] = None, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve audit events.

        Args:
            event_type: Filter by event type
            limit: Maximum number of events to return

        Returns:
            List of audit events
        """
        events = []

        try:
            with open(self.filepath, "r") as f:
                for line in f:
                    event = json.loads(line.strip())
                    if event_type is None or event["event_type"] == event_type:
                        events.append(event)
                        if limit and len(events) >= limit:
                            break
        except FileNotFoundError:
            pass

        return events


class PerformanceMonitor:
    """Monitor and report on performance metrics."""

    def __init__(self, logger: StructuredLogger):
        """
        Initialize performance monitor.

        Args:
            logger: Logger for reporting
        """
        self.logger = logger
        self.metrics = get_metrics_collector()

    def report_analysis_metrics(self, analysis_result: Dict[str, Any]):
        """
        Report metrics from analysis results.

        Args:
            analysis_result: Analysis results dictionary
        """
        # Extract and report key metrics
        if "performance_trends" in analysis_result:
            trends = analysis_result["performance_trends"]

            if "performance_evolution" in trends:
                perf = trends["performance_evolution"]
                self.metrics.gauge("analysis.best_performance", perf.get("best_performance", 0))
                self.metrics.gauge("analysis.performance_gap", perf.get("performance_gap", 0))

        if "claude_insights" in analysis_result:
            insights = analysis_result["claude_insights"]
            self.metrics.gauge(
                "analysis.recommendations_count",
                len(insights.get("optimization_recommendations", [])),
            )

        self.logger.info("Analysis metrics reported", metrics=self.metrics.get_all_metrics())
