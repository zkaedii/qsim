"""
Performance Monitoring Utilities

This module provides performance monitoring decorators and utilities
for tracking execution time, memory usage, and performance statistics.
"""

import logging
import sys
import time

try:
    import functools
except ImportError:
    functools = None  # type: ignore


logger = logging.getLogger(__name__)


def performance_monitor(func):
    """Enhanced performance monitoring with statistical tracking."""
    if functools is None:
        return func

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        start_memory = (
            sum(sys.getsizeof(arg) for arg in args)
            + sum(sys.getsizeof(kwarg) for kwarg in kwargs.items())
            if args or kwargs
            else 0
        )

        try:
            result = func(*args, **kwargs)

            execution_time = time.perf_counter() - start_time
            end_memory = sys.getsizeof(result) if result else 0
            memory_delta = end_memory - start_memory

            if not hasattr(func, "_performance_stats"):
                func._performance_stats = {
                    "call_count": 0,
                    "total_time": 0,
                    "min_time": float("inf"),
                    "max_time": 0,
                    "memory_usage": [],
                }

            stats = func._performance_stats
            stats["call_count"] += 1
            stats["total_time"] += execution_time
            stats["min_time"] = min(stats["min_time"], execution_time)
            stats["max_time"] = max(stats["max_time"], execution_time)
            stats["memory_usage"].append(memory_delta)

            if len(stats["memory_usage"]) > 100:
                stats["memory_usage"] = stats["memory_usage"][-100:]

            avg_time = stats["total_time"] / stats["call_count"]
            avg_memory = sum(stats["memory_usage"]) / len(stats["memory_usage"])

            logger.debug(
                f"Performance [{func.__name__}]: {execution_time:.4f}s "
                f"(avg: {avg_time:.4f}s, calls: {stats['call_count']}), "
                f"Memory delta: {memory_delta} bytes (avg: {avg_memory:.0f})"
            )

            if execution_time > 10.0:
                logger.warning(
                    f"Slow operation detected: {func.__name__} took {execution_time:.4f}s"
                )

            if abs(memory_delta) > 10 * 1024 * 1024:
                logger.warning(
                    f"Large memory change: {func.__name__} delta: {memory_delta/1024/1024:.1f}MB"
                )

            return result

        except Exception as e:
            execution_time = time.perf_counter() - start_time
            logger.error(f"Performance [{func.__name__}] FAILED after {execution_time:.4f}s: {e}")
            raise

    return wrapper


def get_performance_stats(func) -> dict:
    """Get performance statistics for a monitored function."""
    if hasattr(func, "_performance_stats"):
        return func._performance_stats
    return {}


def reset_performance_stats(func) -> None:
    """Reset performance statistics for a monitored function."""
    if hasattr(func, "_performance_stats"):
        func._performance_stats = {
            "call_count": 0,
            "total_time": 0,
            "min_time": float("inf"),
            "max_time": 0,
            "memory_usage": [],
        }
