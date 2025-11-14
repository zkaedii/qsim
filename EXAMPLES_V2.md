# Claude Analysis Agent V2 - Usage Examples

## Table of Contents

1. [Basic Usage](#basic-usage)
2. [Advanced Configuration](#advanced-configuration)
3. [Testing Scenarios](#testing-scenarios)
4. [Error Handling](#error-handling)
5. [Custom Data Sources](#custom-data-sources)
6. [Metrics and Monitoring](#metrics-and-monitoring)
7. [Production Deployment](#production-deployment)

## Basic Usage

### Quick Start

```python
#!/usr/bin/env python3
"""Quick start example for Claude Analysis Agent V2"""

from claude_analysis_agent_v2 import ClaudeAnalysisAgentV2, ConfigurationManager

# Initialize agent with default configuration
config = ConfigurationManager()
agent = ClaudeAnalysisAgentV2(config)

# Run analysis
result = agent.run_comprehensive_analysis()

# Display results
summary = result['executive_summary']
print(f"Analysis Status: {summary['enterprise_status']}")
print(f"Confidence: {summary['optimization_confidence']}")
print(f"Recommendation: {summary['key_achievement']}")
```

### With Test Data

```python
#!/usr/bin/env python3
"""Example using test fixtures"""

from claude_analysis_agent_v2 import ClaudeAnalysisAgentV2, ConfigurationManager
from test_fixtures import TestDataGenerator

# Generate test data files
TestDataGenerator.save_test_data_files()

# Run analysis
config = ConfigurationManager()
agent = ClaudeAnalysisAgentV2(config)
result = agent.run_comprehensive_analysis()

print("✅ Analysis complete!")
print(f"Analyzed {len(result['performance_trends']['performance_evolution']['top_3_methods'])} top methods")
```

## Advanced Configuration

### Custom Configuration File

```python
#!/usr/bin/env python3
"""Example with custom configuration"""

from claude_analysis_agent_v2 import ClaudeAnalysisAgentV2, ConfigurationManager

# Load custom configuration
config = ConfigurationManager("custom_config.yaml")

# Verify configuration
print(f"Environment: {config.get('application.environment')}")
print(f"Confidence Level: {config.get('statistics.confidence_level')}")

# Initialize agent
agent = ClaudeAnalysisAgentV2(config)
result = agent.run_comprehensive_analysis()
```

### Programmatic Configuration Override

```python
#!/usr/bin/env python3
"""Example with configuration override"""

from claude_analysis_agent_v2 import ClaudeAnalysisAgentV2, ConfigurationManager

config = ConfigurationManager()

# Override specific settings
config.config['statistics']['confidence_level'] = 0.99  # 99% confidence
config.config['performance_thresholds']['high_throughput_rps'] = 30000000

agent = ClaudeAnalysisAgentV2(config)
result = agent.run_comprehensive_analysis()

# Check applied confidence level
ci = result['performance_trends']['performance_evolution']['confidence_interval']
print(f"Applied confidence level: {ci['confidence_level']}")
```

## Testing Scenarios

### Testing with Mock Data

```python
#!/usr/bin/env python3
"""Example using mock data sources"""

from claude_analysis_agent_v2 import ClaudeAnalysisAgentV2, ConfigurationManager
from test_fixtures import TestScenarios

config = ConfigurationManager()

# Test normal operation
print("Testing normal operation...")
mock_sources = TestScenarios.normal_operation()
agent = ClaudeAnalysisAgentV2(config, data_sources=mock_sources)
result = agent.run_comprehensive_analysis()
assert result['executive_summary']['enterprise_status'] == 'VALIDATED'
print("✅ Normal operation test passed")

# Test with partial data
print("\nTesting partial data...")
mock_sources = TestScenarios.partial_data()
agent = ClaudeAnalysisAgentV2(config, data_sources=mock_sources)
result = agent.run_comprehensive_analysis()
print(f"✅ Partial data test passed - loaded {len(result['analysis_metadata']['data_sources'])} sources")

# Test with no data
print("\nTesting no data...")
mock_sources = TestScenarios.no_data()
agent = ClaudeAnalysisAgentV2(config, data_sources=mock_sources)
result = agent.run_comprehensive_analysis()
print(f"✅ No data test passed - status: {result['executive_summary']['enterprise_status']}")
```

### Performance Assertions

```python
#!/usr/bin/env python3
"""Example with performance assertions"""

from claude_analysis_agent_v2 import ClaudeAnalysisAgentV2, ConfigurationManager
from test_fixtures import TestScenarios, PerformanceAssertion

config = ConfigurationManager()
mock_sources = TestScenarios.normal_operation()
agent = ClaudeAnalysisAgentV2(config, data_sources=mock_sources)
result = agent.run_comprehensive_analysis()

# Assert performance within expected range
perf_evo = result['performance_trends']['performance_evolution']
PerformanceAssertion.assert_rps_within_range(
    actual_rps=perf_evo['best_performance'],
    expected_rps=95000000,
    tolerance_pct=5.0
)
print("✅ Performance within expected range")

# Assert improvement over baseline
PerformanceAssertion.assert_performance_improvement(
    baseline_rps=5000000,
    optimized_rps=perf_evo['best_performance'],
    min_improvement_pct=100.0
)
print("✅ Performance improvement verified")
```

## Error Handling

### Comprehensive Error Handling

```python
#!/usr/bin/env python3
"""Example with comprehensive error handling"""

from claude_analysis_agent_v2 import ClaudeAnalysisAgentV2, ConfigurationManager
from exceptions import (
    ClaudeAnalysisError,
    DataSourceNotFoundError,
    InsufficientDataError,
    CircuitBreakerError,
    InvalidConfigurationError
)

try:
    config = ConfigurationManager()
    agent = ClaudeAnalysisAgentV2(config)
    result = agent.run_comprehensive_analysis()
    
except DataSourceNotFoundError as e:
    print(f"❌ Data source not found: {e.message}")
    print(f"💡 Recovery hint: {e.recovery_hint}")
    # Fallback: use test data
    from test_fixtures import TestDataGenerator
    TestDataGenerator.save_test_data_files()
    print("✅ Generated test data, retrying...")
    
except InsufficientDataError as e:
    print(f"❌ Insufficient data: {e.message}")
    print(f"   Required: {e.context['required']}, Got: {e.context['actual']}")
    
except CircuitBreakerError as e:
    print(f"❌ Circuit breaker tripped: {e.message}")
    print(f"   Failures: {e.context['failure_count']}/{e.context['threshold']}")
    print(f"💡 Wait {e.context.get('timeout_seconds', 60)}s before retry")
    
except InvalidConfigurationError as e:
    print(f"❌ Configuration error: {e.message}")
    print(f"   Key: {e.context['config_key']}")
    print(f"   Reason: {e.context['reason']}")
    
except ClaudeAnalysisError as e:
    print(f"❌ Analysis error: {e.message}")
    print(f"   Error code: {e.error_code}")
    if e.recovery_hint:
        print(f"💡 {e.recovery_hint}")
    # Log error details
    import json
    print(json.dumps(e.to_dict(), indent=2))
    
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
```

### Retry with Exponential Backoff

```python
#!/usr/bin/env python3
"""Example with retry logic"""

import time
from claude_analysis_agent_v2 import ClaudeAnalysisAgentV2, ConfigurationManager
from exceptions import ClaudeAnalysisError

def run_analysis_with_retry(max_attempts=3):
    """Run analysis with retry logic"""
    config = ConfigurationManager()
    
    for attempt in range(1, max_attempts + 1):
        try:
            print(f"Attempt {attempt}/{max_attempts}...")
            agent = ClaudeAnalysisAgentV2(config)
            result = agent.run_comprehensive_analysis()
            print("✅ Analysis successful!")
            return result
            
        except ClaudeAnalysisError as e:
            if attempt < max_attempts:
                delay = 2 ** attempt  # Exponential backoff
                print(f"⚠️ Attempt failed: {e.message}")
                print(f"   Retrying in {delay}s...")
                time.sleep(delay)
            else:
                print(f"❌ All attempts failed")
                raise

# Run with retry
result = run_analysis_with_retry(max_attempts=3)
```

## Custom Data Sources

### Implementing Custom Data Source

```python
#!/usr/bin/env python3
"""Example with custom data source"""

from typing import Dict, Any
from claude_analysis_agent_v2 import ClaudeAnalysisAgentV2, ConfigurationManager

class DatabaseDataSource:
    """Custom data source that loads from database"""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
    
    def load(self) -> Dict[str, Any]:
        """Load performance data from database"""
        # Simulate database query
        return {
            'performance_evolution': {
                'method_a': 50000000,
                'method_b': 75000000,
                'method_c': 95000000
            }
        }

class APIDataSource:
    """Custom data source that fetches from API"""
    
    def __init__(self, api_endpoint: str):
        self.api_endpoint = api_endpoint
    
    def load(self) -> Dict[str, Any]:
        """Load performance data from API"""
        # Simulate API call
        return {
            'benchmark_results': {
                'test_1': {
                    'tasks_per_second': 85000000,
                    'processing_time_per_task_ns': 11.7,
                    'total_tasks': 1000000,
                    'duration_seconds': 11.7
                }
            }
        }

# Use custom data sources
config = ConfigurationManager()
custom_sources = {
    'performance_comparison': DatabaseDataSource('postgresql://localhost/metrics'),
    'ultimate_speed': APIDataSource('https://api.example.com/benchmarks')
}

agent = ClaudeAnalysisAgentV2(config, data_sources=custom_sources)
result = agent.run_comprehensive_analysis()

print(f"✅ Analysis complete with custom data sources")
print(f"   Data sources: {result['analysis_metadata']['data_sources']}")
```

### Streaming Data Source

```python
#!/usr/bin/env python3
"""Example with streaming data source"""

from typing import Dict, Any, Iterator
import time

class StreamingDataSource:
    """Data source that processes streaming data"""
    
    def __init__(self, stream_url: str):
        self.stream_url = stream_url
        self.buffer = []
    
    def stream_data(self) -> Iterator[Dict[str, float]]:
        """Simulate streaming performance data"""
        for i in range(10):
            yield {
                f'method_{i}': 50000000 + (i * 5000000)
            }
            time.sleep(0.1)  # Simulate real-time data
    
    def load(self) -> Dict[str, Any]:
        """Collect streaming data into analysis format"""
        performance_data = {}
        
        for batch in self.stream_data():
            performance_data.update(batch)
        
        return {
            'performance_evolution': performance_data
        }

# Use streaming source
from claude_analysis_agent_v2 import ClaudeAnalysisAgentV2, ConfigurationManager

config = ConfigurationManager()
streaming_source = StreamingDataSource('tcp://localhost:9000')

agent = ClaudeAnalysisAgentV2(
    config,
    data_sources={'performance_comparison': streaming_source}
)

result = agent.run_comprehensive_analysis()
print(f"✅ Analyzed {len(result['performance_trends']['performance_evolution']['top_3_methods'])} methods from stream")
```

## Metrics and Monitoring

### Accessing Metrics

```python
#!/usr/bin/env python3
"""Example accessing collected metrics"""

from claude_analysis_agent_v2 import ClaudeAnalysisAgentV2, ConfigurationManager
from metrics import get_metrics_collector

config = ConfigurationManager()
agent = ClaudeAnalysisAgentV2(config)
result = agent.run_comprehensive_analysis()

# Get metrics collector
metrics = get_metrics_collector()
all_metrics = metrics.get_all_metrics()

# Display metrics
print("📊 Collected Metrics:")
print(f"\nCounters:")
for name, value in all_metrics['counters'].items():
    print(f"  {name}: {value}")

print(f"\nGauges:")
for name, value in all_metrics['gauges'].items():
    print(f"  {name}: {value}")

print(f"\nTimers:")
for name, stats in all_metrics['timers'].items():
    if stats:
        print(f"  {name}:")
        print(f"    Count: {stats['count']}")
        print(f"    Mean: {stats['mean']:.6f}s")
        print(f"    P95: {stats['p95']:.6f}s")
        print(f"    P99: {stats['p99']:.6f}s")
```

### Custom Metrics

```python
#!/usr/bin/env python3
"""Example with custom metrics"""

from claude_analysis_agent_v2 import ClaudeAnalysisAgentV2, ConfigurationManager
from metrics import get_metrics_collector, trace_operation, StructuredLogger

config = ConfigurationManager()
metrics = get_metrics_collector()
logger = StructuredLogger(__name__)

# Custom operation with metrics
with trace_operation("custom_analysis", logger):
    # Custom metric collection
    metrics.counter("custom.operations.started")
    metrics.gauge("custom.data_sources.count", 4)
    
    agent = ClaudeAnalysisAgentV2(config)
    result = agent.run_comprehensive_analysis()
    
    # Record custom timing
    metrics.timer("custom.end_to_end.duration", 0.025)
    metrics.counter("custom.operations.completed")

print("✅ Custom metrics recorded")

# Export metrics
import json
with open('custom_metrics.json', 'w') as f:
    json.dump(metrics.get_all_metrics(), f, indent=2)
```

### Audit Trail Access

```python
#!/usr/bin/env python3
"""Example accessing audit trail"""

from claude_analysis_agent_v2 import ClaudeAnalysisAgentV2, ConfigurationManager
from metrics import AuditTrail

config = ConfigurationManager()
agent = ClaudeAnalysisAgentV2(config)
result = agent.run_comprehensive_analysis()

# Access audit trail
audit = AuditTrail()

# Get all events
all_events = audit.get_events()
print(f"📋 Total audit events: {len(all_events)}")

# Get specific event types
init_events = audit.get_events(event_type="agent_initialized")
print(f"   Initialization events: {len(init_events)}")

completed_events = audit.get_events(event_type="analysis_completed")
print(f"   Completion events: {len(completed_events)}")

# Display recent events
recent = audit.get_events(limit=5)
print("\n📝 Recent audit events:")
for event in recent:
    print(f"   [{event['timestamp']}] {event['event_type']}: {event['description']}")
```

## Production Deployment

### Docker Deployment

```python
# Dockerfile example
"""
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY claude_analysis_agent_v2.py .
COPY config.yaml .
COPY exceptions.py .
COPY validators.py .
COPY metrics.py .

# Set environment
ENV PYTHONUNBUFFERED=1
ENV ENVIRONMENT=production

# Run agent
CMD ["python3", "claude_analysis_agent_v2.py"]
"""
```

### Kubernetes Deployment

```yaml
# kubernetes-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: claude-analysis-agent
spec:
  replicas: 3
  selector:
    matchLabels:
      app: claude-analysis-agent
  template:
    metadata:
      labels:
        app: claude-analysis-agent
    spec:
      containers:
      - name: agent
        image: claude-analysis-agent:v2.0.0
        env:
        - name: ENVIRONMENT
          value: "production"
        resources:
          limits:
            memory: "2Gi"
            cpu: "1000m"
          requests:
            memory: "512Mi"
            cpu: "250m"
        volumeMounts:
        - name: config
          mountPath: /app/config.yaml
          subPath: config.yaml
      volumes:
      - name: config
        configMap:
          name: claude-agent-config
```

### Scheduled Analysis

```python
#!/usr/bin/env python3
"""Example with scheduled analysis"""

import schedule
import time
from datetime import datetime
from claude_analysis_agent_v2 import ClaudeAnalysisAgentV2, ConfigurationManager

def run_scheduled_analysis():
    """Run analysis on schedule"""
    print(f"\n🕐 Running scheduled analysis at {datetime.now()}")
    
    config = ConfigurationManager()
    agent = ClaudeAnalysisAgentV2(config)
    
    try:
        result = agent.run_comprehensive_analysis()
        
        # Save timestamped results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_{timestamp}.json"
        
        import json
        with open(filename, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        print(f"✅ Analysis saved to {filename}")
        
    except Exception as e:
        print(f"❌ Scheduled analysis failed: {e}")

# Schedule analysis every hour
schedule.every().hour.do(run_scheduled_analysis)

# Or schedule at specific times
schedule.every().day.at("09:00").do(run_scheduled_analysis)
schedule.every().day.at("17:00").do(run_scheduled_analysis)

print("📅 Scheduler started. Press Ctrl+C to exit.")

# Run scheduled tasks
while True:
    schedule.run_pending()
    time.sleep(60)
```

### Monitoring Integration

```python
#!/usr/bin/env python3
"""Example with Prometheus metrics export"""

from claude_analysis_agent_v2 import ClaudeAnalysisAgentV2, ConfigurationManager
from metrics import get_metrics_collector

config = ConfigurationManager()
agent = ClaudeAnalysisAgentV2(config)
result = agent.run_comprehensive_analysis()

# Export metrics in Prometheus format
metrics = get_metrics_collector()
all_metrics = metrics.get_all_metrics()

def export_prometheus_metrics(metrics_dict):
    """Export metrics in Prometheus format"""
    lines = []
    
    # Counters
    for name, value in metrics_dict['counters'].items():
        prom_name = name.replace('.', '_')
        lines.append(f'# TYPE {prom_name} counter')
        lines.append(f'{prom_name} {value}')
    
    # Gauges
    for name, value in metrics_dict['gauges'].items():
        prom_name = name.replace('.', '_')
        lines.append(f'# TYPE {prom_name} gauge')
        lines.append(f'{prom_name} {value}')
    
    return '\n'.join(lines)

# Write metrics
with open('/var/lib/node_exporter/textfile_collector/claude_agent.prom', 'w') as f:
    f.write(export_prometheus_metrics(all_metrics))

print("✅ Metrics exported for Prometheus")
```

## Summary

These examples demonstrate:
- ✅ Basic and advanced usage patterns
- ✅ Testing with mock data
- ✅ Comprehensive error handling
- ✅ Custom data source implementation
- ✅ Metrics collection and monitoring
- ✅ Production deployment strategies

For more information, see [README_CLAUDE_V2.md](README_CLAUDE_V2.md)
