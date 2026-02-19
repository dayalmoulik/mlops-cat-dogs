# Monitoring & Logging Guide

## Overview

The API includes comprehensive monitoring and logging capabilities using Prometheus metrics and structured JSON logging.

## Logging

### Structured JSON Logging

All logs are output in JSON format for easy parsing and aggregation.

**Log Fields:**
- `timestamp`: ISO 8601 timestamp
- `level`: Log level (DEBUG, INFO, WARNING, ERROR)
- `logger`: Logger name
- `service`: Service name (cats-dogs-classifier)
- `message`: Log message
- `method`, `url`, `status`: Request details (for request logs)
- `duration`: Request duration (for completion logs)

**Example Log:**
```json
{
  "timestamp": "2024-02-19T10:30:00.123456",
  "level": "INFO",
  "logger": "__main__",
  "service": "cats-dogs-classifier",
  "message": "Prediction: cat (confidence: 0.9567)",
  "prediction": "cat",
  "confidence": 0.9567,
  "processing_time_ms": 245.67
}
```

### Log Levels

Set via `LOG_LEVEL` environment variable:
- `DEBUG`: Detailed information for debugging
- `INFO`: General informational messages (default)
- `WARNING`: Warning messages
- `ERROR`: Error messages

**Example:**
```powershell
$env:LOG_LEVEL="DEBUG"
python src/api/main.py
```

## Prometheus Metrics

### Available Metrics

#### Request Metrics

**api_requests_total**
- Type: Counter
- Description: Total number of API requests
- Labels: `method`, `endpoint`, `status`

**api_request_duration_seconds**
- Type: Histogram
- Description: API request duration in seconds
- Labels: `method`, `endpoint`

**active_requests**
- Type: Gauge
- Description: Number of currently active requests

#### Prediction Metrics

**predictions_total**
- Type: Counter
- Description: Total predictions made
- Labels: `predicted_class` (cat/dog)

**prediction_confidence**
- Type: Histogram
- Description: Distribution of prediction confidence scores
- Buckets: 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0

#### Model Metrics

**model_load_time_seconds**
- Type: Gauge
- Description: Time taken to load the model on startup

### Accessing Metrics

**Endpoint:** `/metrics`
```powershell
curl http://localhost:8000/metrics
```

**Example Output:**
```
# HELP api_requests_total Total API requests
# TYPE api_requests_total counter
api_requests_total{endpoint="/predict",method="POST",status="200"} 42.0

# HELP prediction_confidence Prediction confidence scores
# TYPE prediction_confidence histogram
prediction_confidence_bucket{le="0.9"} 5.0
prediction_confidence_bucket{le="0.95"} 15.0
prediction_confidence_bucket{le="0.99"} 38.0
prediction_confidence_bucket{le="1.0"} 42.0
prediction_confidence_count 42.0
prediction_confidence_sum 39.87
```

## Integration with Monitoring Systems

### Prometheus

Add to `prometheus.yml`:
```yaml
scrape_configs:
  - job_name: 'cats-dogs-api'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

### Grafana Dashboard

Import the provided dashboard JSON or create custom panels:

**Key Panels:**
1. Request Rate (requests/sec)
2. Request Duration (p50, p95, p99)
3. Prediction Confidence Distribution
4. Predictions by Class (cat vs dog)
5. Error Rate
6. Active Requests

### ELK Stack (Elasticsearch, Logstash, Kibana)

**Logstash Configuration:**
```
input {
  file {
    path => "/var/log/cats-dogs-api/*.log"
    codec => json
  }
}

filter {
  json {
    source => "message"
  }
}

output {
  elasticsearch {
    hosts => ["localhost:9200"]
    index => "cats-dogs-api-%{+YYYY.MM.dd}"
  }
}
```

## Performance Monitoring

### Response Time Header

Every response includes processing time:
```
X-Process-Time: 0.245
```

### Health Check

Monitor service health:
```powershell
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu",
  "timestamp": "2024-02-19T10:30:00"
}
```

## Alerting

### Prometheus Alert Rules
```yaml
groups:
  - name: cats_dogs_api
    rules:
      - alert: HighErrorRate
        expr: rate(api_requests_total{status=~"5.."}[5m]) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: High error rate detected
      
      - alert: SlowRequests
        expr: histogram_quantile(0.95, rate(api_request_duration_seconds_bucket[5m])) > 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: 95th percentile latency > 1s
      
      - alert: ModelNotLoaded
        expr: up{job="cats-dogs-api"} == 1 and model_load_time_seconds == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: Model failed to load
```

## Best Practices

1. **Log Retention**: Rotate logs daily, keep 30 days
2. **Metrics Scraping**: 15-second intervals for Prometheus
3. **Alerting**: Set up alerts for error rates and latency
4. **Dashboards**: Create Grafana dashboards for visualization
5. **Tracing**: Consider adding distributed tracing (OpenTelemetry)

## Troubleshooting

### High Memory Usage
Check metrics for memory leaks:
```
process_resident_memory_bytes
```

### Slow Predictions
Check prediction confidence and duration histograms

### Missing Metrics
Verify `/metrics` endpoint is accessible

---

**Status**: Monitoring and logging complete
**Metrics**: 6 metric types tracking requests, predictions, and performance
**Logging**: Structured JSON logging with multiple levels
