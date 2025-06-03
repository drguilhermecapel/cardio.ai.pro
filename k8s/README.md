# Kubernetes Deployment for Cardio.AI.Pro

This directory contains Kubernetes manifests for deploying the Cardio.AI.Pro ECG analysis system with advanced cost optimization and scaling capabilities.

## Overview

The Kubernetes deployment includes:
- **Core Services**: PostgreSQL, Redis, FastAPI backend, Celery workers, React frontend
- **CAST AI Integration**: 60-66% cost reduction through predictive scaling and optimization
- **Monitoring**: Prometheus-based monitoring with custom ECG analysis metrics
- **Auto-scaling**: Horizontal Pod Autoscaling (HPA) with custom metrics
- **Security**: RBAC, Pod Disruption Budgets, and TLS termination

## Quick Start

### Prerequisites
- Kubernetes cluster (1.20+)
- kubectl configured
- CAST AI account and API key
- Ingress controller (nginx recommended)
- Cert-manager for TLS certificates

### Deployment Steps

1. **Create namespace and secrets:**
```bash
kubectl apply -f namespace.yaml
kubectl apply -f secrets.yaml
kubectl apply -f configmap.yaml
```

2. **Deploy storage and databases:**
```bash
kubectl apply -f postgres.yaml
kubectl apply -f redis.yaml
```

3. **Deploy application services:**
```bash
kubectl apply -f api.yaml
kubectl apply -f celery.yaml
kubectl apply -f frontend.yaml
```

4. **Configure networking:**
```bash
kubectl apply -f ingress.yaml
```

5. **Enable monitoring and cost optimization:**
```bash
kubectl apply -f monitoring.yaml
kubectl apply -f cast-ai-integration.yaml
```

## Architecture Components

### Core Services

#### API Service (`api.yaml`)
- **Replicas**: 3 (min) to 20 (max)
- **Resources**: 1-4Gi memory, 0.5-2 CPU cores, 0-1 GPU
- **Auto-scaling**: CPU (70%), Memory (80%), Custom ECG queue metrics
- **Health checks**: HTTP probes on `/health` endpoint

#### Celery Workers (`celery.yaml`)
- **Replicas**: 2 (min) to 10 (max)
- **Resources**: 0.5-2Gi memory, 0.25-1 CPU cores, 0-1 GPU
- **Auto-scaling**: Based on Celery queue length and resource utilization
- **Components**: Worker processes and Beat scheduler

#### Frontend (`frontend.yaml`)
- **Replicas**: 2 (min) to 8 (max)
- **Resources**: 128-512Mi memory, 0.1-0.5 CPU cores
- **Auto-scaling**: CPU and memory based
- **Serving**: Static React application via nginx

### Data Layer

#### PostgreSQL (`postgres.yaml`)
- **Storage**: 10Gi persistent volume (fast-ssd)
- **Resources**: 256Mi-1Gi memory, 0.25-1 CPU cores
- **Health checks**: pg_isready probes
- **Backup**: Automated via persistent volumes

#### Redis (`redis.yaml`)
- **Storage**: 5Gi persistent volume (fast-ssd)
- **Resources**: 128-512Mi memory, 0.1-0.5 CPU cores
- **Configuration**: Persistence enabled, password protected

### Cost Optimization

#### CAST AI Integration (`cast-ai-integration.yaml`)
- **Target Cost Reduction**: 65%
- **Optimization Level**: Aggressive
- **Spot Instance Ratio**: 70%
- **Features**:
  - Predictive scaling with 15-minute prediction window
  - Right-sizing recommendations
  - Workload rebalancing
  - Cluster hibernation for non-production environments
  - Unused resource cleanup

#### Auto-scaling Configuration
- **Scale-up**: <60 seconds target
- **Scale-down**: 5-minute stabilization window
- **Metrics**: CPU, Memory, Custom ECG processing metrics
- **Policies**: Aggressive scale-up, conservative scale-down

### Monitoring and Observability

#### Prometheus (`monitoring.yaml`)
- **Metrics Collection**: API, Celery, CAST AI, Kubernetes
- **Storage**: 20Gi persistent volume
- **Retention**: 30 days
- **Custom Metrics**:
  - ECG processing latency (95th percentile)
  - Analysis queue length
  - Cost optimization opportunities
  - Memory and CPU utilization

#### Alerting Rules
- **High ECG Processing Latency**: >10 seconds (95th percentile)
- **Queue Backlog**: >50 analyses pending
- **High Memory Usage**: >90% of limits
- **Cost Optimization**: >$1000 monthly savings potential

### Security and Reliability

#### RBAC Configuration
- Service accounts for CAST AI and Prometheus
- Cluster roles with minimal required permissions
- Namespace isolation

#### Pod Disruption Budgets
- **API**: Minimum 2 pods available
- **Celery Workers**: Minimum 1 pod available
- **Frontend**: Minimum 1 pod available

#### Network Policies
- Ingress-only traffic to frontend
- Database access restricted to backend services
- Inter-service communication secured

## Configuration

### Environment Variables
Update `configmap.yaml` with your environment-specific values:
- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string
- `REACT_APP_API_URL`: Frontend API endpoint
- `REACT_APP_WS_URL`: WebSocket endpoint

### Secrets
Update `secrets.yaml` with base64-encoded values:
- `postgres-password`: PostgreSQL password
- `redis-password`: Redis password
- `secret-key`: Application secret key

### CAST AI Configuration
Update `cast-ai-integration.yaml`:
- Replace `api-key` in `cast-ai-secrets` with your CAST AI API key
- Adjust `cluster-id` in `cast-ai-config`
- Modify optimization parameters as needed

## Scaling Behavior

### Predictive Scaling
The system uses machine learning models to predict load and scale proactively:
- **Prediction Window**: 15 minutes
- **Scale-up Threshold**: 70% resource utilization
- **Scale-down Threshold**: 30% resource utilization
- **Custom Metrics**: ECG analysis queue length, processing latency

### Resource Optimization
CAST AI continuously optimizes:
- **Node Selection**: Best price/performance ratio
- **Workload Placement**: Optimal node utilization
- **Spot Instance Management**: Automatic failover and rebalancing
- **Right-sizing**: Container resource recommendations

## Monitoring and Troubleshooting

### Health Checks
```bash
# Check pod status
kubectl get pods -n cardioai-pro

# Check service endpoints
kubectl get endpoints -n cardioai-pro

# View logs
kubectl logs -f deployment/api -n cardioai-pro
kubectl logs -f deployment/celery-worker -n cardioai-pro
```

### Metrics Access
```bash
# Port-forward to Prometheus
kubectl port-forward svc/prometheus 9090:9090 -n cardioai-pro

# Access metrics at http://localhost:9090
```

### CAST AI Dashboard
Access the CAST AI dashboard to monitor:
- Cost savings in real-time
- Optimization recommendations
- Cluster efficiency metrics
- Scaling events and decisions

## Performance Targets

### Latency
- **ECG Analysis**: <50ms for 10-second segments
- **API Response**: <200ms for standard requests
- **Scale-up Time**: <60 seconds

### Throughput
- **Concurrent Analyses**: 100+ simultaneous ECG processing
- **API Requests**: 1000+ requests/second
- **Queue Processing**: Real-time with minimal backlog

### Cost Efficiency
- **Target Savings**: 60-66% infrastructure cost reduction
- **Resource Utilization**: >80% average cluster utilization
- **Spot Instance Usage**: 70% of workloads on spot instances

## Maintenance

### Updates
```bash
# Rolling update for API
kubectl set image deployment/api api=cardioai/backend:new-version -n cardioai-pro

# Update configuration
kubectl apply -f configmap.yaml
kubectl rollout restart deployment/api -n cardioai-pro
```

### Backup
- Database backups via persistent volume snapshots
- Configuration backup via git repository
- Monitoring data retained for 30 days

### Disaster Recovery
- Multi-zone deployment for high availability
- Automated failover for stateless services
- Database replication and point-in-time recovery

## Support

For issues related to:
- **Kubernetes deployment**: Check pod logs and events
- **CAST AI optimization**: Review CAST AI dashboard and logs
- **Application performance**: Monitor Prometheus metrics
- **Cost optimization**: Analyze CAST AI recommendations

## Security Considerations

- All secrets stored in Kubernetes secrets (base64 encoded)
- TLS termination at ingress level
- Network policies restrict inter-pod communication
- RBAC limits service account permissions
- Regular security updates via rolling deployments
