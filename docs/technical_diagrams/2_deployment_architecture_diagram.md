# Mesa-CrewAI Hybrid Architecture: Deployment Architecture

## Runtime Topology and Process Boundaries

```mermaid
deploymentDiagram
    %% Single Process Deployment (Development/Testing)
    node "Single Process Deployment" {
        component "Main Application" {
            [HybridSimulationEngine]
            [UnifiedStateManager]
            [PerformanceManager]
            [ErrorManager]
        }
        
        component "Mesa Runtime" {
            [Mesa Model]
            [Mesa Scheduler]
            [Mesa Agents]
        }
        
        component "CrewAI Runtime" {
            [CrewAI Agents]
            [Memory System]
            [Task Executor]
        }
        
        database "Local Storage" {
            [State Snapshots]
            [Performance Metrics]
            [Error Logs]
            [Agent Memories]
        }
    }

    %% Distributed Deployment (Production)
    node "Compute Node 1" {
        component "Simulation Controller" {
            [HybridSimulationEngine]
            [State Coordinator]
            [Load Balancer]
        }
    }

    node "Compute Node 2" {
        component "Mesa Processing Unit" {
            [Mesa Model Cluster]
            [Spatial Engine]
            [Physics Simulator]
        }
    }

    node "Compute Node 3" {
        component "CrewAI Processing Unit" {
            [Agent Reasoning Pool]
            [Memory Manager]
            [LLM Connection Pool]
        }
    }

    node "State Management Cluster" {
        component "State Service" {
            [UnifiedStateManager]
            [State Validator]
            [Change Processor]
        }
        
        database "Distributed State DB" {
            [Primary State Store]
            [State Replicas]
            [Change Log]
        }
    }

    %% External Services
    cloud "LLM Providers" {
        [Google Gemini API]
        [OpenAI API]
        [Anthropic API]
    }

    cloud "Monitoring Services" {
        [Performance Analytics]
        [Error Tracking]
        [Health Monitoring]
    }

    database "Persistent Storage" {
        [PostgreSQL Cluster]
        [Redis Cache]
        [ChromaDB Vector Store]
    }

    %% Containerized Deployment (Kubernetes)
    node "Kubernetes Cluster" {
        component "Simulation Pods" {
            [Sim Controller Pod]
            [Mesa Workers Pod]
            [CrewAI Workers Pod]
        }
        
        component "State Management Pods" {
            [State Manager Pod]
            [Cache Service Pod]
            [Vector DB Pod]
        }
        
        component "Supporting Services" {
            [Performance Monitor Pod]
            [Error Handler Pod]
            [API Gateway Pod]
        }
    }
```

## Deployment Configurations

### 1. **Single Process Configuration**

```yaml
deployment:
  type: single_process
  
architecture:
  components:
    - hybrid_simulation_engine
    - unified_state_manager
    - performance_manager
    - error_manager
  
  resources:
    memory: 4GB
    cpu: 4 cores
    storage: 1GB SSD
  
  configuration:
    max_agents: 10
    simulation_steps: 1000
    cache_size: 100MB
    
dependencies:
  external_apis:
    - gemini_api
    - openai_api (fallback)
  
  local_storage:
    - sqlite_db
    - file_system
```

### 2. **Distributed Configuration**

```yaml
deployment:
  type: distributed
  
nodes:
  simulation_controller:
    replicas: 1
    resources:
      memory: 8GB
      cpu: 8 cores
    components:
      - hybrid_simulation_engine
      - load_balancer
      - state_coordinator
  
  mesa_processors:
    replicas: 3
    resources:
      memory: 16GB
      cpu: 16 cores
    components:
      - mesa_model_cluster
      - spatial_engine
      - physics_simulator
  
  crewai_processors:
    replicas: 5
    resources:
      memory: 12GB
      cpu: 8 cores
    components:
      - agent_reasoning_pool
      - memory_manager
      - llm_connection_pool
  
  state_managers:
    replicas: 2
    resources:
      memory: 32GB
      cpu: 16 cores
    components:
      - unified_state_manager
      - state_validator
      - change_processor

database:
  primary_store:
    type: postgresql
    replicas: 3
    resources:
      memory: 64GB
      storage: 1TB SSD
  
  cache_layer:
    type: redis
    replicas: 2
    resources:
      memory: 16GB
  
  vector_store:
    type: chromadb
    replicas: 2
    resources:
      memory: 32GB
      storage: 500GB SSD
```

### 3. **Containerized Configuration (Docker/Kubernetes)**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hybrid-simulation-engine
spec:
  replicas: 1
  selector:
    matchLabels:
      app: simulation-controller
  template:
    metadata:
      labels:
        app: simulation-controller
    spec:
      containers:
      - name: simulation-engine
        image: hybrid-simulation:latest
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        env:
        - name: GEMINI_API_KEY
          valueFrom:
            secretKeyRef:
              name: llm-secrets
              key: gemini-key
        - name: DATABASE_URL
          valueFrom:
            configMapKeyRef:
              name: db-config
              key: postgres-url

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mesa-processors
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mesa-worker
  template:
    metadata:
      labels:
        app: mesa-worker
    spec:
      containers:
      - name: mesa-processor
        image: mesa-worker:latest
        resources:
          requests:
            memory: "8Gi"
            cpu: "4"
          limits:
            memory: "16Gi"
            cpu: "8"

---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: state-managers
spec:
  serviceName: state-service
  replicas: 2
  selector:
    matchLabels:
      app: state-manager
  template:
    metadata:
      labels:
        app: state-manager
    spec:
      containers:
      - name: state-manager
        image: state-manager:latest
        resources:
          requests:
            memory: "16Gi"
            cpu: "8"
          limits:
            memory: "32Gi"
            cpu: "16"
        volumeMounts:
        - name: state-storage
          mountPath: /data
  volumeClaimTemplates:
  - metadata:
      name: state-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 100Gi
```

## External Dependencies and Integrations

### LLM API Providers

```python
# Configuration Management
LLM_PROVIDERS = {
    "primary": {
        "provider": "google_gemini",
        "model": "gemini-2.5-flash-lite", 
        "api_key_env": "GEMINI_API_KEY",
        "timeout": 30,
        "retry_attempts": 3,
        "circuit_breaker": {
            "failure_threshold": 5,
            "timeout_duration": 60
        }
    },
    "fallback": [
        {
            "provider": "openai",
            "model": "gpt-4",
            "api_key_env": "OPENAI_API_KEY",
            "timeout": 45,
            "retry_attempts": 2
        },
        {
            "provider": "anthropic", 
            "model": "claude-3-sonnet",
            "api_key_env": "ANTHROPIC_API_KEY",
            "timeout": 60,
            "retry_attempts": 2
        }
    ]
}
```

### Database Integrations

```python
# Multi-Database Configuration
DATABASE_CONFIG = {
    "primary_state": {
        "type": "postgresql",
        "url": "postgresql://user:pass@state-db:5432/hybrid_sim",
        "pool_size": 20,
        "max_overflow": 30,
        "connection_timeout": 30
    },
    "cache": {
        "type": "redis",
        "url": "redis://cache-cluster:6379/0",
        "pool_size": 10,
        "socket_timeout": 5
    },
    "vector_store": {
        "type": "chromadb",
        "host": "vector-db",
        "port": 8000,
        "collection_name": "agent_memories"
    },
    "metrics": {
        "type": "timeseries",
        "url": "influxdb://metrics-db:8086/performance",
        "retention_policy": "7d"
    }
}
```

## Scalability Considerations

### Horizontal Scaling Strategies

1. **Mesa Processing Scaling**
   - Partition simulation space across multiple nodes
   - Spatial locality for agent distribution
   - Grid-based load balancing

2. **CrewAI Agent Scaling**
   - Agent pool management with dynamic scaling
   - LLM request batching and connection pooling
   - Memory system partitioning by agent groups

3. **State Management Scaling**
   - Sharded state storage by entity type
   - Read replicas for query performance
   - Change log streaming for real-time sync

### Performance Optimization

```python
# Scaling Configuration
SCALING_CONFIG = {
    "auto_scaling": {
        "enabled": True,
        "metrics": ["cpu_usage", "memory_usage", "queue_depth"],
        "thresholds": {
            "scale_up": {"cpu": 70, "memory": 80, "queue": 100},
            "scale_down": {"cpu": 30, "memory": 40, "queue": 10}
        }
    },
    
    "resource_limits": {
        "max_mesa_workers": 10,
        "max_crewai_workers": 20,
        "max_state_replicas": 5
    },
    
    "performance_targets": {
        "simulation_step_latency": "100ms",
        "llm_response_time": "2s",
        "state_sync_latency": "50ms"
    }
}
```

## Network Architecture

```mermaid
graph TB
    %% Load Balancer
    LB[Load Balancer<br/>NGINX/HAProxy]
    
    %% API Gateway
    GW[API Gateway<br/>Kong/Istio]
    
    %% Application Layer
    subgraph "Application Cluster"
        SC[Simulation Controller]
        MW1[Mesa Worker 1]
        MW2[Mesa Worker 2] 
        MW3[Mesa Worker 3]
        CW1[CrewAI Worker 1]
        CW2[CrewAI Worker 2]
        CW3[CrewAI Worker 3]
    end
    
    %% State Layer
    subgraph "State Management"
        SM1[State Manager 1]
        SM2[State Manager 2]
        CACHE[Redis Cluster]
    end
    
    %% Data Layer
    subgraph "Data Storage"
        PG[(PostgreSQL Cluster)]
        CHROMA[(ChromaDB)]
        METRICS[(InfluxDB)]
    end
    
    %% External Services
    subgraph "External APIs"
        GEMINI[Google Gemini]
        OPENAI[OpenAI]
        ANTHROPIC[Anthropic]
    end
    
    %% Monitoring
    subgraph "Monitoring Stack"
        PROM[Prometheus]
        GRAF[Grafana]
        ALERT[AlertManager]
    end
    
    %% Connections
    LB --> GW
    GW --> SC
    SC --> MW1
    SC --> MW2
    SC --> MW3
    SC --> CW1
    SC --> CW2
    SC --> CW3
    
    MW1 --> SM1
    MW2 --> SM1
    MW3 --> SM2
    CW1 --> SM1
    CW2 --> SM2
    CW3 --> SM2
    
    SM1 --> PG
    SM2 --> PG
    SM1 --> CACHE
    SM2 --> CACHE
    
    CW1 --> CHROMA
    CW2 --> CHROMA
    CW3 --> CHROMA
    
    CW1 --> GEMINI
    CW2 --> OPENAI
    CW3 --> ANTHROPIC
    
    SC --> METRICS
    SM1 --> METRICS
    SM2 --> METRICS
    
    PROM --> GRAF
    PROM --> ALERT
```

## Security and Configuration Management

### Security Configuration

```yaml
security:
  api_keys:
    storage: kubernetes_secrets
    rotation: 30d
    
  network:
    tls_version: "1.3"
    cipher_suites: ["TLS_AES_256_GCM_SHA384"]
    
  database:
    encryption_at_rest: true
    encryption_in_transit: true
    
  monitoring:
    audit_logging: enabled
    access_logging: enabled

access_control:
  rbac:
    enabled: true
    roles:
      - simulation_admin
      - simulation_operator  
      - read_only_viewer
      
  network_policies:
    default_deny: true
    allowed_ingress:
      - port: 8080
        protocol: HTTP
        source: load_balancer
```

### Environment Configuration

```bash
# Production Environment Variables
export DEPLOYMENT_TYPE=distributed
export ENVIRONMENT=production
export LOG_LEVEL=INFO

# Database URLs
export POSTGRES_URL=postgresql://sim_user:***@postgres-cluster:5432/hybrid_sim
export REDIS_URL=redis://redis-cluster:6379/0
export CHROMADB_URL=http://chromadb-cluster:8000

# LLM API Keys (from Kubernetes secrets)
export GEMINI_API_KEY=${GEMINI_SECRET}
export OPENAI_API_KEY=${OPENAI_SECRET}

# Performance Tuning
export MAX_WORKERS=50
export BATCH_SIZE=10
export CACHE_TTL=300

# Monitoring
export PROMETHEUS_URL=http://prometheus:9090
export GRAFANA_URL=http://grafana:3000
```

This deployment architecture provides multiple deployment options from single-process development to fully distributed production systems, with comprehensive external service integration and scalability planning.