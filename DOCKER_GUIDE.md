# Docker Deployment Guide

## Building the Image

### Build locally
```powershell
docker build -t cats-dogs-classifier:latest .
```

### Build with specific tag
```powershell
docker build -t cats-dogs-classifier:v1.0 .
```

## Running the Container

### Run with docker
```powershell
docker run -d --name cats-dogs-api -p 8000:8000 cats-dogs-classifier:latest
```

### Run with docker-compose
```powershell
docker-compose up -d
```

### Run with custom model
```powershell
docker run -d --name cats-dogs-api -p 8000:8000 -e MODEL_PATH=models/improved/model_e15.pth -e MODEL_NAME=improved cats-dogs-classifier:latest
```

## Container Management

### View running containers
```powershell
docker ps
```

### View logs
```powershell
docker logs cats-dogs-api
docker logs -f cats-dogs-api  # Follow logs
```

### Stop container
```powershell
docker stop cats-dogs-api
```

### Remove container
```powershell
docker rm cats-dogs-api
```

### Restart container
```powershell
docker restart cats-dogs-api
```

## Docker Compose Commands

### Start services
```powershell
docker-compose up -d
```

### Stop services
```powershell
docker-compose down
```

### View logs
```powershell
docker-compose logs -f
```

### Rebuild and restart
```powershell
docker-compose up -d --build
```

## Testing the Container

### Health check
```powershell
curl http://localhost:8000/health
```

### Make prediction
```powershell
curl -X POST -F \"file=@path/to/image.jpg\" http://localhost:8000/predict
```

### Run test script
```powershell
python scripts/test_docker_api.py
```

## Image Information

### View images
```powershell
docker images
```

### Inspect image
```powershell
docker inspect cats-dogs-classifier:latest
```

### Image size
```powershell
docker images cats-dogs-classifier --format \"{{.Repository}}:{{.Tag}} - {{.Size}}\"
```

## Troubleshooting

### Container won't start
```powershell
# Check logs
docker logs cats-dogs-api

# Check if port is already in use
netstat -ano | findstr :8000
```

### Model not loading
```powershell
# Verify model file exists in image
docker run --rm cats-dogs-classifier:latest ls -la models/checkpoints/

# Check environment variables
docker inspect cats-dogs-api | Select-String \"Env\"
```

### Permission issues
```powershell
# Run as root (not recommended for production)
docker run -d --user root cats-dogs-classifier:latest
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| MODEL_PATH | models/checkpoints/best_model.pth | Path to model file |
| MODEL_NAME | improved | Model architecture (simple/improved) |
| PORT | 8000 | Port to run API on |

## Image Layers

The Dockerfile uses multi-stage build to minimize size:
1. **Builder stage**: Installs build dependencies and Python packages
2. **Final stage**: Copies only necessary files and runtime dependencies

## Health Check

The container includes a health check that:
- Runs every 30 seconds
- Calls the /health endpoint
- Retries 3 times before marking unhealthy
- Has a 40-second startup grace period

## Best Practices

1. **Use specific tags**: `cats-dogs-classifier:v1.0` instead of `:latest`
2. **Keep images small**: Multi-stage builds, .dockerignore
3. **Don't run as root**: Add non-root user in production
4. **Use environment variables**: For configuration
5. **Include health checks**: For orchestration systems
6. **Log to stdout/stderr**: For proper log collection

## Production Considerations

### Security
- Use non-root user
- Scan images for vulnerabilities
- Keep base images updated
- Don't include sensitive data in images

### Performance
- Use GPU-enabled base image if available
- Adjust worker count based on load
- Configure resource limits

### Monitoring
- Collect logs centrally
- Monitor health endpoint
- Track response times
- Set up alerts

## Next Steps

1. ✅ Docker image created
2. ✅ Container tested locally
3. 🚀 Push to container registry
4. ☸️ Deploy to Kubernetes

