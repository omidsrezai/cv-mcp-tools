# CV MCP Tools

A collection of Model Context Protocol (MCP) servers and services that integrate specialized computer vision capabilities with language models. This repository demonstrates how to build modular CV tools that can be easily composed and orchestrated through MCP.

## 🔧 Components

### MCP Servers
- **[Object Detection MCP](object_detection_mcp/)** - YOLO-based object detection with MinIO integration
- **[OCR + Image Generation MCP](ocr_imagen_mcp/)** - Combined OCR and image generation with iterative validation workflows

### Standalone Services  
- **[Image Generator Server](image_generator_server/)** - FLUX.1-schnell diffusion model service
- **[OCR Server](ocr_server/)** - Multi-model OCR service (Qwen-VL, Janus)

## 🚀 Quick Start

### Prerequisites
- Python 3.11+ 
- [UV package manager](https://github.com/astral-sh/uv)
- Docker with GPU support
- MinIO server (for MCP servers)

### Running MCP Servers

```bash
# Object Detection
cd object_detection_mcp
uv run object_detector.py

# OCR + Image Generation  
cd ocr_imagen_mcp
uv run ocr_imagen.py
```

### Running Standalone Services

```bash
# Image Generator
docker buildx build -t flux-schnell -f image_generator_server/Dockerfile .
docker run --gpus all -p 6070:6070 flux-schnell

# OCR Server
docker buildx build -t ocr-server -f ocr_server/Dockerfile .
docker run --gpus all -p 6080:6080 -p 6081:6081 ocr-server
```

## 🔗 Integration with Claude Desktop

Add to your Claude Desktop configuration:

```json
{
    "mcpServers": {
        "object_detection": {
            "command": "uv",
            "args": ["--directory", "/path/to/object_detection_mcp", "run", "object_detector.py"],
            "env": {
                "YOLO_MODEL_NAME": "yolo11m.pt",
                "YOLO_CONF_THRESHOLD": "0.45",
                "MINIO_URL": "localhost:9000",
                "MINIO_ACCESS_KEY": "your-key",
                "MINIO_SECRET_KEY": "your-secret"
            }
        }
    }
}
```

## 📁 Repository Structure

```
cv-mcp-tools/
├── object_detection_mcp/     # YOLO object detection MCP server
├── ocr_imagen_mcp/          # Combined OCR + image generation MCP
├── image_generator_server/   # Standalone FLUX image generation service
├── ocr_server/              # Standalone OCR service
└── CLAUDE.md                # Development guide for Claude Code
```

## 🎯 Use Cases

- **Automated Content Analysis** - Object detection and OCR for document processing
- **Iterative Image Generation** - Generate images with text validation loops
- **Multi-Modal Workflows** - Combine vision and language models for complex tasks
- **Modular CV Pipeline** - Mix and match components as needed

## 📖 Documentation

Each component has its own README with detailed setup instructions:
- [Object Detection MCP Setup](object_detection_mcp/README.md)
- [OCR + Image Generation MCP Setup](ocr_imagen_mcp/README.md) 
- [Image Generator Service Setup](image_generator_server/README.md)
- [OCR Service Setup](ocr_server/README.md)
