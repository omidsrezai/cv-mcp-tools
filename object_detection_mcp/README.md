# Object Detection MCP Server

## Using the MCP Server with Claude Desktop

To use this **MCP server** with Claude Desktop on Mac, follow these steps:

1. Create the configuration file:
   ```bash
   code ~/Library/Application\ Support/Claude/claude_desktop_config.json
   ```

2. Copy and paste the following JSON into `claude_desktop_config.json`:

   ```json
   {
       "mcpServers": {
           "object_detection": {
               "command": "uv",
               "args": [
                   "--directory",
                   "/ABSOLUTE/PATH/TO/PARENT/FOLDER/object_detection_mcp",
                   "run",
                   "object_detection.py"
               ],
               "env": {
                   "YOLO_MODEL_NAME": "yolo11m.pt",
                   "YOLO_CONF_THRESHOLD": "0.45",
                   "MINIO_URL": "your-minio-server-url",
                   "MINIO_ACCESS_KEY": "your-access-key",
                   "MINIO_SECRET_KEY": "your-secret-key"
               }
           }
       }
   }
   ```

   **Note about environment variables:**
   - `YOLO_MODEL_NAME` specifies the YOLO model file (default: "yolo11m.pt")
   - `YOLO_CONF_THRESHOLD` sets detection confidence (0.45 default, higher for fewer false positives)
   - MinIO credentials are used to read source images and write annotated detection results

**Note**: For Windows installation or troubleshooting, check the [Model Context Protocol Quickstart Guide](https://modelcontextprotocol.io/quickstart/server)


## Using the MCP Server with Ollama

Instead of Claude Desktop, you can use [Ollama](https://github.com/ollama/ollama), but you also need to install [mcp-cli](https://github.com/chrishayuk/mcp-cli):

1. Install mcp-cli:
   ```bash
   git clone https://github.com/chrishayuk/mcp-cli
   cd mcp-cli
   ```

2. Copy and paste the following JSON into `server_config.json`:

   ```json
   {
       "mcpServers": {
           "object_detection": {
               "command": "uv",
               "args": [
                   "--directory",
                   "/ABSOLUTE/PATH/TO/PARENT/FOLDER/object_detection_mcp",
                   "run",
                   "object_detection.py"
               ],
            "env": {
                "YOLO_MODEL_NAME" : "yolo11m.pt",
                "YOLO_CONF_THRESHOLD": "0.45", 
                "MINIO_URL": "your-minio-server-url",
                "MINIO_ACCESS_KEY": "your-access-key",
                "MINIO_SECRET_KEY": "your-secret-key"
            }
           }
       }
   }
   ```