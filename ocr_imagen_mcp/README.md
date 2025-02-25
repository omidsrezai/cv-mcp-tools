# Optical Character Recognition (ORC) and Image Generation MCP Server

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
                   "/ABSOLUTE/PATH/TO/PARENT/FOLDER/ocr_imagen_mcp",
                   "run",
                   "ocr_imagen.py"
               ],
               "env": {
                   "IMAGEN_API_ENDPOINT": "https://your-imagen-api.com",
                   "OCR_API_ENDPOINT": "https://your-ocr-api.com",
                   "MINIO_URL": "your-minio-server-url",
                   "MINIO_ACCESS_KEY": "your-access-key",
                   "MINIO_SECRET_KEY": "your-secret-key"
               }
           }
       }
   }
   ```

   **Note about environment variables:**
   - `IMAGEN_API_ENDPOINT` defines the endpoint for the [image generator Gradio server](../image_generator_server) 
   - `OCR_API_ENDPOINT` specifies the endpoint for the [OCR Gradio server](../ocr_server) 
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
                   "/ABSOLUTE/PATH/TO/PARENT/FOLDER/ocr_imagen_mcp",
                   "run",
                   "ocr_imagen.py"
               ],
               "env": {
                   "IMAGEN_API_ENDPOINT": "https://your-imagen-api.com",
                   "OCR_API_ENDPOINT": "https://your-ocr-api.com",
                   "MINIO_URL": "your-minio-server-url",
                   "MINIO_ACCESS_KEY": "your-access-key",
                   "MINIO_SECRET_KEY": "your-secret-key"
               }
           }
       }
   }
   ```