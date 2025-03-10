import json

from mcp.server.fastmcp import FastMCP

from inspect_ai.log import (
    list_eval_logs,
    read_eval_log_async,
    read_eval_log_sample_async,
)

# Initialize FastMCP server
mcp = FastMCP("weather")

# Constants
LOG_DIR = "~/inspect_ai/logs"


@mcp.tool()
async def list_logs() -> str:
    """List all the logs in the log directory."""
    logs = list_eval_logs(LOG_DIR)

    # Convert logs to dictionaries first
    logs_dicts = [log.model_dump() for log in logs]

    # Then serialize the entire list at once
    return json.dumps(logs_dicts, indent=2)


@mcp.tool()
async def read_eval_log_headers(log_file: str) -> str:
    """Read the headers of an eval log."""
    log = await read_eval_log_async(log_file, header_only=True)
    return log.model_dump_json(indent=2)


@mcp.tool()
async def read_eval_log_sample_messages(log_file: str, id: int, epoch: int) -> str:
    """Read a sample from an eval log."""
    sample = await read_eval_log_sample_async(log_file, id, epoch)
    return sample.model_dump_json(
        indent=2,
        include={"messages", "input", "target", "scores", "metadata", "total_time"},
    )


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport="stdio")
