import asyncio
import json
import os
import subprocess
from typing import Any, Dict, Optional

import aiohttp
from mcp.server.fastmcp import FastMCP

from inspect_ai.log import (
    list_eval_logs,
    read_eval_log_async,
    read_eval_log_sample_async,
)

# Constants
LOG_DIR = os.path.expanduser("~/inspect_ai/logs")
DEFAULT_HTTP_PORT = 7474

# Initialize FastMCP server
mcp = FastMCP("inspect_ai")

# Store active evaluations
active_evals: Dict[str, Dict[str, Any]] = {}


def format_eval_info(
    eval_id: str, eval_info: Dict[str, Any], detailed: bool = True
) -> Dict[str, Any]:
    """
    Format evaluation info for JSON response.

    Args:
        eval_id: ID of the evaluation
        eval_info: The evaluation info dictionary
        detailed: Whether to include detailed information

    Returns:
        Formatted dictionary for JSON response
    """
    result = {
        "eval_id": eval_id,
        "task": eval_info["task"],
        "model": eval_info["model"],
        "status": eval_info["status"],
        "url": f"http://localhost:{eval_info['port']}",
    }

    if detailed:
        result.update(
            {
                "logs": eval_info.get("logs", []),
                "max_samples": eval_info.get("max_samples"),
                "shuffle": eval_info.get("shuffle"),
                "error": eval_info.get("error"),
                "progress": eval_info.get("progress", {}),
            }
        )
    else:
        result["logs"] = len(eval_info.get("logs", []))

    return result


async def update_eval_task_status(eval_id: str) -> None:
    """
    Update the status of an evaluation task.

    Args:
        eval_id: ID of the evaluation to update
    """
    if eval_id not in active_evals:
        return

    eval_info = active_evals[eval_id]

    # Skip if already marked as completed
    if eval_info.get("completed", False):
        return

    # Check if the process is still running
    process = eval_info.get("process")
    if process:
        returncode = process.returncode
        if returncode is not None:  # Process has terminated
            if returncode == 0:
                # Successfully completed
                eval_info["status"] = "completed"
                eval_info["completed"] = True
            else:
                # Failed
                eval_info["status"] = "failed"
                eval_info["error"] = f"Process exited with code {returncode}"
                eval_info["completed"] = True


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


@mcp.tool()
async def run_example_eval() -> str:
    task = "./examples/hello_world.py"
    model = "anthropic/claude-3-5-haiku-latest"

    return await start_http_eval(task, model)


async def monitor_process(process: subprocess.Popen, eval_id: str) -> None:
    """
    Monitor a subprocess until it completes.

    Args:
        process: The subprocess to monitor
        eval_id: The ID of the evaluation
    """
    try:
        await asyncio.to_thread(process.wait)

        # Update status when process is complete
        if eval_id in active_evals:
            if process.returncode == 0:
                active_evals[eval_id]["status"] = "completed"
            else:
                active_evals[eval_id]["status"] = "failed"
                active_evals[eval_id]["error"] = (
                    f"Process exited with code {process.returncode}"
                )

            active_evals[eval_id]["completed"] = True
    except Exception as e:
        if eval_id in active_evals:
            active_evals[eval_id]["status"] = "failed"
            active_evals[eval_id]["error"] = str(e)
            active_evals[eval_id]["completed"] = True


@mcp.tool()
async def start_http_eval(
    task: str,
    model: str,
    port: int = DEFAULT_HTTP_PORT,
    max_samples: Optional[int] = None,
    shuffle: bool = False,
) -> str:
    """
    Start an Inspect AI evaluation with HTTP display mode.

    Args:
        task: Task name
        model: Model name to use for evaluation
        port: Port for HTTP server (default: 7474)
        max_samples: Maximum number of samples to evaluate (optional)
        shuffle: Whether to shuffle dataset samples (default: False)

    Returns:
        JSON string with evaluation ID and HTTP API URL
    """
    # Create a unique ID for this evaluation
    eval_id = f"eval_{len(active_evals) + 1}"

    # Build the command with appropriate arguments
    cmd = [
        "inspect",
        "eval",
        task,
        "--model",
        model,
        "--display",
        "http",
    ]

    # Add optional arguments
    if max_samples is not None:
        cmd.extend(["--max-samples", str(max_samples)])

    if shuffle:
        cmd.append("--shuffle")

    # Add task at the end
    cmd.append(task)

    # Start the process
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # Line buffered
    )

    # Store information about this evaluation
    eval_info = {
        "id": eval_id,
        "task": task,
        "model": model,
        "status": "running",
        "port": port,
        "max_samples": max_samples,
        "shuffle": shuffle,
        "logs": [],
        "process": process,
        "completed": False,
    }
    active_evals[eval_id] = eval_info

    # Start monitoring thread
    asyncio.create_task(monitor_process(process, eval_id))

    # Return information about the started evaluation
    return json.dumps(
        format_eval_info(eval_id, active_evals[eval_id], detailed=False), indent=2
    )


@mcp.tool()
async def get_eval_status(eval_id: str) -> str:
    """
    Get the status of a running or completed evaluation.

    Args:
        eval_id: ID of the evaluation to check

    Returns:
        JSON string with evaluation status and details
    """
    if eval_id not in active_evals:
        return json.dumps(
            {"error": f"Evaluation with ID '{eval_id}' not found"}, indent=2
        )

    # Update the status of the task
    await update_eval_task_status(eval_id)

    eval_info = active_evals[eval_id]

    # Update process status if it's still running
    process = eval_info.get("process")
    if process and not eval_info.get("completed", False) and process.returncode is None:
        eval_info["status"] = "running"

        # Try to read some output to include in the status
        if hasattr(process, "stdout") and process.stdout:
            try:
                # Read up to 10 lines of output without blocking
                output_lines = []
                for _ in range(10):
                    line = process.stdout.readline()
                    if not line:
                        break
                    output_lines.append(line.strip())

                if output_lines:
                    eval_info["latest_output"] = output_lines
            except Exception:
                pass

    # Return information about the evaluation
    return json.dumps(format_eval_info(eval_id, eval_info, detailed=True), indent=2)


@mcp.tool()
async def list_active_evals() -> str:
    """
    List all active and completed evaluations.

    Returns:
        JSON string with list of evaluations and their statuses
    """
    evals_list = []

    # Update all evaluation statuses first
    for eval_id in active_evals:
        await update_eval_task_status(eval_id)

    # Then build the list of evaluations
    for eval_id, eval_info in active_evals.items():
        process = eval_info.get("process")

        # If process is still running, update status explicitly
        if (
            process
            and not eval_info.get("completed", False)
            and process.returncode is None
        ):
            eval_info["status"] = "running"

        # Add evaluation info to the list
        evals_list.append(format_eval_info(eval_id, eval_info, detailed=False))

    return json.dumps(evals_list, indent=2)


@mcp.tool()
async def get_http_metrics(eval_id: str) -> str:
    """
    Get metrics from a running HTTP display mode evaluation.

    Args:
        eval_id: ID of the evaluation to get metrics for

    Returns:
        JSON string with metrics data from the HTTP API
    """
    if eval_id not in active_evals:
        return json.dumps(
            {"error": f"Evaluation with ID '{eval_id}' not found"}, indent=2
        )

    eval_info = active_evals[eval_id]
    port = eval_info.get("port", DEFAULT_HTTP_PORT)

    try:
        async with aiohttp.ClientSession() as session:
            # Fetch tasks from the HTTP API
            async with session.get(f"http://localhost:{port}/tasks") as response:
                if response.status != 200:
                    return json.dumps(
                        {
                            "error": f"Failed to fetch metrics, HTTP status: {response.status}",
                            "message": await response.text(),
                        },
                        indent=2,
                    )

                # Get tasks data
                tasks_data = await response.json()

                # Extract metrics, progress and other useful information
                result = {"eval_id": eval_id, "tasks": []}

                for task in tasks_data:
                    task_info = {
                        "task_id": task.get("task_id"),
                        "profile": task.get("profile", {}),
                        "progress": task.get("progress"),
                        "samples": {
                            "complete": task.get("samples_complete", 0),
                            "total": task.get("samples_total", 0),
                        },
                        "metrics": task.get("metrics", []),
                        "result": task.get("result"),
                    }
                    result["tasks"].append(task_info)

                return json.dumps(result, indent=2)

    except aiohttp.ClientConnectionError:
        return json.dumps(
            {
                "error": f"Failed to connect to HTTP API at port {port}",
                "message": "The HTTP server may not be running or accessible",
            },
            indent=2,
        )
    except Exception as e:
        return json.dumps(
            {"error": "Failed to fetch metrics", "message": str(e)}, indent=2
        )


@mcp.tool()
async def get_http_task(eval_id: str, task_id: str) -> str:
    """
    Get detailed information about a specific task in an HTTP display evaluation.

    Args:
        eval_id: ID of the evaluation
        task_id: ID of the task within the evaluation

    Returns:
        JSON string with detailed task data from the HTTP API
    """
    if eval_id not in active_evals:
        return json.dumps(
            {"error": f"Evaluation with ID '{eval_id}' not found"}, indent=2
        )

    eval_info = active_evals[eval_id]
    port = eval_info.get("port", DEFAULT_HTTP_PORT)

    try:
        async with aiohttp.ClientSession() as session:
            # Fetch the specific task from the HTTP API
            async with session.get(
                f"http://localhost:{port}/tasks/{task_id}"
            ) as response:
                if response.status == 404:
                    return json.dumps(
                        {
                            "error": f"Task with ID '{task_id}' not found in evaluation '{eval_id}'",
                        },
                        indent=2,
                    )
                elif response.status != 200:
                    return json.dumps(
                        {
                            "error": f"Failed to fetch task, HTTP status: {response.status}",
                            "message": await response.text(),
                        },
                        indent=2,
                    )

                # Get task data
                task_data = await response.json()

                # Return the task data directly
                return json.dumps(task_data, indent=2)

    except aiohttp.ClientConnectionError:
        return json.dumps(
            {
                "error": f"Failed to connect to HTTP API at port {port}",
                "message": "The HTTP server may not be running or accessible",
            },
            indent=2,
        )
    except Exception as e:
        return json.dumps(
            {"error": "Failed to fetch task details", "message": str(e)}, indent=2
        )


@mcp.tool()
async def terminate_eval(eval_id: str) -> str:
    """
    Terminate a running evaluation.

    Args:
        eval_id: ID of the evaluation to terminate

    Returns:
        JSON string with result of the termination attempt
    """
    if eval_id not in active_evals:
        return json.dumps(
            {"error": f"Evaluation with ID '{eval_id}' not found"}, indent=2
        )

    eval_info = active_evals[eval_id]
    process = eval_info.get("process")

    if not process:
        return json.dumps({"error": "No process found for this evaluation"}, indent=2)

    if process.returncode is not None:
        return json.dumps(
            {
                "message": f"Process already finished with return code {process.returncode}"
            },
            indent=2,
        )

    try:
        # Try graceful termination first
        process.terminate()

        # Update status
        eval_info["status"] = "terminating"

        return json.dumps(
            {"message": f"Termination signal sent to evaluation '{eval_id}'"}, indent=2
        )
    except Exception as e:
        return json.dumps(
            {"error": f"Failed to terminate evaluation '{eval_id}'", "message": str(e)},
            indent=2,
        )


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport="stdio")
