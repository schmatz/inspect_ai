import asyncio
import json
import os
from typing import Any, Callable, Dict, List, Optional, Union

import aiohttp
from mcp.server.fastmcp import FastMCP

from inspect_ai._eval.eval import eval_async
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
        "tasks": eval_info["tasks"],
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
    eval_task = eval_info.get("eval_task")

    # Skip if no task or already marked as completed
    if not eval_task or eval_info.get("completed", False):
        return

    # Check if task is done and update status accordingly
    if eval_task.done():
        try:
            # Try to get result - this will raise any exception that happened
            log_files = eval_task.result()
            eval_info["logs"] = log_files
            eval_info["status"] = "completed"
            eval_info["completed"] = True
        except asyncio.CancelledError:
            eval_info["status"] = "cancelled"
            eval_info["completed"] = True
        except Exception as e:
            eval_info["status"] = "failed"
            eval_info["error"] = str(e)
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

    return start_http_eval(task, model)


def create_eval_complete_callback(eval_id: str) -> Callable[[asyncio.Task], None]:
    """
    Create a callback function for when an evaluation completes.

    Args:
        eval_id: ID of the evaluation

    Returns:
        Callback function
    """

    async def on_eval_complete(task: asyncio.Task) -> None:
        try:
            # Get the result (log files) if the task completed successfully
            log_files = await task
            active_evals[eval_id]["logs"] = log_files
            active_evals[eval_id]["status"] = "completed"
            active_evals[eval_id]["completed"] = True
        except Exception as e:
            active_evals[eval_id]["status"] = "failed"
            active_evals[eval_id]["error"] = str(e)
            active_evals[eval_id]["completed"] = True

    return lambda t: asyncio.create_task(on_eval_complete(t))


@mcp.tool()
async def start_http_eval(
    tasks: Union[str, List[str]],
    model: str,
    port: int = DEFAULT_HTTP_PORT,
    max_samples: Optional[int] = None,
    shuffle: bool = False,
) -> str:
    """
    Start an Inspect AI evaluation with HTTP display mode.

    Args:
        tasks: Task name or list of task names to run
        model: Model name to use for evaluation
        port: Port for HTTP server (default: 7474)
        max_samples: Maximum number of samples to evaluate (optional)
        shuffle: Whether to shuffle dataset samples (default: False)

    Returns:
        JSON string with evaluation ID and HTTP API URL
    """
    # Create a unique ID for this evaluation
    eval_id = f"eval_{len(active_evals) + 1}"

    # Store information about this evaluation
    eval_info = {
        "id": eval_id,
        "tasks": tasks,
        "model": model,
        "status": "starting",
        "port": port,
        "max_samples": max_samples,
        "shuffle": shuffle,
        "logs": [],
        "eval_task": None,  # Will store the task for eval_async
        "completed": False,  # Track if evaluation is completed
    }
    active_evals[eval_id] = eval_info

    # Create the evaluation task
    eval_task = asyncio.create_task(
        eval_async(
            tasks=tasks,
            model=model,
            display="http",
            max_samples=max_samples,
            shuffle=shuffle,
            http_port=port,
        )
    )

    # Store the task for later status checks
    active_evals[eval_id]["eval_task"] = eval_task
    active_evals[eval_id]["status"] = "running"

    # Add the callback to the task
    eval_task.add_done_callback(create_eval_complete_callback(eval_id))

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
    eval_task = eval_info.get("eval_task")

    # If task is still running, update status explicitly
    if eval_task and not eval_info.get("completed", False) and not eval_task.done():
        eval_info["status"] = "running"

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
        eval_task = eval_info.get("eval_task")

        # If task is still running, update status explicitly
        if eval_task and not eval_info.get("completed", False) and not eval_task.done():
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
                    return json.dumps({
                        "error": f"Failed to fetch metrics, HTTP status: {response.status}",
                        "message": await response.text()
                    }, indent=2)
                
                # Get tasks data
                tasks_data = await response.json()
                
                # Extract metrics, progress and other useful information
                result = {
                    "eval_id": eval_id,
                    "tasks": []
                }
                
                for task in tasks_data:
                    task_info = {
                        "task_id": task.get("task_id"),
                        "profile": task.get("profile", {}),
                        "progress": task.get("progress"),
                        "samples": {
                            "complete": task.get("samples_complete", 0),
                            "total": task.get("samples_total", 0)
                        },
                        "metrics": task.get("metrics", []),
                        "result": task.get("result")
                    }
                    result["tasks"].append(task_info)
                
                return json.dumps(result, indent=2)
                
    except aiohttp.ClientConnectionError:
        return json.dumps({
            "error": f"Failed to connect to HTTP API at port {port}",
            "message": "The HTTP server may not be running or accessible"
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "error": "Failed to fetch metrics",
            "message": str(e)
        }, indent=2)


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
            async with session.get(f"http://localhost:{port}/tasks/{task_id}") as response:
                if response.status == 404:
                    return json.dumps({
                        "error": f"Task with ID '{task_id}' not found in evaluation '{eval_id}'",
                    }, indent=2)
                elif response.status != 200:
                    return json.dumps({
                        "error": f"Failed to fetch task, HTTP status: {response.status}",
                        "message": await response.text()
                    }, indent=2)
                
                # Get task data
                task_data = await response.json()
                
                # Return the task data directly
                return json.dumps(task_data, indent=2)
                
    except aiohttp.ClientConnectionError:
        return json.dumps({
            "error": f"Failed to connect to HTTP API at port {port}",
            "message": "The HTTP server may not be running or accessible"
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "error": "Failed to fetch task details",
            "message": str(e)
        }, indent=2)


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport="stdio")
