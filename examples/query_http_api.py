#!/usr/bin/env python
"""
Sample script to query the Inspect AI HTTP API.

This script demonstrates how to interact with the HTTP display mode API.
It polls the API to monitor task progress and results in real-time.

Usage:
    python query_http_api.py [--url http://127.0.0.1:7474] [--interval 2]

Options:
    --url       Base URL of the HTTP API (default: http://127.0.0.1:7474)
    --interval  Polling interval in seconds (default: 2)
"""

import argparse
import json
import time
import sys
from typing import Dict, Any, List, Optional

import requests


def format_progress(data: Dict[str, Any]) -> str:
    """Format progress information for display."""
    samples = data.get("samples_complete", 0)
    total_samples = data.get("samples_total", 0)
    
    progress = data.get("progress", {})
    current_step = progress.get("current", 0) if progress else 0
    total_steps = progress.get("total", 0) if progress else 0
    
    progress_bar = ""
    if total_samples > 0:
        percent = samples / total_samples
        width = 20
        filled = int(width * percent)
        progress_bar = f"[{'#' * filled}{'-' * (width - filled)}] {percent:.1%}"
    
    return (
        f"Progress: {samples}/{total_samples} samples, "
        f"Step: {current_step}/{total_steps}\n"
        f"{progress_bar}"
    )


def format_metrics(metrics: List[Dict[str, Any]]) -> str:
    """Format metrics information for display."""
    if not metrics:
        return "No metrics available yet."
    
    lines = ["Current metrics:"]
    for metric in metrics:
        scorer = metric.get("scorer", "unknown")
        name = metric.get("name", "value")
        value = metric.get("value", 0)
        reducer = metric.get("reducer", "")
        
        # Format value with appropriate precision
        if isinstance(value, float):
            value_str = f"{value:.4f}"
        else:
            value_str = str(value)
            
        lines.append(f"  {scorer}.{name}: {value_str} ({reducer})")
    
    return "\n".join(lines)


def format_result(result: Dict[str, Any]) -> str:
    """Format task result information for display."""
    if not result:
        return ""
    
    result_type = result.get("type", "Unknown")
    
    lines = [f"Task complete! Result type: {result_type}"]
    
    # Add error information if present
    if "error" in result:
        error_type = result["error"].get("type", "Unknown")
        error_message = result["error"].get("message", "")
        lines.append(f"Error: {error_type}: {error_message}")
    
    # Add statistics if present
    if "stats" in result:
        stats = result["stats"]
        lines.append("Statistics:")
        
        if "time_elapsed" in stats:
            lines.append(f"  Time elapsed: {stats['time_elapsed']:.2f} seconds")
        
        if "time_per_sample" in stats:
            lines.append(f"  Time per sample: {stats['time_per_sample']:.2f} seconds")
        
        if "tokens_per_second" in stats:
            lines.append(f"  Tokens per second: {stats['tokens_per_second']:.2f}")
        
        if "tokens_per_sample" in stats:
            lines.append(f"  Tokens per sample: {stats['tokens_per_sample']:.2f}")
        
        if "total_tokens" in stats:
            lines.append(f"  Total tokens: {stats['total_tokens']}")
    
    if "samples_completed" in result:
        lines.append(f"Samples completed: {result['samples_completed']}")
    
    return "\n".join(lines)


def monitor_task(base_url: str, task_id: str, interval: int) -> None:
    """Monitor a specific task continuously."""
    print(f"Monitoring task {task_id}...")
    
    while True:
        try:
            response = requests.get(f"{base_url}/tasks/{task_id}")
            if response.status_code != 200:
                print(f"Error: Received status code {response.status_code}")
                print(response.text)
                time.sleep(interval)
                continue
                
            task_data = response.json()
            
            # Clear screen (works on most terminals)
            print("\033c", end="")
            
            # Print task info
            print(f"Task: {task_data['profile']['name']}")
            print(f"Model: {task_data['profile']['model']}")
            print()
            
            # Print progress
            print(format_progress(task_data))
            print()
            
            # Print metrics
            metrics = task_data.get("metrics", [])
            print(format_metrics(metrics))
            print()
            
            # Check if task has a result (is complete)
            result = task_data.get("result")
            if result:
                print(format_result(result))
                return
                
            time.sleep(interval)
            
        except (requests.RequestException, KeyboardInterrupt) as e:
            if isinstance(e, KeyboardInterrupt):
                print("\nMonitoring stopped by user.")
                return
            else:
                print(f"Error connecting to API: {e}")
                time.sleep(interval)


def list_tasks(base_url: str) -> Optional[str]:
    """List all available tasks and return the selected task ID."""
    try:
        response = requests.get(f"{base_url}/tasks")
        if response.status_code != 200:
            print(f"Error: Received status code {response.status_code}")
            print(response.text)
            return None
            
        tasks = response.json()
        
        if not tasks:
            print("No tasks found. Make sure the Inspect AI evaluation is running.")
            return None
        
        if len(tasks) == 1:
            # If only one task, select it automatically
            task_id = tasks[0]["task_id"]
            print(f"Found one task: {task_id}")
            return task_id
        
        # Multiple tasks, let user choose
        print("Available tasks:")
        for i, task in enumerate(tasks):
            task_id = task["task_id"]
            name = task["profile"]["name"]
            print(f"{i+1}. {name} (ID: {task_id})")
        
        while True:
            try:
                choice = input("Select a task number to monitor (or q to quit): ")
                if choice.lower() == 'q':
                    return None
                    
                idx = int(choice) - 1
                if 0 <= idx < len(tasks):
                    return tasks[idx]["task_id"]
                else:
                    print(f"Please enter a number between 1 and {len(tasks)}")
            except ValueError:
                print("Please enter a valid number")
    
    except requests.RequestException as e:
        print(f"Error connecting to API: {e}")
        return None


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Query Inspect AI HTTP API')
    parser.add_argument('--url', default='http://127.0.0.1:7474', 
                        help='Base URL of the HTTP API')
    parser.add_argument('--interval', type=int, default=2,
                        help='Polling interval in seconds')
    parser.add_argument('--task', help='Specific task ID to monitor (optional)')
    args = parser.parse_args()
    
    print(f"Connecting to Inspect AI HTTP API at {args.url}")
    
    # Check if the API is accessible
    try:
        response = requests.get(args.url)
        if response.status_code != 200:
            print(f"Error: Received status code {response.status_code}")
            print(response.text)
            return
    except requests.RequestException as e:
        print(f"Error connecting to API: {e}")
        print("Make sure Inspect AI is running with the HTTP display mode.")
        return
    
    # Get task to monitor
    task_id = args.task or list_tasks(args.url)
    if task_id:
        monitor_task(args.url, task_id, args.interval)


if __name__ == "__main__":
    main()