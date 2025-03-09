"""
Example of using the HTTP display mode for Inspect AI.

This creates a simple task that runs with the HTTP display mode,
making task progress and results available via an HTTP API.

To run this example:
    inspect eval http_display_example.py --model anthropic/claude-3-haiku-20240307 --display http

Then in another terminal, you can query the API:
    curl http://127.0.0.1:7474/tasks

Or to get a specific task:
    curl http://127.0.0.1:7474/tasks/{task_id}
    
You can also view API documentation:
    curl http://127.0.0.1:7474/

## HTTP API Endpoints

The HTTP display mode exposes the following endpoints:

1. `GET /` - API documentation
   - Lists all available endpoints with descriptions

2. `GET /tasks` - List all tasks
   - Returns an array of all active tasks with their current state
   - Example response: `[{"task_id": "task_0", "profile": {"name": "http_display_demo", ...}, ...}]`

3. `GET /tasks/{task_id}` - Get details for a specific task
   - Returns detailed information about a single task
   - Includes task profile, progress, samples completed, and metrics
   - Example response:
     ```json
     {
       "task_id": "task_0",
       "profile": {
         "name": "http_display_demo",
         "file": "http_display_example.py",
         "model": "anthropic/claude-3-haiku-20240307",
         "dataset": "...",
         "scorer": "exact",
         "samples": 100,
         "steps": 5
       },
       "progress": {"current": 3, "total": 5},
       "samples_complete": 60,
       "samples_total": 100,
       "metrics": [
         {"scorer": "exact", "name": "score", "value": 0.75, "reducer": "mean"}
       ]
     }
     ```

## Sample Query Script

You can use this Python script to query the HTTP API:

```python
import requests
import json
import time

def poll_http_api():
    base_url = "http://127.0.0.1:7474"
    
    # Get list of tasks
    tasks_response = requests.get(f"{base_url}/tasks")
    tasks = tasks_response.json()
    
    if not tasks:
        print("No tasks found.")
        return
    
    # Get the first task
    task_id = tasks[0]["task_id"]
    task_response = requests.get(f"{base_url}/tasks/{task_id}")
    task_data = task_response.json()
    
    # Pretty print the task data
    print(json.dumps(task_data, indent=2))
    
    # Monitor progress
    while True:
        time.sleep(2)
        task_response = requests.get(f"{base_url}/tasks/{task_id}")
        task_data = task_response.json()
        
        samples = task_data.get("samples_complete", 0)
        total = task_data.get("samples_total", 0)
        progress = task_data.get("progress", {})
        metrics = task_data.get("metrics", [])
        
        print(f"Progress: {samples}/{total} samples, ", end="")
        if progress:
            print(f"Step: {progress.get('current', 0)}/{progress.get('total', 0)}")
        
        if metrics:
            print("Current metrics:")
            for metric in metrics:
                print(f"  {metric['scorer']}.{metric['name']}: {metric['value']}")
        
        # Check if task has a result (is complete)
        if task_data.get("result"):
            print("Task complete!")
            print(json.dumps(task_data["result"], indent=2))
            break
```

You can save this script as `query_http_api.py` and run it in a separate terminal
while this example is running.
"""

import time
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import match, exact
from inspect_ai.solver import generate


# Create a larger dataset for better demonstration
def _dataset(num_samples=50):
    base_samples = [
        Sample(input="What is 2+2?", target="4"),
        Sample(input="What is the capital of France?", target="Paris"),
        Sample(input="Who wrote Romeo and Juliet?", target="William Shakespeare"),
        Sample(input="What is the largest planet in our solar system?", target="Jupiter"),
        Sample(input="What is the chemical symbol for gold?", target="Au"),
    ]
    
    # Duplicate samples to create a larger dataset for demonstration
    samples = []
    for i in range(num_samples):
        sample_idx = i % len(base_samples)
        sample = base_samples[sample_idx]
        # Add a slight variation to make them unique
        samples.append(Sample(
            input=f"Q{i+1}: {sample.input}",
            target=sample.target
        ))
    
    return samples


# Define a simple match scorer for reliability
def _get_match_scorer():
    return match()


@task
def http_display_demo():
    """
    A comprehensive task demonstrating the HTTP display mode.
    
    This task uses a larger dataset to better demonstrate the
    real-time progress tracking through the HTTP API.
    
    Run with:
        inspect eval http_display_example.py --model anthropic/claude-3-haiku-20240307 --display http
    """
    return Task(
        dataset=_dataset(num_samples=50),
        solver=[
            generate(), 
        ],
        scorer=_get_match_scorer(),
        epochs=3,  # Run multiple epochs to better demonstrate progress
    )


@task
def multi_task_demo():
    """
    A second task that can be run in the same evaluation.
    
    This demonstrates how multiple tasks can be tracked simultaneously
    through the HTTP API.
    """
    return Task(
        dataset=_dataset(num_samples=25),
        solver=[generate()],
        scorer=match(),
        epochs=2,
    )