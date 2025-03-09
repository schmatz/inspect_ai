import json
import requests

# Query the tasks endpoint
response = requests.get("http://127.0.0.1:7474/tasks")
tasks = response.json()

print("Available tasks:")
print(json.dumps(tasks, indent=2))

# If there are tasks available, query the first one
if tasks:
    task_id = tasks[0]["task_id"]
    response = requests.get(f"http://127.0.0.1:7474/tasks/{task_id}")
    task_details = response.json()
    
    print(f"\nTask details for {task_id}:")
    print(json.dumps(task_details, indent=2))