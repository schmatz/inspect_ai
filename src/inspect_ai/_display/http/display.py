import contextlib
import json
import threading
from dataclasses import asdict, dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, Iterator, List, Optional

import aiohttp
import anyio
from aiohttp import web

from inspect_ai._display.core.rich import rich_initialise
from inspect_ai._util._async import configured_async_backend
from inspect_ai._util.constants import DEFAULT_SERVER_HOST

from ..core.display import (
    TR,
    Display,
    Progress,
    TaskDisplay,
    TaskDisplayMetric,
    TaskProfile,
    TaskResult,
    TaskScreen,
    TaskSpec,
    TaskWithResult,
)


class HttpProgress(Progress):
    """Simple progress tracker for HTTP display mode."""
    
    def __init__(self, total: int):
        self.total = total
        self.current = 0

    def update(self, n: int = 1) -> None:
        self.current += n
        
        # If this progress tracker is associated with a task, update the task state
        if hasattr(self, 'task_id') and hasattr(self, 'state_dict') and self.task_id in self.state_dict:
            state = self.state_dict[self.task_id]
            if state.progress:
                state.progress["current"] = self.current

    def complete(self) -> None:
        self.current = self.total
        
        # If this progress tracker is associated with a task, update the task state
        if hasattr(self, 'task_id') and hasattr(self, 'state_dict') and self.task_id in self.state_dict:
            state = self.state_dict[self.task_id]
            if state.progress:
                state.progress["current"] = self.current


@dataclass
class TaskState:
    """Task state that can be serialized to JSON."""
    
    task_id: str
    profile: Dict[str, Any]
    progress: Optional[Dict[str, int]] = None
    samples_complete: int = 0
    samples_total: int = 0
    metrics: Optional[List[Dict[str, Any]]] = None
    result: Optional[Dict[str, Any]] = None


class HttpTaskDisplay(TaskDisplay):
    """Task display implementation for HTTP server mode."""
    
    def __init__(self, task: TaskWithResult, task_id: str, state_dict: Dict[str, TaskState]):
        self.task = task
        self.task_id = task_id
        self.state_dict = state_dict
        self.progress_tracker: Optional[HttpProgress] = None
        self.samples_complete = 0
        self.samples_total = 0
        self.current_metrics: Optional[List[TaskDisplayMetric]] = None
        
        # Initialize state in the shared dictionary
        profile_dict = {
            "name": task.profile.name,
            "file": task.profile.file,
            "model": str(task.profile.model),
            "dataset": task.profile.dataset,
            "scorer": task.profile.scorer,
            "samples": task.profile.samples,
            "steps": task.profile.steps,
            "log_location": task.profile.log_location,
        }
        
        self.state_dict[self.task_id] = TaskState(
            task_id=self.task_id,
            profile=profile_dict,
        )

    @contextlib.contextmanager
    def progress(self) -> Iterator[Progress]:
        self.progress_tracker = HttpProgress(self.task.profile.steps)
        
        # Associate the progress tracker with this task for state updates
        self.progress_tracker.task_id = self.task_id
        self.progress_tracker.state_dict = self.state_dict
        
        # Update the state
        state = self.state_dict[self.task_id]
        state.progress = {"total": self.progress_tracker.total, "current": 0}
        
        yield self.progress_tracker

    def sample_complete(self, complete: int, total: int) -> None:
        self.samples_complete = complete
        self.samples_total = total
        
        # Update the state
        state = self.state_dict[self.task_id]
        state.samples_complete = complete
        state.samples_total = total
        
        if self.progress_tracker:
            state.progress = {
                "total": self.progress_tracker.total, 
                "current": self.progress_tracker.current
            }

    def update_metrics(self, metrics: List[TaskDisplayMetric]) -> None:
        self.current_metrics = metrics
        
        # Update the state
        state = self.state_dict[self.task_id]
        metrics_list = []
        for metric in metrics:
            metrics_list.append({
                "scorer": metric.scorer,
                "name": metric.name,
                "value": metric.value,
                "reducer": metric.reducer,
            })
        state.metrics = metrics_list

    def complete(self, result: TaskResult) -> None:
        self.task.result = result
        
        # Update the state with result information
        state = self.state_dict[self.task_id]
        
        # Convert the result to a dict based on its type
        result_dict: Dict[str, Any] = {"type": result.__class__.__name__}
        
        if hasattr(result, "samples_completed"):
            result_dict["samples_completed"] = result.samples_completed
            
        if hasattr(result, "stats"):
            stats_dict = {}
            stats = result.stats
            
            # Only include attributes that exist on the stats object
            if hasattr(stats, "time_elapsed"):
                stats_dict["time_elapsed"] = stats.time_elapsed
            if hasattr(stats, "time_per_sample"):
                stats_dict["time_per_sample"] = stats.time_per_sample
            if hasattr(stats, "tokens_per_second"):
                stats_dict["tokens_per_second"] = stats.tokens_per_second
            if hasattr(stats, "tokens_per_sample"):
                stats_dict["tokens_per_sample"] = stats.tokens_per_sample
            if hasattr(stats, "total_tokens"):
                stats_dict["total_tokens"] = stats.total_tokens
                
            result_dict["stats"] = stats_dict
            
        # For errors, include error info
        if hasattr(result, "exc_type") and hasattr(result, "exc_value"):
            result_dict["error"] = {
                "type": result.exc_type.__name__,
                "message": str(result.exc_value),
            }
            
        state.result = result_dict


class HttpDisplay(Display):
    """HTTP server display implementation using aiohttp."""
    
    def __init__(self) -> None:
        self.parallel = False
        self.task_states: Dict[str, TaskState] = {}
        self.task_counter = 0
        self.server_thread: Optional[threading.Thread] = None
        self.app = web.Application()
        self.host = DEFAULT_SERVER_HOST
        self.port = 7474  # Use fixed port 7474
        self.server_running = False
        self.setup_routes()
        rich_initialise()

    def setup_routes(self) -> None:
        """Setup the aiohttp application routes."""
        self.app.add_routes([
            web.get('/tasks', self.get_tasks),
            web.get('/tasks/{task_id}', self.get_task),
            web.get('/', self.get_root)
        ])
        
        # Add CORS middleware
        @web.middleware
        async def cors_middleware(request, handler):
            response = await handler(request)
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
            return response
            
        self.app.middlewares.append(cors_middleware)

    async def get_root(self, request: web.Request) -> web.Response:
        """Root endpoint that returns available endpoints."""
        return web.json_response({
            "endpoints": [
                {"path": "/tasks", "method": "GET", "description": "List all tasks"},
                {"path": "/tasks/{task_id}", "method": "GET", "description": "Get a specific task"}
            ]
        })

    async def get_tasks(self, request: web.Request) -> web.Response:
        """Get all tasks."""
        tasks_list = []
        for task_state in self.task_states.values():
            tasks_list.append(asdict(task_state))
        return web.json_response(tasks_list)

    async def get_task(self, request: web.Request) -> web.Response:
        """Get a specific task by ID."""
        task_id = request.match_info.get('task_id')
        if task_id not in self.task_states:
            return web.json_response(
                {"message": f"Task with ID {task_id} not found"},
                status=404
            )
        return web.json_response(asdict(self.task_states[task_id]))

    def start_server(self) -> None:
        """Start the HTTP server in a background thread."""
        if self.server_running:
            return
            
        async def run_app():
            runner = web.AppRunner(self.app)
            await runner.setup()
            site = web.TCPSite(runner, host=self.host, port=self.port)
            await site.start()
            
            # Keep the site running
            while True:
                await anyio.sleep(3600)  # Sleep for an hour or until process ends
            
        def run_server() -> None:
            anyio.run(run_app)
            
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        self.server_running = True
        print(f"HTTP API server running at http://{self.host}:{self.port}")

    def print(self, message: str) -> None:
        print(message)

    @contextlib.contextmanager
    def progress(self, total: int) -> Iterator[Progress]:
        yield HttpProgress(total)

    def run_task_app(self, main: Callable[[], Awaitable[TR]]) -> TR:
        return anyio.run(main, backend=configured_async_backend())

    @contextlib.contextmanager
    def suspend_task_app(self) -> Iterator[None]:
        yield

    @contextlib.asynccontextmanager
    async def task_screen(
        self, tasks: List[TaskSpec], parallel: bool
    ) -> AsyncIterator[TaskScreen]:
        self.parallel = parallel
        self.task_states = {}  # Reset task states
        
        # Start the HTTP server
        self.start_server()
        
        try:
            print(f"Running {len(tasks)} tasks...")
            yield TaskScreen()
        finally:
            # Final task states remain available via the HTTP API
            pass

    @contextlib.contextmanager
    def task(self, profile: TaskProfile) -> Iterator[TaskDisplay]:
        # Generate a unique ID for this task
        task_id = f"task_{self.task_counter}"
        self.task_counter += 1
        
        # Create and yield task display
        task = TaskWithResult(profile, None)
        yield HttpTaskDisplay(task, task_id, self.task_states)

    def display_counter(self, caption: str, value: str) -> None:
        # Not implemented for HTTP display
        pass