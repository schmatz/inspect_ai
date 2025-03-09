import socket
import threading
from unittest.mock import patch, MagicMock

import pytest
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from inspect_ai._display.http.display import HttpDisplay, HttpProgress, TaskState, HttpTaskDisplay
from inspect_ai._display.core.display import TaskSpec, TaskProfile, TaskResult, TaskDisplayMetric, EvalStats
from inspect_ai.model import ModelName


def get_free_port():
    """Get a free port to use for testing."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def session():
    """Create a requests session with retry capability."""
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=0.1,
        status_forcelist=[500, 502, 503, 504],
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    yield session
    session.close()


@pytest.fixture
def http_display(session):
    """Create an HttpDisplay instance with a dynamic port."""
    original_init = HttpDisplay.__init__
    
    def patched_init(self):
        original_init(self)
        self.port = get_free_port()
        
    with patch.object(HttpDisplay, '__init__', patched_init):
        display = HttpDisplay()
        display.start_server()
        
        # Wait for server to be available
        base_url = f"http://{display.host}:{display.port}"
        max_retries = 10
        retry_count = 0
        retry_delay = 0.1
        
        while retry_count < max_retries:
            try:
                response = session.get(base_url, timeout=0.5)
                if response.status_code == 200:
                    break
            except (requests.RequestException, ConnectionError):
                retry_count += 1
                if retry_count < max_retries:
                    import time
                    time.sleep(retry_delay)
                    # Increase delay slightly with each retry
                    retry_delay *= 1.2
        
        assert retry_count < max_retries, f"Server failed to start after {max_retries} attempts"
        yield display


@pytest.fixture
def task_profile():
    """Create a standard task profile for testing."""
    return TaskProfile(
        name="test_task",
        file="test.py",
        model=ModelName("anthropic/claude-3-5-haiku-latest"),
        dataset="test_dataset",
        scorer="test_scorer",
        samples=10,
        steps=5,
        eval_config={},
        task_args={},
        generate_config={},
        tags=["test"],
        log_location="test.log"
    )


@pytest.mark.asyncio
async def test_http_display_basics(http_display, session):
    """Test basic HttpDisplay functionality including initialization and server start."""
    # Test server is running
    assert http_display.server_running
    assert http_display.server_thread is not None
    assert http_display.server_thread.is_alive()
    
    # Test connection and API endpoints
    response = session.get(f"http://{http_display.host}:{http_display.port}/tasks")
    assert response.status_code == 200
    assert response.json() == []
    
    # Test API documentation endpoint
    response = session.get(f"http://{http_display.host}:{http_display.port}/")
    assert response.status_code == 200
    assert "endpoints" in response.json()
    assert any(e["path"] == "/tasks" for e in response.json()["endpoints"])
    assert any(e["path"] == "/tasks/{task_id}" for e in response.json()["endpoints"])
    
    # Test idempotence of start_server
    thread_id = id(http_display.server_thread)
    http_display.start_server()  # Should not create another thread
    assert http_display.server_thread.is_alive()
    assert id(http_display.server_thread) == thread_id  # Same thread object


@pytest.mark.asyncio
async def test_http_display_task_lifecycle(http_display, session, task_profile):
    """Test the complete lifecycle of a task from creation to completion."""
    # Reset task states for this test
    http_display.task_states = {}
    http_display.task_counter = 0
    
    with http_display.task(task_profile) as task_display:
        # Verify task was created
        assert len(http_display.task_states) == 1
        task_id = list(http_display.task_states.keys())[0]
        
        # Test task API
        response = session.get(f"http://{http_display.host}:{http_display.port}/tasks/{task_id}")
        assert response.status_code == 200
        assert response.json()["task_id"] == task_id
        assert response.json()["profile"]["name"] == "test_task"
        
        # Test progress updates
        with task_display.progress() as progress:
            progress.update(2)
            
            response = session.get(f"http://{http_display.host}:{http_display.port}/tasks/{task_id}")
            assert response.json()["progress"]["current"] == 2
            assert response.json()["progress"]["total"] == 5
            
            # Test sample updates
            task_display.sample_complete(3, 10)
            
            response = session.get(f"http://{http_display.host}:{http_display.port}/tasks/{task_id}")
            assert response.json()["samples_complete"] == 3
            assert response.json()["samples_total"] == 10
            
            # Test metrics updates
            metrics = [
                TaskDisplayMetric(scorer="accuracy", name="score", value=0.75, reducer="mean"),
                TaskDisplayMetric(scorer="f1", name="score", value=0.82, reducer="mean")
            ]
            task_display.update_metrics(metrics)
            
            response = session.get(f"http://{http_display.host}:{http_display.port}/tasks/{task_id}")
            assert len(response.json()["metrics"]) == 2
            assert response.json()["metrics"][0]["scorer"] == "accuracy"
            assert response.json()["metrics"][0]["value"] == 0.75
            assert response.json()["metrics"][1]["scorer"] == "f1"
            assert response.json()["metrics"][1]["value"] == 0.82
            
            # Test progress completion
            progress.complete()
            
            response = session.get(f"http://{http_display.host}:{http_display.port}/tasks/{task_id}")
            assert response.json()["progress"]["current"] == 5
            assert response.json()["progress"]["total"] == 5
        
        # Test task completion with result
        result = MagicMock(spec=TaskResult)
        result.__class__.__name__ = "MockTaskResult"
        result.samples_completed = 10
        
        stats = MagicMock(spec=EvalStats)
        stats.time_elapsed = 15.5
        stats.time_per_sample = 1.55
        stats.tokens_per_second = 45.2
        stats.tokens_per_sample = 70.1
        stats.total_tokens = 701
        
        result.stats = stats
        
        task_display.complete(result)
        
        response = session.get(f"http://{http_display.host}:{http_display.port}/tasks/{task_id}")
        result_data = response.json()["result"]
        assert result_data["type"] == "MockTaskResult"
        assert result_data["samples_completed"] == 10
        assert result_data["stats"]["time_elapsed"] == 15.5
        assert result_data["stats"]["total_tokens"] == 701


@pytest.mark.asyncio
async def test_http_display_error_handling(http_display, session, task_profile):
    """Test error handling in HTTP display."""
    # Reset task states
    http_display.task_states = {}
    http_display.task_counter = 0
    
    # Test task not found error
    response = session.get(f"http://{http_display.host}:{http_display.port}/tasks/nonexistent")
    assert response.status_code == 404
    assert "not found" in response.json()["message"]
    
    # Test error result handling
    with http_display.task(task_profile) as task_display:
        task_id = list(http_display.task_states.keys())[0]
        
        # Create mock error result
        result = MagicMock(spec=TaskResult)
        result.__class__.__name__ = "ErrorTaskResult"
        result.samples_completed = 5
        result.exc_type = ValueError
        result.exc_value = ValueError("Test error message")
        
        # Complete the task with error result
        task_display.complete(result)
        
        # Verify error in API
        response = session.get(f"http://{http_display.host}:{http_display.port}/tasks/{task_id}")
        assert response.status_code == 200
        
        result_data = response.json()["result"]
        assert result_data["type"] == "ErrorTaskResult"
        assert result_data["samples_completed"] == 5
        assert result_data["error"]["type"] == "ValueError"
        assert result_data["error"]["message"] == "Test error message"


@pytest.mark.asyncio
async def test_http_display_multiple_tasks(http_display, session, task_profile):
    """Test handling multiple tasks simultaneously."""
    # Reset task states
    http_display.task_states = {}
    http_display.task_counter = 0
    
    # Create a second profile with different values
    profile2 = TaskProfile(
        name="task_2",
        file="test2.py",
        model=ModelName("anthropic/claude-3-5-haiku-latest"),
        dataset="dataset2",
        scorer="scorer2",
        samples=20,
        steps=10,
        eval_config={},
        task_args={},
        generate_config={},
        tags=["test2"],
        log_location="test2.log"
    )
    
    # Create first task
    with http_display.task(task_profile) as task_display1:
        task_id1 = list(http_display.task_states.keys())[0]
        
        # Create second task while first is still active
        with http_display.task(profile2) as task_display2:
            # Should have two tasks now
            assert len(http_display.task_states) == 2
            task_ids = list(http_display.task_states.keys())
            task_id2 = task_ids[1] if task_ids[1] != task_id1 else task_ids[0]
            
            # Both tasks should be accessible via API
            response = session.get(f"http://{http_display.host}:{http_display.port}/tasks")
            assert response.status_code == 200
            assert len(response.json()) == 2
            
            # Update both tasks
            with task_display1.progress() as progress1:
                progress1.update(2)
                
                task_display1.sample_complete(3, 10)
                
                with task_display2.progress() as progress2:
                    progress2.update(5)
                    
                    task_display2.sample_complete(10, 20)
                    
                    # Check both tasks are updated correctly
                    response1 = session.get(f"http://{http_display.host}:{http_display.port}/tasks/{task_id1}")
                    response2 = session.get(f"http://{http_display.host}:{http_display.port}/tasks/{task_id2}")
                    
                    assert response1.json()["samples_complete"] == 3
                    assert response1.json()["progress"]["current"] == 2
                    
                    assert response2.json()["samples_complete"] == 10
                    assert response2.json()["progress"]["current"] == 5


@pytest.mark.asyncio
async def test_progress_methods():
    """Test HttpProgress class methods."""
    progress = HttpProgress(10)
    assert progress.total == 10
    assert progress.current == 0
    
    progress.update(3)
    assert progress.current == 3
    
    progress.update()  # Default increment by 1
    assert progress.current == 4
    
    progress.complete()
    assert progress.current == 10


@pytest.mark.asyncio
async def test_task_screen(http_display, session):
    """Test task screen creation and API availability during screen lifetime."""
    tasks = [TaskSpec(name="test_task", model=ModelName("anthropic/claude-3-5-haiku-latest"))]
    
    # Reset task states for this test
    http_display.task_states = {}
    
    async with http_display.task_screen(tasks, False) as screen:
        assert http_display.server_running
        assert http_display.task_states == {}
        
        # Test API while task screen is active
        response = session.get(f"http://{http_display.host}:{http_display.port}/tasks")
        assert response.status_code == 200
        assert response.json() == []