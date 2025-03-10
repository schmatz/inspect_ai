from test_helpers.utils import file_check

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import includes
from inspect_ai.solver import generate


@task(chdir=True)
def task2():
    return Task(
        dataset=[
            Sample(id=id, input="What is 1+1?", target="2") for id in range(0, 10)
        ],
        solver=[file_check("task2.py"), generate()],
        scorer=includes(),
        metadata={"task_idx": 2},
    )
