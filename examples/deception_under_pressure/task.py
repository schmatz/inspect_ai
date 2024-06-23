from dataset import prepare_dataset
from scorer import deception_scorer
from solvers import (
    generate_trade_or_refusal,
    test_for_deception,
    test_for_doubling_down,
)

from inspect_ai import Task, task


@task
def deception_under_pressure():
    return Task(
        dataset=prepare_dataset(),
        plan=[
            generate_trade_or_refusal(),
            test_for_deception(),
            test_for_doubling_down(),
        ],
        # I have a variation on the buy action
        scorer=deception_scorer(),
    )
