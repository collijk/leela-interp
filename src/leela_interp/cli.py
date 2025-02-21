import click

from leela_interp import extract


@click.group()
def lirun() -> None:
    """Run a leela interp pipeline."""


@click.group()
def litask() -> None:
    """Run an individual task in the leela interp pipeline."""


for module in [extract]:
    runners = getattr(module, "RUNNERS", {})
    task_runners = getattr(module, "TASK_RUNNERS", {})

    if not runners or not task_runners:
        continue

    command_name = module.__name__.split(".")[-1]

    @click.group(name=command_name)
    def workflow_runner() -> None:
        pass

    for name, runner in runners.items():
        workflow_runner.add_command(runner, name)

    lirun.add_command(workflow_runner)

    @click.group(name=command_name)
    def task_runner() -> None:
        pass

    for name, runner in task_runners.items():
        task_runner.add_command(runner, name)

    litask.add_command(task_runner)
