import click

from leela_interp import inputs


@click.group()
def lirun() -> None:
    """Run a leela interp command."""


for module in [inputs]:
    runners = getattr(module, "RUNNERS", {})

    if not runners:
        continue

    command_name = module.__name__.split(".")[-1]

    @click.group(name=command_name)
    def _runner() -> None:
        pass

    for name, runner in runners.items():
        _runner.add_command(runner, name)

    lirun.add_command(_runner)
