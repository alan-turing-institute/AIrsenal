Whenever you run a command in the terminal:
- If using python, make sure you run it from within the virtual environment (source .venv/bin/activate) or with uv by preprending `uv run` to the command.
- If a file has linting errors try to fix them with `ruff format` first rather than changing the code directly yourself.
- Pipe the output to a file, output.txt, that you can read from. Make sure to overwrite each time so that it doesn't grow too big. There is a bug in the current version of Copilot that causes it to not read the output of commands correctly. This workaround allows you to read the output from the temporary file instead.
