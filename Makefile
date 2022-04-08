install:
	# Install poetry
	curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python3 -
	# Install dependancies
	poetry install

run:
	poetry run python3 main.py
