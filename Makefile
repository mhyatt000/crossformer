
clean:
	find . --name __pycache__ | xargs rm -r

format:
	black .

env:
	conda activate bafl

