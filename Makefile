
clean:
	find . -name __pycache__ | xargs rm -r
sclean:
	find scripts -name __pycache__ | xargs rm -r
wclean:
	find . -name wandb | xargs rm -r

format:
	black .

env:
	conda activate bafl

