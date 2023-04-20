install:
	# install commands
	pip install --upgrade pip &&\
	pip install -r requirements.txt
test:
	# test
deploy:
	# deploy
all: install test deploy