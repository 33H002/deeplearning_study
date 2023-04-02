build:
		@echo "---- BUILD ----"
		@docker build -t torch-cluster .

start:
		@echo "---- START ----"
		@sh ./startCluster.sh