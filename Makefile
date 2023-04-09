build :
	@echo "---- BUILD ----"
	@docker build -t torch-cluster .

start :
	@echo "---- START ----"
	@chmod +x bin/startHadoopCluster.sh
	@./bin/startCluster.sh

stop :
	@echo "---- STOP ----"
	@chmod +x stopCluster.sh
	@./bin/stopCluster.sh