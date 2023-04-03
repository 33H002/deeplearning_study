build :
	@echo "---- BUILD ----"
	@docker build -t torch-cluster .

start :
	@echo "---- START ----"
	@chmod +x startHadoopCluster.sh
	@s./startCluster.sh

stop :
	@echo "---- STOP ----"
	@chmod +x stopCluster.sh
	@./stopCluster.sh