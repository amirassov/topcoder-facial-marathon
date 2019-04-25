APP_NAME=amirassov/topcoder-facial-detection
CONTAINER_NAME=code

# HELP
.PHONY: help

help: ## This help.
	@awk 'BEGIN (FS = ":.*?## ") /^[a-zA-Z_-]+:.*?## / (printf "\033[36m%-30s\033[0m %s\n", $$1, $$2)' $(MAKEFILE_LIST)

build:  ## Build the container
	nvidia-docker build -t $(APP_NAME) .

run: ## Run container in dgx
	nvidia-docker run \
		-it \
		--ipc=host \
		--name=$(CONTAINER_NAME) \
		-v $(shell pwd):/code \
		-v /raid/data_share/topcoder/data/:/data \
		-v /raid/data_share/amirassov/detection/code:/wdata $(APP_NAME)

run-omen: ## Run container in omen
	nvidia-docker run \
		-it \
		--ipc=host \
		-v $(shell pwd):/topcoder-facial-detection \
		-v /home/videoanalytics/data/topcoder:/data \
		-v /home/videoanalytics/data/dumps:/wdata $(APP_NAME)
		--name=$(CONTAINER_NAME) $(APP_NAME)

exec: ## Run a bash in a running container
	nvidia-docker exec -it $(CONTAINER_NAME) bash

stop: ## Stop and remove a running container
	docker stop $(CONTAINER_NAME); docker rm $(CONTAINER_NAME)

download: ## Download pretrained weights
	wget -O /wdata/cascade_rcnn_dconv_c3-c5_r50_fpn_1x_20190125-dfa53166.pth $(WEIGHTS) --no-check-certificate
