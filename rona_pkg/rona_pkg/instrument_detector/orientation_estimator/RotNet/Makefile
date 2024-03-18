dockerContainer = rotnet_test
dockerParams = -v $$PWD/:/project -it
#dockerParamsLive= -v $$PWD/:/project -it --net=elastic-ml-network
#dockerExtract=-v $$PWD/:/project -v /data/dc_credentials_data/dc_credentials_text_training_set_20211223:/data -it
dockerGPUParams = -v $$PWD/:/project -it --gpus all
#

ifeq ($(env), ci)
	dockerParams =
endif

default:
	@echo "hello world"

build:
	docker build -t $(dockerContainer) .


bash:
	docker run $(dockerParams) $(dockerContainer) bash

train-full-image:
	docker run $(dockerGPUParams) $(dockerContainer) python3 -m  train 