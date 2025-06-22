#!/bin/bash
set -e
set -u

if [ $# -eq 0 ]
then
    echo "running docker without display"
    docker run -it --network=host -v ~/Human2LocoMan:/home/gymuser/Human2LocoMan/ \
		--gpus='"device=0"' \
		--name=locoman_container isaacgym_locoman /bin/bash \
		-c "echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc && source ~/.bashrc \
			&& conda create -n lmdog python=3.8 && source activate lmdog && conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia && conda install mkl==2024.0.0 \
    		&& cd /home/gymuser/Human2LocoMan && pip install -e . && conda install pinocchio -c conda-forge"

else export DISPLAY=$DISPLAY
	echo "setting display to $DISPLAY"
	xhost +
	docker run -it --volume="$HOME/.Xauthority:/home/gymuser/.Xauthority:rw" -v /tmp/.X11-unix:/tmp/.X11-unix \
		-e DISPLAY=$DISPLAY --ipc=host --network=host --privileged=true -v ~/Human2LocoMan:/home/gymuser/Human2LocoMan/ \
		--gpus='"device=0"' \
		--name=locoman_container isaacgym_locoman /bin/bash \
		-c "echo 'source /opt/ros/noetic/setup.bash' >> ~/.bashrc && source ~/.bashrc \
			&& conda create -n lmdog python=3.8 && source activate lmdog && conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia && conda install mkl==2024.0.0 \
			&& cd /home/gymuser/Human2LocoMan && pip install -e . && conda install pinocchio -c conda-forge"
	xhost -
fi