instruction on how to setup vast.ai

steps

0) create a VM using the 22.04 template : https://cloud.vast.ai/?template_id=e2f271ce754aeef772181fbf9c82a354
NB:
* for the moment it's still beta and i only see RTX boxes available, no H100

1) install docker
* instructions https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository 
* start the deamon: sudo systemctl start docker
* test installation: sudo docker run hello-world
 

2) nvidia container toolkit
* instructions: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

3) install miniconda
* instructions: https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html
https://docs.anaconda.com/miniconda/
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /opt/miniconda-installer.sh
i use /opt/conda as prefix and chose to allow to autostart the bash script
3.1) create the mlm environment
conda create -n mlm

4) install cuda 12.6
* instructions: https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local
* remarks:
  - i ran into an issue with /etc/enviroment where my ssh key was interfering with the command 
  - after that was not sure what to do next. i tried to run "nvcc --version" but nvcc was not present yet

* additional instructions: https://gist.github.com/denguir/b21aa66ae7fb1089655dd9de8351a202
then i found i had to install the nvidia toolkit 
sudo apt install nvidia-cuda-toolkit

5) install cudnn runtime
[instructions: https://gist.github.com/denguir/b21aa66ae7fb1089655dd9de8351a202]
instructions: https://developer.nvidia.com/cudnn-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local

6) install pytorch
conda install pytorch torchvision -c pytorch 

7) other stuff
conda install jupyter

8) nemo-skills: 
instructions: https://github.com/Kipok/NeMo-Skills/blob/main/docs/prerequisites.md

remarks:
* comments from Rob (vast.ai support)
You can use the template onstart script to automate the install somewhat once you have worked out what you need to do to get it up and running.
* i do not know what is the difference between cudnn and cuda
Generally this would be:

Install docker
Install Nvidia container toolkit
Restart docker to pick up the changes
Pull and launch your intended docker image

This should be much more straightforward when we complete testing


