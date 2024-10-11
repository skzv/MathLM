instruction on how to setup vast.ai

COMMON

basic stuff
  sudo apt update
  sudo apt install ubuntu-drivers-common
  ssh-keygen and add pubkey to github

bashrc env variables
  export WANDB_API_KEY=<>
  export HF_TOKEN=<>
  export OPENAI_API_KEY=<>
  export NVIDIA_API_KEY=<>
  source ~/.bashrc
NB: OPENAI_API_KEY is for the llm_judge is not strictly necessary for MATH and GSM8K datasets

system checks
  - torch/torchvision
    python -c "import torch;print(torch.__version__)"
    python -c "import torchvision;print(torchvision.__version__)"
  - docker
    sudo docker run hello-world
  - gpu hw
    - list devices and find recommended driver version
      ubuntu-drivers devices
    - driver working fine
      cat /proc/driver/nvidia/version
      lsmod | grep nvidia
      nvidia-smi
  - cuda version
    nvcc --version
  - cudnn version
    python -c "import torch;print(torch.backends.cudnn.version())"
  -  nvidia container toolkit: 
    https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

nemo prereqs
  # instructions: https://github.com/Kipok/NeMo-Skills/blob/main/docs/prerequisites.md
  git clone git@github.com:Kipok/NeMo-Skills.git
  pip install git+https://github.com/Kipok/NeMo-Skills.git
  python -m nemo_skills.dataset.prepare
  python -c "import nemo_skills; print(nemo_skills.__path__)"
  folders setup
    - create workspace folder
    - copy example-local.yaml => local.yaml
    - add the mount spec in local.yaml
    - ln -s cluser_configs folder to the workspace 
  ns eval \
    --cluster local \
    --server_type openai \
    --model meta/llama-3.1-8b-instruct \
    --server_address https://integrate.api.nvidia.com/v1 \
    --benchmarks gsm8k:0,math:0 \
    --output_dir /workspace/test-eval \
    ++max_samples=10

nemo run

  judge
ns llm_math_judge \
    --cluster=local \
    --model=gpt-4o \
    --server_type=openai \
    --server_address=https://api.openai.com/v1 \
    --input_files="/workspace/openmath2-llama3.1-8b-eval/eval-results/**/output*.jsonl" \
    ++batch_size=10



#LAMBDALABS
0. request an instance
1. instance comes with user ubuntu instead of root => need to add it to the docker group
    sudo usermod -aG docker $USER
   then logout and login for the group change to take effect
2. do #COMMON
 

pros:
* can get h100 easily
* docker and cuda, cudnn preinstalled
cons:
* cannot pause an instance, upon termination all data is deleted :/

#VAST.AI

0. create a VM using the 22.04 template 
  https://cloud.vast.ai/?template_id=e2f271ce754aeef772181fbf9c82a354
NB: for the moment it's still beta and i only see RTX boxes available, no H100

1. install docker
  - instructions 
    https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository 
  - start the deamon
    sudo systemctl start docker
  - check
    sudo docker run hello-world

2. install gpu driver
  - purge nvdia stuff
  - list devices and find recommended driver version
    sudo apt install ubuntu-drivers-common
    ubuntu-drivers devices 
  - i need 560 from the command above
    sudo apt install nvidia-driver-560
    sudo reboot

2. install cuda
  # the following commands are from https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local
  # Pin the CUDA repository
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
    sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
  # Download and install the local CUDA repository
    wget https://developer.download.nvidia.com/compute/cuda/12.6.2/local_installers/cuda-repo-ubuntu2204-12-6-local_12.6.2-560.35.03-1_amd64.deb
    sudo dpkg -i cuda-repo-ubuntu2204-12-6-local_12.6.2-560.35.03-1_amd64.deb
  # Add the repository GPG key
    sudo cp /var/cuda-repo-ubuntu2204-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/
  # Install the CUDA Toolkit 12.6
    sudo apt-get -y install cuda-toolkit-12-6
  # optional in case the nvcc step below does not work
  # add the following to ~/.bashrc and source it
    echo "export PATH=/usr/local/cuda-12.6/bin${PATH:+:${PATH}}" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" >> ~/.bashrc
  # check nvcc
    nvcc -version

3. install cudnn
  # now i need cuDNN for doing DL stuff with cuda
  # https://developer.nvidia.com/cudnn-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local
  # choose cuda 12 installation

3. miniconda
  - instructions
    https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html
    https://docs.anaconda.com/miniconda/
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /opt/miniconda-installer.sh
  nb: i use /opt/conda as prefix and chose to allow to autostart the bash script
  - create the mlm environment
    conda create -n mlm
  - install pytorch
    conda install pytorch torchvision -c pytorch 

3. do #COMMON


4) install cuda 12.6
* instructions: https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local
* remarks:
  - i ran into an issue with /etc/enviroment where my ssh key was interfering with the command 
  - after that was not sure what to do next. i tried to run "nvcc --version" but nvcc was not present yet

* additional instructions: https://gist.github.com/denguir/b21aa66ae7fb1089655dd9de8351a202
then i found i had to install the nvidia toolkit 
sudo apt install nvidia-cuda-toolkit


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


