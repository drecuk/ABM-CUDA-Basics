#!/bin/bash

##########################################################
#	By Eugene Ch'ng | www.complexity.io
#	Email: genechng@gmail.com
#	----------------------------------------------------------
#  The ERC 'Lost Frontiers' Project
#  Development for the Parallelisation of ABM Simulation
#	----------------------------------------------------------
# This is an installation instruction file for developing CUDA code
# Make sure that your Linux is set up for developing C/C++, OpenGL, etc.
# At the time of writing I am using Ubuntu Linux 18.04
#
# Refer to the installation development foundation instructions at:
# https://github.com/drecuk/ABM-Basics-Installation
# ----------------------------------------------------------
# sudo chmod +x installcuda.h
# ./installcuda.h
#	##########################################################




echo "---------------- install glxinfo"
# Install glxinfo to check your driver and renderer:
sudo apt-get install -y mesa-utils

echo "---------------- check graphics driver used"
# The vendor should show NVIDIA Corporation
# The renderer should show your NVIDIA GPU type/version
glxinfo|egrep "OpenGL vendor|OpenGL renderer*"



###############################################################################
# Carry out the installation below ONLY IF your GPU isn't installed correctly #
# Installation is commented for safety, uncomment the code below if you are   #
# sure that you've got everything installed properly                          #
###############################################################################

echo "---------------- purge all NVIDIA"
# The installation may need you to boot to root command line

# sudo apt-get purge nvidia*
# sudo add-apt-repository ppa:graphics-drivers/ppa
# sudo apt-get update

echo "---------------- install NVIDIA GPU driver"
# Make sure you check your driver version of your NVIDIA GPU
# nvidia-352 is for AlienWare's GeFORCE GTX 750M
# sudo apt-get install nvidia-352

echo "---------------- check graphics driver used"
# glxinfo|egrep "OpenGL vendor|OpenGL renderer*"

echo "---------------- install CUDA toolkit and gcc-6"
# Sudo apt install nvidia-cuda-toolkit gcc-6

echo "---------------- checking for cuda compiler nvcc version"
# This will show the version if everything has been installed correctly
nvcc --version
