Training Robust Deep Neural Networks via Adversarial Noise Propagation
==

This repository contains the source code for paper 'Training Robust Deep Neural Networks via Adversarial Noise Propagation'. We give the codes for VGG-16 on CIFAR-10.


Dependencies
--
This library uses Pytorch to accelerate graph computations performed by many machine learning models.<br>
Installing Pytorch will take care of all other dependencies like numpy and scipy.

Train models
--
sh train.sh

Test models
--
sh test.sh

Model Robustness Evaluation
--
* Empirical Worst Case Decision Boundary Distance<br>
sh db.sh
* Empirical Noise Insensitivity<br>
sh eni.sh
* Corruption Robustness Evaluation<br>
sh cpre1.sh and sh cpre2.sh
