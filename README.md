# federated_unlearning_project
Project work for the course "Sicurezza dell'Informazione" of the University of Bologna about federated unlearning

`./lib` folder contains just a collection of files with the copy of the functions I made and used in the notebooks. The notebooks are self contained.

`my_simple_backdoor_unlearning.ipynb` contains the test I made on a simple unlearning method that keeps private the last layer of each client, showing that it works with backdoor patterns.

`my_simple_privacy_unlearning.ipynb` contains the test I made on the very same model of the previous one on unlearning data for privacy reasons, showing with membership inference attacks how it isn't effective to completely erase the contribution of a client in case the privacy aspect is concerned

`mia_analysis_unlearning.ipynb` tests some federated unlearning algorithms (FedEraser, Gradient Ascent unlearning and Knowledge distillation based unlearning) compered with a retrained model. All the methods are tested running a membership inference attack based on LIRA. I further tested every method performing the membership inference attack based on some statistics from the last layer of the CNN used to test also if there was information inside the model that could lead to tell apart the unlearned data from the ones the network never seen.

`unlearning_tests.ipynb` tests different approachs which can show differences in how a model behaves with training, unlearned and never seen data, trying to find if some of them can show if, after running the unlearning algorithm, the model has truly unlearned the data or not.

For all the tests a resnet18 has been used as model and has been trained on CIFAR10. I let the model overfit on purpose to stand out the result of the membership inference attacks.
