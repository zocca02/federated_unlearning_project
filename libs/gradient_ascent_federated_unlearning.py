import torch
import copy
from torch.utils.data import ConcatDataset, DataLoader
import torch.nn as nn
import numpy as np

from utils import *

def gradient_ascent_unlearning(global_model, create_model_fn, clients, unl_client_model, client_id_to_unlearn, num_local_epochs_unlearn = 5, ulr = 0.001, 
              distance_threshold = 2.2, clip_grad = 5, num_updates_in_epoch = None, device="cpu", eval_dl=None, verbose=False):

    retained_clients = [c for c in clients if c.id != client_id_to_unlearn]
    unleanred_client = [c for c in clients if c.id == client_id_to_unlearn][0]

    unleanred_dl = unleanred_client.dl

    if verbose:
        retained_ds = ConcatDataset([c.ds for c in retained_clients])
        retained_dl = DataLoader(retained_ds, batch_size=256)
        

    global_model = copy.deepcopy(global_model).cpu()
    unl_client_model = copy.deepcopy(unl_client_model).cpu()

    #compute reference model
    #w_ref = N/(N-1)w^T - 1/(N-1)w^{T-1}_i = \sum{i \ne j}w_j^{T-1}
    model_ref_vec = len(clients) / (len(clients) - 1) * nn.utils.parameters_to_vector(global_model.parameters()) \
                                - 1 / (len(clients) - 1) * nn.utils.parameters_to_vector(unl_client_model.parameters())

    #compute threshold
    model_ref = create_model_fn().cpu()
    nn.utils.vector_to_parameters(model_ref_vec, model_ref.parameters())

    dist_ref_random_lst = []
    random_model = create_model_fn().cpu()
    for _ in range(30):
        dist_ref_random_lst.append(get_distance_between_models(model_ref, random_model))    

    threshold = np.mean(dist_ref_random_lst) / 3


    ###############################################################
    #### Unlearning
    ###############################################################
    model_ref = model_ref.to(device)
    unl_client_model = unl_client_model.to(device)

    unlearned_model = copy.deepcopy(model_ref).to(device)
    
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(unlearned_model.parameters(), lr=ulr, momentum=0.9) 

    for epoch in range(num_local_epochs_unlearn):
       
        for batch_id, (images, labels) in enumerate(unleanred_dl):
            unlearned_model.train()
            images, labels = images.to(device), labels.to(device)
            opt.zero_grad()

            loss = criterion(unlearned_model(images), labels)
            loss_joint = -loss # negate the loss for gradient ascent
            loss_joint.backward()
            
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(unlearned_model.parameters(), clip_grad)

            opt.step()

            with torch.no_grad():
                distance = get_distance_between_models(unlearned_model, model_ref)
                if distance > threshold:
                    dist_vec = nn.utils.parameters_to_vector(unlearned_model.parameters()) - nn.utils.parameters_to_vector(model_ref.parameters())
                    dist_vec = dist_vec/torch.norm(dist_vec)*np.sqrt(threshold)
                    proj_vec = nn.utils.parameters_to_vector(model_ref.parameters()) + dist_vec
                    nn.utils.vector_to_parameters(proj_vec, unlearned_model.parameters())

            distance_ref_unlearned_cleint = get_distance_between_models(unlearned_model, unl_client_model)
            
            if distance_ref_unlearned_cleint > distance_threshold:
                break

            if num_updates_in_epoch is not None and batch_id >= num_updates_in_epoch:
                break

        if verbose:

            train_acc = compute_accuracy(unlearned_model, retained_dl, device=device)
            unleanred_acc = compute_accuracy(unlearned_model, unleanred_dl, device=device)

            eval_acc = -1
            if eval_dl != None:    
                eval_acc = compute_accuracy(unlearned_model, eval_dl, device=device)
            
            print('Distance from the unlearned model to unlearned client:', distance_ref_unlearned_cleint.item())
            print(f"Ended Gradient Ascent for epoch {epoch+1}/{num_local_epochs_unlearn}, train acc = {train_acc:.4f}, unlearned acc = {unleanred_acc:.4f}, eval acc = {eval_acc:.4f}")

            
    ####################################################################                           

    return unlearned_model