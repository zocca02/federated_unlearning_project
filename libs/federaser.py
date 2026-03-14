from torch.utils.data import DataLoader, ConcatDataset
import copy
import numpy as np
import torch

# My libraries
from federated_learning import RoundLog, fedavg_from_models, train_client
from utils import compute_accuracy

###################
# FedEraser
###################     

def federaser(old_logs, clients, client_ids_to_unlearn, train_fn, unlearning_interval, epochs, device="cpu", eval_dl=None, verbose=False, return_logs = False):
    #old_logs = copy.deepcopy(old_logs)
    #unlearned_global_models = []
    unlearning_logs = []
    retained_clients = [c for c in clients if c.id not in client_ids_to_unlearn]
    unleanred_clients = [c for c in clients if c.id in client_ids_to_unlearn]

    if verbose:
        train_ds = ConcatDataset([c.ds for c in retained_clients])
        train_dl = DataLoader(train_ds, batch_size=256)

        unlearned_ds = ConcatDataset([c.ds for c in unleanred_clients])
        unleanred_dl = DataLoader(unlearned_ds, batch_size=256)
        

    deleted_logs = []
    # Delete clients to unlearn from logs
    for log in old_logs:
        deleted_logs.append({})
        for client_id in client_ids_to_unlearn:
            if client_id in log.client_updates.keys():
                deleted_logs[-1][client_id] = log.client_updates[client_id]
                log.client_updates.pop(client_id, None)
    
    # Take the tarting model in the first log and then pop the very first log
    new_global_model = old_logs[0].global_model
    #unlearned_global_models.append(copy.deepcopy(new_global_model))
    if return_logs:
        new_global_model = new_global_model.cpu()
        unlearning_logs.append(RoundLog(0, copy.deepcopy(new_global_model), {}))
        new_global_model = new_global_model.to(device)
    #old_logs.pop(0)

    # Select idxs based on unlearning interval

    selected_idxs = np.arange(1, len(old_logs), unlearning_interval)
    selected_GMs = [old_logs[i].global_model for i in selected_idxs]
    selected_CMs = [old_logs[i].client_updates for i in selected_idxs]
    rounds = len(selected_GMs)

    # First round reconstruction don't need calibration
    new_global_model = fedavg_from_models(list(selected_CMs[0].values()), [1 for _ in selected_CMs[0]])
    client_updates = {}
    for key in selected_CMs[0].keys():
        client_updates[key] = selected_CMs[0][key]
    if return_logs:
        new_global_model = new_global_model.cpu()
        unlearning_logs.append(RoundLog(1, copy.deepcopy(new_global_model), client_updates))
        new_global_model = new_global_model.to(device)
    #unlearned_global_models.append(copy.deepcopy(new_global_model))

    if verbose:
        print(f"FedEraser round {1}/{rounds} completed")


    # Next rounds: recalibration
    
    for r in range(1, rounds):
        global_model = new_global_model
        new_client_models = {}
        for client in retained_clients:
            new_client_model = train_client(train_fn, global_model, client, epochs, device=device, inplace=False)
            new_client_model = new_client_model.cpu()
            new_client_models[client.id] = new_client_model
        
        #new_client_models  = global_train_once(global_model, client_data_loaders, n_clients, lr, unl_epochs, device)

        new_global_model = compute_unlearned_model(list(selected_CMs[r].values()), list(new_client_models.values()), selected_GMs[r], global_model)
        #unlearned_global_models.append(copy.deepcopy(new_global_model))
        if return_logs:
            new_global_model = new_global_model.cpu()
            unlearning_logs.append(RoundLog(r+1, copy.deepcopy(new_global_model), new_client_models))
            new_global_model = new_global_model.to(device)

        if verbose:
            new_global_model = new_global_model.to(device)

            train_acc = compute_accuracy(new_global_model, train_dl, device=device)
            unleanred_acc = compute_accuracy(new_global_model, unleanred_dl, device=device)

            eval_acc = -1
            if eval_dl != None:    
                eval_acc = compute_accuracy(new_global_model, eval_dl, device=device)
                
            new_global_model = new_global_model.cpu()
            
            print(f"FedEraser for round {r+1}/{rounds} completed, train acc = {train_acc:.4f}, unlearned acc = {unleanred_acc:.4f}, eval acc = {eval_acc:.4f}")

    for i, log in enumerate(deleted_logs):
        for key in log.keys():
            old_logs[i].client_updates[key] = log[key]

    if return_logs:
        return new_global_model, unlearning_logs
    else:
        return new_global_model

def compute_unlearned_model(old_client_models, new_client_models, global_model_before_forget, global_model_after_forget):
    assert len(old_client_models) == len(new_client_models)
    
    n_clients = len(new_client_models)
    old_update, new_update = {}, {}
    new_global_state_dict = global_model_after_forget.state_dict()
    
    return_model_state = {}#newGM_t + ||oldCM - oldGM_t||*(newCM - newGM_t)/||newCM - newGM_t||
    
    
    
    for key in global_model_before_forget.state_dict().keys():
        old_update[key], new_update[key], return_model_state[key] = 0.0, 0.0, 0.0
        
        # avg of all the client updates for a certain param
        for i in range(n_clients):
            old_update[key] += old_client_models[i].state_dict()[key]
            new_update[key] += new_client_models[i].state_dict()[key]
        old_update[key] /= n_clients
        new_update[key] /= n_clients
        
        old_update[key] = old_update[key] - global_model_before_forget.state_dict()[key]
        new_update[key] = new_update[key] - global_model_after_forget.state_dict()[key]
        
        step_length = torch.norm(old_update[key])#||oldCM - oldGM_t||
        step_direction = new_update[key]/torch.norm(new_update[key])#(newCM - newGM_t)/||newCM - newGM_t||
        
        new_global_state_dict[key] = new_global_state_dict[key].float() + step_length*step_direction
    
    
    new_global_model = copy.deepcopy(global_model_after_forget)
    new_global_model.load_state_dict(new_global_state_dict)
    
    return new_global_model
