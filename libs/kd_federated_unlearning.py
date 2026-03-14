import torch
import copy
from torch.utils.data import ConcatDataset, DataLoader
import torch.nn as nn
import torch.optim as optim

from utils import *

def knowledge_distillation_unlearning(global_model, logs, clients, client_id_to_unlearn, create_model_fn, distillation_ds, dist_epochs, temp,
                                      hard_loss_weight=0.0, dist_lr=0.001, device="cpu", verbose=True, eval_dl = None):
    retained_clients = [c for c in clients if c.id != client_id_to_unlearn]
    unleanred_client = [c for c in clients if c.id == client_id_to_unlearn][0]

    if verbose:
        retained_ds = ConcatDataset([c.ds for c in retained_clients])
        retained_dl = DataLoader(retained_ds, batch_size=256)
    
    global_model = copy.deepcopy(global_model).cpu()

    # Computation of the unlearned model
    total_unlearned_client_updates = nn.utils.parameters_to_vector(logs[1].client_updates[client_id_to_unlearn].parameters()) - \
                                        nn.utils.parameters_to_vector(logs[0].global_model.parameters())
    for i in range(1, len(logs)-1):
        total_unlearned_client_updates += nn.utils.parameters_to_vector(logs[i+1].client_updates[client_id_to_unlearn].parameters()) - \
                                        nn.utils.parameters_to_vector(logs[i].global_model.parameters())
    
    unlearned_model_vec = nn.utils.parameters_to_vector(global_model.parameters()) - \
                        total_unlearned_client_updates/len(logs)
    
    unlearned_model = create_model_fn().cpu()
    nn.utils.vector_to_parameters(unlearned_model_vec, unlearned_model.parameters())

    if verbose:
        unlearned_model.to(device)
        train_acc = compute_accuracy(unlearned_model, retained_dl, device=device)
        unleanred_acc = compute_accuracy(unlearned_model, unleanred_client.dl, device=device)

        eval_acc = -1
        if eval_dl != None:    
            eval_acc = compute_accuracy(unlearned_model, eval_dl, device=device)

        print(f"Unlearned model: train acc = {train_acc:.4f}, unlearned acc = {unleanred_acc:.4f}, eval acc = {eval_acc:.4f}")

            

    # Knowledge distillation
    teacher, student = global_model, unlearned_model
    teacher.to(device)
    student.to(device)
    teacher.eval()
    student.train()

    optimizer = optim.SGD(student.parameters(), lr=dist_lr, momentum=0.9)
    ce_loss = nn.CrossEntropyLoss()
    distillation_dl = DataLoader(distillation_ds, batch_size=64, shuffle=True)   


    for e in range(dist_epochs):
        for imgs, lbls in distillation_dl:
            imgs, lbls = imgs.to(device), lbls.to(device)

            with torch.no_grad():
                teacher_logits = teacher(imgs)
            student_logits = student(imgs)

            soft_teacher = nn.functional.softmax(teacher_logits / temp, dim=-1)
            soft_stedent = nn.functional.log_softmax(student_logits / temp, dim=-1)

            soft_loss = torch.sum(soft_teacher * (soft_teacher.log() - soft_stedent)) / soft_stedent.size()[0] * (temp**2)
            hard_loss = ce_loss(student_logits, lbls)
            loss = (1.0 - hard_loss_weight) * soft_loss + hard_loss_weight * hard_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if verbose:
            train_acc = compute_accuracy(student, retained_dl, device=device)
            unleanred_acc = compute_accuracy(student, unleanred_client.dl, device=device)

            eval_acc = -1
            if eval_dl != None:    
                eval_acc = compute_accuracy(student, eval_dl, device=device)

            print(f"KD epoch {e+1}/{dist_epochs}: train acc = {train_acc:.4f}, unlearned acc = {unleanred_acc:.4f}, eval acc = {eval_acc:.4f}")

    return student
