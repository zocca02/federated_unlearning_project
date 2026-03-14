import numpy as np
import zipfile
import os
from torch.utils.data import Subset, DataLoader
import torch
import torch.nn as nn
from scipy.stats import norm

# My libraries
from utils import save_model, save_array, load_model, load_array, get_activations

import numpy as np
import zipfile
import os
from torch.utils.data import Subset, DataLoader
import torch
import torch.nn as nn
from scipy.stats import norm



class LIRAShadowModel:
    def __init__(self, id, model, idxs_to_keep, keep_bool):
        self.id = id
        self.model = model
        self.idxs = idxs_to_keep
        self.keep_bool = keep_bool
        self.phis = None
        self.statistic = None

class MIALira:
    def __init__(self, n_shadow_models, ds_att, target_model, target_model_memberships, create_model_fn, device = "cpu", seed=42):
        self.n_shadow_models = n_shadow_models
        self.seed = seed
        np.random.seed(seed)
        self.device = device

        self.ds_att = ds_att
        self.target_model = LIRAShadowModel(
            n_shadow_models,
            target_model,
            np.arange(len(ds_att))[target_model_memberships],
            target_model_memberships
        )

        keep = np.random.uniform(0, 1, size=(n_shadow_models, len(ds_att)))         # Draw a random number for each data point for each shadow model
        order = keep.argsort(0)                                                     # Get a matrix with each column filled with the idxs to sort that columns (like the position of each sm for that data point)
        self.global_keep = order < int(0.5 * n_shadow_models)                       # Only the first half sm will keep that data point

        self.shadow_models: list[LIRAShadowModel] = []
        self.shadow_models_with_target: list[LIRAShadowModel] = [self.target_model]

        self.create_model_fn = create_model_fn

        self.true_memberships = []

    def set_target_model(self, target_model, target_model_memberships):
        self.target_model = LIRAShadowModel(
            self.n_shadow_models,
            target_model,
            np.arange(len(self.ds_att))[target_model_memberships],
            target_model_memberships
        )

        self.shadow_models_with_target = [sm for sm in self.shadow_models]
        self.shadow_models_with_target.append(self.target_model)        

    def save_shadow_models(self, save_folder, save_zip=False):
        os.makedirs(save_folder, exist_ok=True)

        if save_zip:
            files_to_zip = []

        for shadow_model in self.shadow_models:
            sm_filename, idxs_filename = f"{save_folder}/shadow_model_{shadow_model.id}", f"{save_folder}/idxs_{shadow_model.id}"
            if save_zip:
                if self.device != "cpu":
                    files_to_zip.extend([f"{sm_filename}.pth", f"{idxs_filename}.npy", f"{sm_filename}_cpu.pth"])
                else:
                    files_to_zip.extend([f"{idxs_filename}.npy", f"{sm_filename}_cpu.pth"])

            save_model(shadow_model.model, sm_filename, device=self.device, verbose=False)
            save_array(shadow_model.idxs, idxs_filename, verbose=False)
        
        if save_zip:
            zip_filename = f"{save_folder}/shadow_models.zip"
            with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as z:
                for f in files_to_zip:
                    z.write(f)

    def train_shadow_models(self, train_model_fn, save_folder=None, save=False, save_zip=False, first_index=0):
        if save and (save_folder is None):
            raise Exception("Save folder must be specified")

        # Train shadow models
        for i in range(first_index, first_index+self.n_shadow_models):
            idxs = self.global_keep[i-first_index].nonzero()[0]
            keep_bool = np.full(len(self.ds_att), False)
            keep_bool[idxs] = True
            
            ds_shadow = Subset(self.ds_att, indices=idxs)

            shadow_model = self.create_model_fn().to(self.device)
            train_model_fn(shadow_model, ds_shadow)

            self.shadow_models.append(LIRAShadowModel(
                i, shadow_model, idxs, keep_bool
            ))

            print(f"Ended train of shadow model {i+1}")

        if save:
            self.save_shadow_models(save_folder, save_zip=save_zip)

        self.shadow_models_with_target = [sm for sm in self.shadow_models]
        self.shadow_models_with_target.append(self.target_model)

    def load_shadow_models(self, folder, first_index=0):
        for i in range(first_index, first_index+self.n_shadow_models):
            shadow_model = self.create_model_fn().to(self.device)
            load_model(shadow_model, f"{folder}/shadow_model_{i}", verbose=False)
            idxs = load_array(f"{folder}/idxs_{i}", verbose=False)
            keep_bool = np.full(len(self.ds_att), False)
            keep_bool[idxs] = True

            self.shadow_models.append(LIRAShadowModel(
                i, shadow_model, idxs, keep_bool
            ))
        print(f"Loaded {self.n_shadow_models} shadow models")

        self.shadow_models_with_target = [sm for sm in self.shadow_models]
        self.shadow_models_with_target.append(self.target_model)

class StandardMIALira(MIALira):
    def __init__(self, n_shadow_models, ds_att, target_model, target_model_memberships, create_model_fn, device="cpu", seed=42):
        super().__init__(n_shadow_models, ds_att, target_model, target_model_memberships, create_model_fn, device=device, seed=seed)
        self.scores_online, self.scores_offline = [], []
        #self.scores_exact_online, self.scores_exact_offline = [], []
        self.mean_in, self.mean_out = np.array([]), np.array([])
        self.std_in, self.std_out = np.array([]), np.array([])

    def reset_scores(self):
        self.scores_online, self.scores_offline = [], []
        #self.scores_exact_online, self.scores_exact_offline = [], []
        self.true_memberships = []


    def compute_loss(self):
        self.shadow_models_with_target = [sm for sm in self.shadow_models]
        self.shadow_models_with_target.append(self.target_model)

        dl_att = DataLoader(self.ds_att, batch_size=128, shuffle=False)
        labels = []

        for i, shadow_model in enumerate(self.shadow_models_with_target):
            sm = shadow_model.model
            sm.eval()

            logits = []
            do_labels = len(labels)==0
            with torch.no_grad():
                for x, y in dl_att:
                    x = x.to(self.device)

                    if do_labels:
                        labels.append(y)
                    outputs = sm(x)
                    logits.append(outputs.cpu().numpy())
            logits = np.concatenate(logits)
            if do_labels:
                labels = torch.concat(labels).numpy()

            # Numerically stable softmax
            predictions = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            predictions = predictions / np.sum(predictions, axis=-1, keepdims=True)

            # Compute probability of the true class and the probability of all other classes
            # np.arange(predictions.shape[0]) is needed to correctly indexing of numpy array
            p_true = predictions[np.arange(predictions.shape[0]), labels]
            predictions[np.arange(predictions.shape[0]), labels] = 0
            p_wrong = np.sum(predictions, axis=-1)

            # Compute phi
            eps = 1e-45
            phis = np.log(p_true + eps) - np.log(p_wrong + eps)
            shadow_model.phis = phis

    def compute_scores(self, inverse_order=False):
        self.compute_loss()
        # Global phi scores and keep matrix
        global_phis, global_keep_bool = [], []
        for shadow_model in self.shadow_models_with_target:
            global_phis.append(shadow_model.phis)
            global_keep_bool.append(shadow_model.keep_bool)
        global_phis = np.array(global_phis)                     # [shadow_id, ith data point]
        global_keep_bool = np.array(global_keep_bool)           # [shadow_id, ith data point]

        n_test_shadow_models = 1
        train_phis, test_phis = global_phis[:-n_test_shadow_models], global_phis[-n_test_shadow_models:]
        train_keep_bool, test_keep_bool = global_keep_bool[:-n_test_shadow_models], global_keep_bool[-n_test_shadow_models:]

        # Split phis if they are in or out the shadow dataset
        phis_in = []
        phis_out = []
        for i in range(train_phis.shape[1]):
            phis_in.append(train_phis[train_keep_bool[:, i], i])
            phis_out.append(train_phis[~train_keep_bool[:, i], i])

        # in_size and out_size should be 0.5*len(ds_att_train) but keep this in the case [actually could not be so because of the test shadow models]
        in_size = min(map(lambda x: len(x), phis_in))
        out_size = min(map(lambda x: len(x), phis_out))

        phis_in = np.array([x[:in_size] for x in phis_in])
        phis_out = np.array([x[:out_size] for x in phis_out])

        # Compute mean and variance for each data point phis
        # Note: the paper uses median because it's more robust to outliers (but teorethically we'd need mean)
        self.mean_in = np.median(phis_in, axis=1)
        self.mean_out = np.median(phis_out, axis=1)

        self.std_in = np.std(phis_in, axis=1)
        self.std_out = np.std(phis_out, axis=1)

        # Test on the target
        self.reset_scores()
        eps = 1e-30
        for true_memebrship, phis in zip(test_keep_bool, test_phis):
            pr_in = norm.logpdf(phis, self.mean_in, self.std_in + eps)
            pr_out = norm.logpdf(phis, self.mean_out, self.std_out + eps)

            online = pr_in - pr_out
            offline = -pr_out

            if inverse_order:
                online, offline = -online, -offline

            self.scores_online.extend(online)
            self.scores_offline.extend(offline)

            # self.scores_exact_online.extend(norm.pdf(phis, loc=self.mean_in, scale=self.std_in)/norm.pdf(phis, loc=self.mean_out, scale=self.std_out))
            # self.scores_exact_offline.extend(np.full(len(phis), 1) - norm.sf(phis, loc=self.mean_out, scale=self.std_out))

            self.true_memberships.extend(true_memebrship)

        # Notes:
        # Logpdf is used because of better numerical stability
        # Again scores_offline is compute in a different way for numerical reasons



class StatisticsMIALira(MIALira):
    def __init__(self, n_shadow_models, ds_att, target_model, target_model_memberships, create_model_fn, device="cpu", seed=42):
        super().__init__(n_shadow_models, ds_att, target_model, target_model_memberships, create_model_fn, device=device, seed=seed)
        self.scores_online, self.scores_offline = [], []
        # self.scores_exact_online, self.scores_exact_offline = [], []
        self.mean_in, self.mean_out = np.array([]), np.array([])
        self.std_in, self.std_out = np.array([]), np.array([])
        self.stats_in, self.stats_out = np.array([]), np.array([])
        self.test_stats, self.test_keep_bool = np.array([]), np.array([])
    
    def reset_scores(self):
        self.scores_online, self.scores_offline = [], []
        self.true_memberships = []
    
    def compute_statistics(self):
        raise NotImplemented()
    
    def get_stats_for_input(self, i):
        if len(self.stats_in)==0 and len(self.stats_out)==0:
            self.divide_stats_in_out()
        
        return self.stats_in[i], self.stats_out[i], self.test_stats[0][i]
    
    def divide_stats_in_out(self):
        self.shadow_models_with_target = [sm for sm in self.shadow_models]
        self.shadow_models_with_target.append(self.target_model)
        self.compute_statistics()

        # Global maxes scores and keep matrix
        global_stats, global_keep_bool = [], []
        for shadow_model in self.shadow_models_with_target:
            global_stats.append(shadow_model.statistic)
            global_keep_bool.append(shadow_model.keep_bool)
        global_stats = np.array(global_stats)                     # [shadow_id, ith data point]
        global_keep_bool = np.array(global_keep_bool)           # [shadow_id, ith data point]

        n_test_shadow_models = 1
        train_stats, test_stats = global_stats[:-n_test_shadow_models], global_stats[-n_test_shadow_models:]
        train_keep_bool, test_keep_bool = global_keep_bool[:-n_test_shadow_models], global_keep_bool[-n_test_shadow_models:]

        # Split phis if they are in or out the shadow dataset
        stats_in = []
        stats_out = []
        for i in range(train_stats.shape[1]):
            stats_in.append(train_stats[train_keep_bool[:, i], i])
            stats_out.append(train_stats[~train_keep_bool[:, i], i])
        
        in_size = min(map(lambda x: len(x), stats_in))
        out_size = min(map(lambda x: len(x), stats_out))

        self.stats_in = np.array([x[:in_size] for x in stats_in])
        self.stats_out = np.array([x[:out_size] for x in stats_out])
        self.test_stats = test_stats
        self.test_keep_bool = test_keep_bool
    
    def compute_scores(self, inverse_order=False):

        # Compute mean and variance for each data point phis
        # Note: the paper uses median because it's more robust to outliers (but teorethically we'd need mean)
        self.mean_in = np.median(self.stats_in, axis=1)
        self.mean_out = np.median(self.stats_out, axis=1)

        self.std_in = np.std(self.stats_in, axis=1)
        self.std_out = np.std(self.stats_out, axis=1)

        # Test on the target
        self.reset_scores()
        eps = 1e-30
        for true_memebrship, stats in zip(self.test_keep_bool, self.test_stats):
            pr_in = norm.logpdf(stats, self.mean_in, self.std_in + eps)
            pr_out = norm.logpdf(stats, self.mean_out, self.std_out + eps)

            online = pr_in - pr_out
            offline = -pr_out

            if inverse_order:
                online, offline = -online, -offline

            self.scores_online.extend(online)
            self.scores_offline.extend(offline)

            # self.scores_exact_online.extend(norm.pdf(maxes, loc=self.mean_in, scale=self.std_in)/norm.pdf(maxes, loc=self.mean_out, scale=self.std_out))
            # self.scores_exact_offline.extend(np.full(len(maxes), 1) - norm.sf(maxes, loc=self.mean_out, scale=self.std_out))

            self.true_memberships.extend(true_memebrship)

        # Notes:
        # Logpdf is used because of better numerical stability
        # Again scores_offline is compute in a different way for numerical reasons


class MaxFeatureMIALira(StatisticsMIALira):
    def __init__(self, n_shadow_models, ds_att, target_model, target_model_memberships, create_model_fn, features_extractor, device="cpu", seed=42):
        super().__init__(n_shadow_models, ds_att, target_model, target_model_memberships, create_model_fn, device=device, seed=seed)
        self.features_extractor = features_extractor

    def compute_statistics(self):
        self.shadow_models_with_target = [sm for sm in self.shadow_models]
        self.shadow_models_with_target.append(self.target_model)

        dl_att = DataLoader(self.ds_att, batch_size=128, shuffle=False)

        for i, shadow_model in enumerate(self.shadow_models_with_target):
            sm = shadow_model.model
            sm.eval()

            maxes = []
            with torch.no_grad():
                for x, y in dl_att:
                    x = x.to(self.device)

                    features = self.features_extractor(sm, x)

                    max_features = torch.max(features, dim=1).values

                    maxes.extend(max_features.cpu().numpy())
            maxes = np.stack(maxes)

            shadow_model.statistic = maxes

class NormFeatureMIALira(StatisticsMIALira):
    def __init__(self, n_shadow_models, ds_att, target_model, target_model_memberships, create_model_fn, features_extractor, device="cpu", seed=42):
        super().__init__(n_shadow_models, ds_att, target_model, target_model_memberships, create_model_fn, device=device, seed=seed)
        self.features_extractor = features_extractor

    def compute_statistics(self):
        self.shadow_models_with_target = [sm for sm in self.shadow_models]
        self.shadow_models_with_target.append(self.target_model)

        dl_att = DataLoader(self.ds_att, batch_size=128, shuffle=False)

        for i, shadow_model in enumerate(self.shadow_models_with_target):
            sm = shadow_model.model
            sm.eval()

            norms = []
            with torch.no_grad():
                for x, y in dl_att:
                    x = x.to(self.device)

                    features = self.features_extractor(sm, x)
                    
                    norms.extend(torch.norm(features, dim=1).cpu().numpy())
            norms = np.stack(norms)

            shadow_model.statistic = norms


class MeanFeatureMIALira(StatisticsMIALira):
    def __init__(self, n_shadow_models, ds_att, target_model, target_model_memberships, create_model_fn, features_extractor, device="cpu", seed=42):
        super().__init__(n_shadow_models, ds_att, target_model, target_model_memberships, create_model_fn, device=device, seed=seed)
        self.features_extractor = features_extractor

    def compute_statistics(self):
        self.shadow_models_with_target = [sm for sm in self.shadow_models]
        self.shadow_models_with_target.append(self.target_model)

        dl_att = DataLoader(self.ds_att, batch_size=128, shuffle=False)

        for i, shadow_model in enumerate(self.shadow_models_with_target):
            sm = shadow_model.model
            sm.eval()

            means = []
            with torch.no_grad():
                for x, y in dl_att:
                    x = x.to(self.device)

                    # activations = get_activations(sm, x)
                    # features = torch.flatten(nn.AdaptiveAvgPool2d((1,1))(activations["features"]), 1)
                    features = self.features_extractor(sm, x)

                    mask = features != 0
                    means_without_zero = (features * mask).sum(dim=1) / mask.sum(dim=1)

                    means.extend(means_without_zero.cpu().numpy())
            means = np.stack(means)

            shadow_model.statistic = means

class VarFeatureMIALira(StatisticsMIALira):
    def __init__(self, n_shadow_models, ds_att, target_model, target_model_memberships, create_model_fn, features_extractor, device="cpu", seed=42):
        super().__init__(n_shadow_models, ds_att, target_model, target_model_memberships, create_model_fn, device=device, seed=seed)
        self.features_extractor = features_extractor
    
    def compute_statistics(self):
        self.shadow_models_with_target = [sm for sm in self.shadow_models]
        self.shadow_models_with_target.append(self.target_model)

        dl_att = DataLoader(self.ds_att, batch_size=128, shuffle=False)

        for i, shadow_model in enumerate(self.shadow_models_with_target):
            sm = shadow_model.model
            sm.eval()

            variances = []
            with torch.no_grad():
                for x, y in dl_att:
                    x = x.to(self.device)

                    # activations = get_activations(sm, x)
                    # features = torch.flatten(nn.AdaptiveAvgPool2d((1,1))(activations["features"]), 1)
                    features = self.features_extractor(sm, x)

                    var = torch.var(features, dim=1)

                    variances.extend(var.cpu().numpy())
            variances = np.stack(variances)

            shadow_model.statistic = variances

class NZerosFeatureMIALira(StatisticsMIALira):
    def __init__(self, n_shadow_models, ds_att, target_model, target_model_memberships, create_model_fn, features_extractor, device="cpu", seed=42):
        super().__init__(n_shadow_models, ds_att, target_model, target_model_memberships, create_model_fn, device=device, seed=seed)
        self.features_extractor = features_extractor
    
    def compute_statistics(self):
        self.shadow_models_with_target = [sm for sm in self.shadow_models]
        self.shadow_models_with_target.append(self.target_model)

        dl_att = DataLoader(self.ds_att, batch_size=128, shuffle=False)

        for i, shadow_model in enumerate(self.shadow_models_with_target):
            sm = shadow_model.model
            sm.eval()

            n_zeros = []
            with torch.no_grad():
                for x, y in dl_att:
                    x = x.to(self.device)

                    # activations = get_activations(sm, x)
                    # features = torch.flatten(nn.AdaptiveAvgPool2d((1,1))(activations["features"]), 1)
                    features = self.features_extractor(sm, x)

                    nz = torch.sum(features == 0.0, dim=1)

                    n_zeros.extend(nz.cpu().numpy())
            n_zeros = np.stack(n_zeros)

            shadow_model.statistic = n_zeros

class CombinedStatisticsMIALira(StatisticsMIALira):
    def __init__(self, n_shadow_models, ds_att, target_model, target_model_memberships, create_model_fn, features_extractor, device="cpu",  seed=42):
        super().__init__(n_shadow_models, ds_att, target_model, target_model_memberships, create_model_fn, device=device, seed=seed)
        self.features_extractor = features_extractor
    
    def compute_statistics(self):
        self.shadow_models_with_target = [sm for sm in self.shadow_models]
        self.shadow_models_with_target.append(self.target_model)

        dl_att = DataLoader(self.ds_att, batch_size=128, shuffle=False)



        for i, shadow_model in enumerate(self.shadow_models_with_target):
            sm = shadow_model.model
            sm.eval()

            values = []
            with torch.no_grad():
                for x, y in dl_att:
                    x = x.to(self.device)

                    # activations = get_activations(sm, x)
                    # features = torch.flatten(nn.AdaptiveAvgPool2d((1,1))(activations["features"]), 1)
                    features = self.features_extractor(sm, x)

                    max_features = torch.max(features, dim=1).values

                    mask = features != 0
                    means_without_zero = (features * mask).sum(dim=1) / mask.sum(dim=1)

                    var = torch.var(features, dim=1)

                    norm = torch.norm(features, dim=1)

                    combination = norm*1/20 + max_features*1/2.5 + means_without_zero*1/0.8 + var*1/0.2

                    values.extend(combination.cpu().numpy())
            values = np.stack(values)

            shadow_model.statistic = values
