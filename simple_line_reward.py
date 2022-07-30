import sys

import numpy as np
import torch
import vigra
from reward_abc import RewardFunctionAbc


class SimpleLineReward(RewardFunctionAbc):

    def __init__(self, s_subgraph, expected_length=100.0, expected_width=5.0, *args, **kwargs):
        super().__init__(s_subgraph, *args, **kwargs)
        self.s_subgraph = s_subgraph
        self.expected_length = expected_length
        self.expected_width = expected_width
        self.expected_ratio = expected_length / expected_width
        self.line_thresh = 10

    def __call__(
        self, prediction_segmentation, superpixel_segmentation, node_feats, dir_edges, subgraph_indices, actions,
        *args, **kwargs
    ):

        dev = prediction_segmentation.device
        edge_scores = []
        exp_factor = 3

        # we consider objects that are bigger than this size to be background
        bg_size_threshold = 10000
        # we consider objects that are smaller than this size to be noise
        false_obj_size_threshold = 100
        # -> this means all objects that are in the size range false_obj_size_threshold to bg_size_threshold are
        # considered as potential foreground objects

        for single_pred, single_sp_seg, s_dir_edges in zip(
            prediction_segmentation, superpixel_segmentation, dir_edges
        ):
            if single_pred.max() == 0:  # image is empty
                edge_score = torch.zeros(int((single_sp_seg.max()) + 1, ), device=dev)
                edges = s_dir_edges[:, :int(s_dir_edges.shape[1] / 2)]
                edge_score = edge_score[edges].max(dim=0).values
                edge_scores.append(edge_score)
                continue

            # compute the prediction ids and sizes of the current object predictions
            pred_ids, label_masses = torch.unique(single_pred, return_counts=True)

            # get the ids of bg objects, false positives and foreground objects
            bg_pred_ids = pred_ids[label_masses >= bg_size_threshold]
            false_pred_ids = pred_ids[label_masses <= false_obj_size_threshold]
            fg_pred_ids = pred_ids[
                (label_masses > false_obj_size_threshold).logical_and(label_masses < bg_size_threshold)
            ]

            # compute the region features with a dummy input
            region_features = vigra.analysis.extractRegionFeatures(
                np.zeros(single_pred.shape, dtype="float32"), single_pred.detach().numpy().astype("uint32")
            )
            region_radii = region_features["RegionRadii"]

            # number of bg objects
            n_bg = len(bg_pred_ids)

            # the scores we calculate
            edge_score = torch.zeros(int((single_sp_seg.max()) + 1, ), device=dev)

            for pred_id in pred_ids:

                # case 1: this is a foreground object: we compute the score based on how good
                # this object matches our expected shape (measured by the radii)
                if pred_id in fg_pred_ids:
                    this_radii = region_radii[pred_id]
                    assert this_radii.shape == (2,)
                    assert this_radii[0] >= this_radii[1]

                    # derive the score from the radii
                    length, width = this_radii
                    ratio = length / width
                    score_len = 1.0 - np.clip(np.abs(length - self.expected_length) / self.expected_length, 0.0, 1.0)
                    score_wid = 1.0 - np.clip(np.abs(width - self.expected_width) / self.expected_width, 0.0, 1.0)
                    score_ratio = 1.0 - np.clip(np.abs(ratio - self.expected_ratio) / self.expected_ratio, 0.0, 1.0)
                    this_score = 0.33 * score_len + 0.33 * score_wid + 0.33 * score_ratio

                # case 2: this is a background object. we only want a single background object, so we assign
                # a lower reward for more than one object
                elif pred_id in bg_pred_ids:
                    if n_bg == 1:
                        this_score = 1.0
                    else:
                        # the / 10 is to have some degree in the rewards, for more
                        # than 10 bg objects we don't provide any reward
                        this_score = np.clip(1.0 - n_bg / 10.0, 0, 1)

                # case 3: this is a noise object, we don't want any of them
                elif pred_id in false_pred_ids:
                    this_score = 0

                # get the superpixels belonging to this object and assign the score to them
                line_sp = torch.unique(single_sp_seg[single_pred == pred_id]).long()
                edge_score[line_sp] = this_score

            if torch.isnan(edge_score).any() or torch.isinf(edge_score).any():
                print(Warning("NaN or inf in scores this should not happen"))
                sys.stdout.flush()
                assert False
            edges = s_dir_edges[:, :int(s_dir_edges.shape[1] / 2)]
            edge_score = edge_score[edges].max(dim=0).values
            edge_scores.append(edge_score)

        t_edge_scores = torch.cat(edge_scores)
        t_edge_scores = (t_edge_scores * exp_factor).exp() / (torch.ones_like(t_edge_scores) * exp_factor).exp()
        assert not torch.isnan(t_edge_scores).any() and \
               not torch.isinf(t_edge_scores).any() and \
               (t_edge_scores >= 0).any(), "### found invalid reward"
        sg_scores = []
        for i, sz in enumerate(self.s_subgraph):
            sg_scores.append(t_edge_scores[subgraph_indices[i].view(-1, sz)].mean(1))

        # return sg_scores, edge_scores.mean(), t_edge_scores.mean()
        return sg_scores, edge_scores, t_edge_scores.mean()
