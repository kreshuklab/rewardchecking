import sys
from reward_abc import RewardFunctionAbc
from skimage.measure import approximate_polygon, find_contours
from skimage.draw import polygon_perimeter, line
from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line
import torch
from skimage.draw import disk, line
from numpy.linalg import norm
import numpy as np
import matplotlib.pyplot as plt

class HoughLinesReward(RewardFunctionAbc):

    def __init__(self, s_subgraph, *args, **kwargs):
        self.max_p = torch.nn.MaxPool2d(3, padding=1, stride=1)
        self.s_subgraph = s_subgraph

        self.line_thresh = 10
        self.range_rad = [10, 20] #?????????
        self.range_num = [20, 20]
        # super().__init__(s_subgraph, *args, **kwargs)

    def __call__(self, prediction_segmentation, superpixel_segmentation, node_feats, dir_edges, subgraph_indices, actions, *args, **kwargs):

        dev = prediction_segmentation.device
        edge_scores = []
        exp_factor = 3

        # return super().__call__(prediction_segmentation, superpixel_segmentation, dir_edges, subgraph_indices, *args, **kwargs)
        for single_pred, single_sp_seg, s_dir_edges in zip(prediction_segmentation, superpixel_segmentation, dir_edges):
            # print("single_sp_seg.max()", single_sp_seg.max())
            edge_score = torch.zeros(int((single_sp_seg.max()) + 1, ), device=dev)
            if single_pred.max() == 0:  # image is empty
                edges = s_dir_edges[:, :int(s_dir_edges.shape[1] / 2)]
                # print(edge_score.shape)
                # print(edges.shape)
                edge_score = edge_score[edges].max(dim=0).values
                edge_scores.append(edge_score)
                continue
            # get one-hot representation
            one_hot = torch.zeros((int(single_pred.max()) + 1,) + single_pred.size(), device=dev, dtype=torch.long) \
                .scatter_(0, single_pred[None], 1)

            # need masses to determine what potential_objects can be considered background
            label_masses = one_hot.flatten(1).sum(-1)
            # everything else are potential potential_objects
            bg_obj_mask = label_masses > 2000
            potenial_obj_mask = label_masses <= 2000
            false_obj_mask = label_masses < 800
            bg_object_ids = torch.nonzero(bg_obj_mask).squeeze(1)  # object label IDs
            potential_object_ids = torch.nonzero(potenial_obj_mask).squeeze(1)  # object label IDs

            potential_objects = one_hot[potential_object_ids]  # get object masks
            bg_sp_ids = torch.unique((single_sp_seg[None] + 1) * one_hot[bg_object_ids])[1:] - 1
            object_sp_ids = [torch.unique((single_sp_seg[None] + 1) * obj)[1:] - 1 for obj in potential_objects]
            false_sp_ids = torch.unique((single_sp_seg[None] + 1) * one_hot[false_obj_mask])[1:] - 1

            # Detect two radii
            potential_fg = (potential_objects * torch.arange(len(potential_objects), device=dev)[:, None, None]).sum(0).float()
            edge_image = ((- self.max_p(-potential_fg.unsqueeze(0)).squeeze()) != potential_fg).float().cpu().numpy()
            # hough_radii = np.arange(self.range_rad[0], self.range_rad[1]) maybe some range for angles
            # hough_res = hough_circle(edge_image, hough_radii)
            # accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=self.range_num[1])
            
            # calculations for hough line reward
            tested_angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
            hough_res, angles, distc = hough_line(edge_image, theta=tested_angles)
            # accums, angles, distcs = hough_line_peaks(hough_res, angles, distc, num_peaks=self.range_num[1])
            hough_pred_lines = probabilistic_hough_line(edge_image, line_length=20,line_gap=10)
            r_dists, thetas = self.compute_r_theta(prediction_segmentation, hough_pred_lines)
            accums = self.find_accums(r_dists, thetas, hough_res)

            r0 = []
            c0 = []
            r1 = []
            c1 = []
            for lineـ in hough_pred_lines:
                p0, p1 = lineـ

                r0.append(p0[0]) 
                c0.append(p1[0])
                r1.append(p0[1])
                c1.append(p1[1])

            r0 = np.array(r0)
            c0 = np.array(c0)
            r1 = np.array(r1)            #accums = 259, accepted_lines = (259,), r0 = (264,)
            c1 = np.array(c1)
            # mp_lines = torch.from_numpy(np.stack([cy, cx], axis=1))
            accums = np.array(accums)
            accepted_lines = accums > self.line_thresh
            good_obj_cnt = 0

            if any(accepted_lines):

                r0 = r0[accepted_lines]
                c0 = c0[accepted_lines]
                r1 = r1[accepted_lines]
                c1 = c1[accepted_lines]

                accums = accums[accepted_lines]
                # circle_idxs = [disk(mp, rad, shape=single_sp_seg.shape) for mp, rad in zip(mp_circles, radii)]
                # circle_sps = [torch.unique(single_sp_seg[circle_idx[0], circle_idx[1]]).long() for circle_idx in circle_idxs]
                # obj_ids = [torch.unique(single_pred[circle_idx[0], circle_idx[1]]) for circle_idx in circle_idxs]
                line_idxs = [line(R0, C0, R1, C1) for R0, C0, R1, C1 in zip(r0, c0, r1, c1)]
                line_sps = [torch.unique(single_sp_seg[line_idx[0], line_idx[1]]).long() for line_idx in line_idxs]
                obj_ids = [torch.unique(single_pred[line_idx[0], line_idx[1]]) for line_idx in line_idxs]

                for line_sp, val, obj_id in zip(line_sps, accums, obj_ids):
                    hough_score = (val - self.line_thresh) / (1 - self.line_thresh)
                    ## hough_score = torch.sigmoid(torch.tensor([8 * (hough_score - 0.5)])).item()
                    num_obj_score = 1 / max(len(obj_id), 1)
                    if num_obj_score == 1 and obj_id[0] in potential_object_ids:
                        good_obj_cnt += 1
                    edge_score[line_sp] = 0.7 * hough_score + 0.3 * num_obj_score

            score = 1.0 * (good_obj_cnt / 15) * int(good_obj_cnt > 5) + 0.0 * (1 / len(bg_object_ids))
            # score = 1 / len(bg_object_ids)
            score = np.exp((score * exp_factor)) / np.exp(np.array([exp_factor]))
            edge_score[bg_sp_ids] = score.item()
            edge_score[false_sp_ids] = 0.0
            if torch.isnan(edge_score).any() or torch.isinf(edge_score).any():
                print(Warning("NaN or inf in scores this should not happen"))
                sys.stdout.flush()
                assert False
            edges = s_dir_edges[:, :int(s_dir_edges.shape[1] / 2)]
            edge_score = edge_score[edges].max(dim=0).values
            edge_scores.append(edge_score)

        # edge_scores = torch.cat(edge_scores)
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
        
    def compute_r_theta(self, pred_seg, lines):
        origin_point = np.array([0, pred_seg.shape[1]])
        # r_dist = np.linalg.norm(np.cross(p2-p1, p1-p3))/norm(p2-p1)
        r_dists = [norm(np.cross(np.asarray(l[1]) - np.asarray(l[0]), np.asarray(l[0]) - origin_point))/norm(np.asarray(l[1])-np.asarray(l[0])) for l in lines]
        # thetas = [np.arctan(-(l[1][0] - l[0][0]) / (l[1][1] - l[0][1])) for l in lines]
        thetas = []
        for l in lines:
            if l[1][1] == l[0][1]  and l[0][1] < 0:
                thetas.append(3 * np.pi /2)
            elif l[1][1] == l[0][1] and l[0][1] >= 0:
                thetas.append(np.pi / 2)
            else:
                thetas.append(np.arctan(-(l[1][0] - l[0][0]) / (l[1][1] - l[0][1])))
        return r_dists, thetas

    def find_accums(self, r_dists, thetas, hough_res):
        return [hough_res[int(r)][int(np.rad2deg(th))] for r, th in zip(r_dists, thetas)]