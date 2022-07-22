import sys
from reward_abc import RewardFunctionAbc
# from skimage.measure import approximate_polygon, find_contours
# from skimage.draw import polygon_perimeter, line
from skimage.transform import hough_line, probabilistic_hough_line
# from skimage.transform import hough_line_peaks
import torch
from skimage.draw import line
from numpy.linalg import norm
import numpy as np
import matplotlib.pyplot as plt

# from matplotlib import cm


def plot_debug(single_pred, edge_image, hough_pred_lines):
    # generate figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    ax = axes.ravel()
    # Detect two radii

    ax[0].imshow(single_pred)
    ax[0].set_title('Input image')
    ax[1].imshow(edge_image)
    ax[1].set_title('edges')

    # plot detected lines
    ax[2].imshow(edge_image)
    for ln in hough_pred_lines:
        p0, p1 = ln
        ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))
    ax[2].set_xlim((0, single_pred.shape[1]))
    ax[2].set_ylim((single_pred.shape[0], 0))
    ax[2].set_title('Probabilistic Hough')

    for a4 in ax:
        a4.set_axis_off()

    plt.tight_layout()
    fig.savefig('debug-hough-lines.png')


class HoughLinesReward(RewardFunctionAbc):

    def __init__(self, s_subgraph, *args, **kwargs):
        self.max_p = torch.nn.MaxPool2d(3, padding=1, stride=1)
        self.s_subgraph = s_subgraph

        self.line_thresh = 10
        self.range_rad = [10, 20]  # :?????????
        self.range_num = [20, 20]
        # super().__init__(s_subgraph, *args, **kwargs)

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

            # compute the prediction ids and sizes of the current object predictions
            pred_ids, label_masses = torch.unique(single_pred, return_counts=True)
            # get the ids of bg objects, false positives and foreground objects
            bg_pred_ids = pred_ids[label_masses >= bg_size_threshold]
            false_pred_ids = pred_ids[label_masses <= false_obj_size_threshold]
            fg_pred_ids = pred_ids[
                (label_masses > false_obj_size_threshold).logical_and(label_masses < bg_size_threshold)
            ]

            # get the ids of the superpixels corresponding to bg and false objects
            bg_sp_ids = torch.unique(
                single_sp_seg[torch.isin(single_pred, bg_pred_ids)]
            )
            false_sp_ids = torch.unique(
                single_sp_seg[torch.isin(single_pred, false_pred_ids)]
            )

            # FIXME this doesn't make much sense, we have to compute the hough scores for the individual
            # predicted objects, and not for the whole foreground!!
            # (otherwise it doesn't make a difference if an object is split up into many objects or not)
            # get a binary image of all foreground objects
            # NOTE: in the circle example it looks like the hough trafo is computed for the outlines of the
            # circle. I have no idea why that is done instead of doing it for the actual circles
            edge_image = torch.isin(single_pred, fg_pred_ids).detach().cpu().numpy().squeeze().astype("float32")

            # calculations for hough line reward
            tested_angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
            hough_res, angles, distc = hough_line(edge_image, theta=tested_angles)
            # accums, angles, distcs = hough_line_peaks(hough_res, angles, distc, num_peaks=self.range_num[1])
            hough_pred_lines = probabilistic_hough_line(edge_image, line_length=20, line_gap=10)
            r_dists, thetas = self.compute_r_theta(prediction_segmentation, hough_pred_lines)
            accums = self.find_accums(r_dists, thetas, hough_res)

            # for debugging
            plot_debug(single_pred, edge_image, hough_pred_lines)

            r0 = []
            c0 = []
            r1 = []
            c1 = []
            for lineـ in hough_pred_lines:
                p0, p1 = lineـ

                r0.append(p0[0])
                r1.append(p1[0])
                c0.append(p0[1])
                c1.append(p1[1])

            r0 = np.array(r0)
            c0 = np.array(c0)
            r1 = np.array(r1)
            c1 = np.array(c1)

            accums = np.array(accums)
            accepted_lines = accums > self.line_thresh
            good_obj_cnt = 0

            if any(accepted_lines):
                print("we accepted", len(accepted_lines), "lines")

                r0 = r0[accepted_lines]
                c0 = c0[accepted_lines]
                r1 = r1[accepted_lines]
                c1 = c1[accepted_lines]

                accums = accums[accepted_lines]

                line_idxs = [line(R0, C0, R1, C1) for R0, C0, R1, C1 in zip(r0, c0, r1, c1)]
                line_sps = [torch.unique(single_sp_seg[line_idx[0], line_idx[1]]).long() for line_idx in line_idxs]
                obj_ids = [torch.unique(single_pred[line_idx[0], line_idx[1]]) for line_idx in line_idxs]

                for line_sp, val, obj_id in zip(line_sps, accums, obj_ids):
                    hough_score = (val - self.line_thresh) / (1 - self.line_thresh)
                    # hough_score = torch.sigmoid(torch.tensor([8 * (hough_score - 0.5)])).item()
                    # num_obj_score = 1 / max(len(obj_id), 1)
                    # if num_obj_score == 1 and obj_id[0] in potential_object_ids:
                    #     good_obj_cnt += 1
                    # edge_score[line_sp] = 0.7 * hough_score + 0.3 * num_obj_score
                    # if num_obj_score == 1 and obj_id[0] in potential_object_ids:
                    if obj_id[0] in fg_pred_ids:
                        good_obj_cnt += 1
                    # edge_score[line_sp] = 0.7 * hough_score + 0.3 * num_obj_score
                    edge_score[line_sp] = hough_score

            score = 1.0 * (good_obj_cnt / 15) * int(good_obj_cnt > 5) + 0.0 * (1 / len(bg_pred_ids))
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
        else:
            print("No lines were accepted!!!!!")

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
        r_dists = [norm(np.cross(np.asarray(ll[1]) - np.asarray(ll[0]),
                                 np.asarray(ll[0]) - origin_point)) / norm(np.asarray(ll[1])-np.asarray(ll[0]))
                   for ll in lines]
        # thetas = [np.arctan(-(l[1][0] - l[0][0]) / (l[1][1] - l[0][1])) for l in lines]
        thetas = []
        for ll in lines:
            if ll[1][1] == ll[0][1] and ll[0][1] < 0:
                thetas.append(3 * np.pi / 2)
            elif ll[1][1] == ll[0][1] and ll[0][1] >= 0:
                thetas.append(np.pi / 2)
            else:
                thetas.append(np.arctan(-(ll[1][0] - ll[0][0]) / (ll[1][1] - ll[0][1])))
        return r_dists, thetas

    def find_accums(self, r_dists, thetas, hough_res):
        return [hough_res[int(r)][int(np.rad2deg(th))] for r, th in zip(r_dists, thetas)]
