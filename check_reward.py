import os
from glob import glob

import imageio
import numpy as np
import torch

from elf.segmentation import compute_rag
from graphs import get_edge_indices, collate_edges
from rag_utils import find_dense_subgraphs

from lines_reward import HoughLinesReward
from circles_reward import HoughCirclesReward


def compute_reward(super_pixel, pred_img, s_subgraph, RewardFunction):
    super_pixelT = torch.from_numpy(super_pixel)
    pred_imgT = torch.from_numpy(pred_img)
    graph = compute_rag(super_pixelT)
    edge_ids = graph.uvIds().T
    edge_ids = edge_ids.astype(np.int64)
    edge_ids = torch.from_numpy(edge_ids)
    edge_ids = [edge_ids]
    dir_edge_ids = [torch.cat([
        _edge_ids,
        torch.stack([_edge_ids[1], _edge_ids[0]], dim=0)
    ], dim=1) for _edge_ids in edge_ids]
    _subgraphs, _sep_subgraphs = find_dense_subgraphs(
        [eids.transpose(0, 1).cpu().numpy() for eids in edge_ids], s_subgraph
    )

    bs = 1
    _subgraphs = [torch.from_numpy(sg.astype(np.int64)).permute(2, 0, 1) for sg in _subgraphs]
    _sep_subgraphs = [torch.from_numpy(sg.astype(np.int64)).permute(2, 0, 1) for sg in _sep_subgraphs]
    _edge_ids, (n_offs, e_offs) = collate_edges(edge_ids)
    subgraphs, sep_subgraphs = [], []

    for i in range(len(s_subgraph)):
        subgraphs.append(
            torch.cat([sg + n_offs[i] for i, sg in enumerate(_subgraphs[i*bs:(i+1)*bs])], -2).flatten(-2, -1)
        )
        sep_subgraphs.append(torch.cat(_sep_subgraphs[i*bs:(i+1)*bs], -2).flatten(-2, -1))

    SI = get_edge_indices(_edge_ids, subgraphs)

    reward_function = RewardFunction(s_subgraph)

    pred_imgT = torch.unsqueeze(pred_imgT, 0)
    super_pixelT = torch.unsqueeze(super_pixelT, 0)

    reward = reward_function(prediction_segmentation=pred_imgT,
                             gt=None, dir_edges=dir_edge_ids,
                             superpixel_segmentation=super_pixelT,
                             node_feats=None, actions=None,
                             subgraph_indices=SI, sg_gt_edges=None)

    return reward


def compare_rewards(folder, RewardFunction):
    print("Compare rewards for:", folder)
    s_subgraph = [4]
    sp = imageio.imread(os.path.join(folder, "superpixel.tif"))
    merged_paths = glob(os.path.join(folder, "merged*.tif"))
    merged_paths.sort()
    merged_ims = [imageio.imread(pp) for pp in merged_paths]

    reward0 = compute_reward(sp, sp, s_subgraph, RewardFunction)
    print("The reward of the superpixels is:", reward0)

    for pp, im in zip(merged_paths, merged_ims):
        reward = compute_reward(sp, im, s_subgraph, RewardFunction)
        print("The reward of", pp, "is:", reward)


def main():
    compare_rewards("./line_data", HoughLinesReward)
    print()
    print()
    # compare_rewards("./circle_data", HoughCirclesReward)


if __name__ == "__main__":
    main()
