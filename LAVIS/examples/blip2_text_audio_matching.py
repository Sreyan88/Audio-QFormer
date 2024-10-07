import torch
import os

from PIL import Image
import json
from lavis.models import load_model, load_model_and_preprocess
from lavis.processors import load_processor
from lavis.models.blip2_models.blip2 import compute_sim_matrix

from lavis.datasets.datasets.audio_captioning_datasets import (
    AudioSetDataset,
    AudioSetEvalDataset,
    AudioSetInstructDataset,
    AudioCapsDataset,
    AudioCapsEvalDataset,
    AudioCapsInstructDataset,
    ClothoV2Dataset,
    ClothoV2InstructDataset,
    ClothoV2EvalDataset,
    AudioLanguagePretrainDataset_CK,
    AudioLanguagePretrainDataset,
    AudioLanguagePretrainEvalDataset,
    AudioLanguagePretrainInstructDataset
)

import numpy as np

def _report_metrics(scores_i2t, scores_t2i, txt2img, img2txt):

    # Images->Text
    ranks = np.zeros(scores_i2t.shape[0])
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        print(inds)
        # Score
        rank = 1e20
        print(img2txt[index])
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    # Text->Images
    ranks = np.zeros(scores_t2i.shape[0])

    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    agg_metrics = (tr1 + tr5 + tr10) / 3

    eval_result = {
        "txt_r1": tr1,
        "txt_r5": tr5,
        "txt_r10": tr10,
        "txt_r_mean": tr_mean,
        "img_r1": ir1,
        "img_r5": ir5,
        "img_r10": ir10,
        "img_r_mean": ir_mean,
        "r_mean": r_mean,
        "agg_metrics": agg_metrics,
    }
    with open(
        "/fs/nexus-projects/brain_project/acl_sk_24/LAVIS/lavis/output/BLIP2/pretrain_retrieval.json", "a"
    ) as f:
        f.write(json.dumps(eval_result) + "\n")
    return eval_result

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

_, _, text_processors = load_model_and_preprocess("blip2_image_text_matching", "pretrain", device=device, is_eval=True)

model = load_model(name="blip2_image_text_matching", model_type="pretrain", checkpoint="/fs/nexus-projects/brain_project/acl_sk_24/LAVIS/lavis/output/BLIP2/Pretrain_stage1/20240531130/checkpoint_99.pth", device=device, is_eval=True)

dataloader = AudioLanguagePretrainDataset_CK("/fs/nexus-projects/brain_project/CLAP/test_clotho_rt_ck.csv", text_processors)

scores_i2t, scores_t2i = compute_sim_matrix(model=model, data_loader=dataloader, k_test=4)

print(_report_metrics(scores_i2t, scores_t2i, dataloader.txt2img, dataloader.img2txt))