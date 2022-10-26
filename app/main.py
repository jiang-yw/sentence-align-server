import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from laserembeddings import Laser
import faiss
import numpy as np
import torch
from calculate import knn, score_candidates
from typing import List

use_gpu = torch.cuda.is_available()
laser = Laser()
app = FastAPI()
output_keys = ("score", "src_text", "trg_text")


class InputDataText(BaseModel):
    src_lang: str
    src_text: str
    trg_lang: str
    trg_text: str

    class Config:
        schema_extra = {
            "example": {
                "src_text": "他在弹钢琴。\n这是一只猫。",
                "src_lang": "zh",
                "trg_text": "This is a cat.\nHe is playing the piano.",
                "trg_lang": "en",
            },
        }


class OutputData(BaseModel):
    score: str
    src_text: str
    trg_text: str

    class Config:
        schema_extra = {
            "example": [
                    {
                        "score": "1.3109767306262696",
                        "src_text": "这是一只猫。",
                        "trg_text": "This is a cat."
                    },
                    {
                        "score": "1.296414417009608",
                        "src_text": "他在弹钢琴。",
                        "trg_text": "He is playing the piano."
                    }
                ],
        }


def unique_embeddings(emb, ind):
    aux = {j: i for i, j in enumerate(ind)}
    return emb[[aux[i] for i in range(len(aux))]]


def cal(src_enc, trg_enc, src_ind_list, trg_ind_list):
    """calculate knn in both directions

    :param src_enc:
    :param trg_enc:
    :param src_ind_list:
    :param trg_ind_list:
    :return: indices, scores
    """
    src_enc = unique_embeddings(src_enc, src_ind_list)
    trg_enc = unique_embeddings(trg_enc, trg_ind_list)
    faiss.normalize_L2(src_enc)
    faiss.normalize_L2(trg_enc)

    x2y_sim, x2y_ind = knn(src_enc, trg_enc, min(trg_enc.shape[0], 4), use_gpu)
    x2y_mean = x2y_sim.mean(axis=1)
    y2x_sim, y2x_ind = knn(trg_enc, src_enc, min(src_enc.shape[0], 4), use_gpu)
    y2x_mean = y2x_sim.mean(axis=1)
    # margin function
    margin = lambda a, b: a / b

    fwd_scores = score_candidates(src_enc, trg_enc, x2y_ind, x2y_mean, y2x_mean, margin)
    bwd_scores = score_candidates(trg_enc, src_enc, y2x_ind, y2x_mean, x2y_mean, margin)
    fwd_best = x2y_ind[np.arange(src_enc.shape[0]), fwd_scores.argmax(axis=1)]
    bwd_best = y2x_ind[np.arange(trg_enc.shape[0]), bwd_scores.argmax(axis=1)]
    indices = np.stack((np.concatenate((np.arange(src_enc.shape[0]), bwd_best)),
                        np.concatenate((fwd_best, np.arange(trg_enc.shape[0])))), axis=1)
    scores = np.concatenate((fwd_scores.max(axis=1), bwd_scores.max(axis=1)))
    return indices, scores


@app.post("/align_text", response_model=List[OutputData])
async def align_text(data: InputDataText):
    """Align Sentence Text

    :param data: InputDataText
    :return: List[OutputData]
    """
    try:
        # Get text list and index list
        src_text_str = data.src_text
        trg_text_str = data.trg_text
        src_text_list = src_text_str.splitlines()
        trg_text_list = trg_text_str.splitlines()
        src_ind_list = [i for i in range(len(src_text_list))]
        trg_ind_list = [j for j in range(len(trg_text_list))]

        # Load the embeddings
        src_enc = laser.embed_sentences(src_text_list, data.src_lang)
        trg_enc = laser.embed_sentences(trg_text_list, data.trg_lang)

        # Mine bidirectional text
        indices, scores = cal(src_enc, trg_enc, src_ind_list, trg_ind_list)

        seen_src, seen_trg = set(), set()

        data = []
        for i in np.argsort(-scores):
            src_ind, trg_ind = indices[i]
            src_sent = src_text_list[src_ind]
            trg_sent = trg_text_list[trg_ind]
            score = scores[i]
            if src_ind not in seen_src and trg_ind not in seen_trg:
                seen_src.add(src_ind)
                seen_trg.add(trg_ind)
                if score > 0:
                    dic = {output_keys[0]: score, output_keys[1]: src_sent,
                           output_keys[2]: trg_sent}
                    data.append(dic)

        for i in range(len(src_ind_list)):
            if i not in seen_src:
                dic = {output_keys[0]: 0, output_keys[1]: src_text_list[i],
                       output_keys[2]: ""}
                data.append(dic)

        for j in range(len(trg_ind_list)):
            if j not in seen_trg:
                dic = {output_keys[0]: 0, output_keys[1]: "",
                       output_keys[2]: trg_text_list[j]}
                data.append(dic)

        req = jsonable_encoder(data)
        return req
    except Exception as e:
        raise HTTPException(
            status_code=500, detail="Unexpected error during processing: " + str(e)
        ) from e


if __name__ == '__main__':
    uvicorn.run('main:app', host="0.0.0.0", port=8086, reload=True)
