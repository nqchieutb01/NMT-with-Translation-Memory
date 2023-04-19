# Neural Machine Translation with Monolingual Translation Memory

## Environment 

The code is written and tested with the following packages:

- transformers==2.11.0
- faiss-gpu==1.6.1
- torch==1.5.1+cu101

# Hướng dẫn chạy


0. do `export MTPATH=đường_dẫn_đến_thư_mục_data` 
1. Sinh ra spiecemodel: `python3 spm.py` 
2. Huấn luyện trước mô hình retrieval: `sh scripts/envi/pretrain.sh`

3. Xây dựng index cho dữ liệu nhớ: `sh scripts/envi/build_index.sh` (`input_file` chứa các câu trong tập ngôn ngữ đích. Để chạy được, cần phải xóa bỏ hết các câu trùng nhau trong `input_file`)
4. training: `sh scripts/envi/train.multihead.dynamic.sh` (đây là phiên bản cố định $E_{tgt}$) 
    Phiên bản không cố định E_{tgt}:  `sh scripts/esen/train.fully.dynamic.sh`
5. testing:   `sh scripts/work.sh`

Các scripts khác:

Transformer, see `sh scripts/train.vanilla.sh` .

BM 25, see `sh scripts/train.bm25.sh`.

## Citation

```
@inproceedings{cai-etal-2021-neural,
    title = "Neural Machine Translation with Monolingual Translation Memory",
    author = "Cai, Deng  and
      Wang, Yan  and
      Li, Huayang  and
      Lam, Wai  and
      Liu, Lemao",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.567",
    doi = "10.18653/v1/2021.acl-long.567",
    pages = "7307--7318",
    abstract = "Prior work has proved that Translation Memory (TM) can boost the performance of Neural Machine Translation (NMT). In contrast to existing work that uses bilingual corpus as TM and employs source-side similarity search for memory retrieval, we propose a new framework that uses monolingual memory and performs learnable memory retrieval in a cross-lingual manner. Our framework has unique advantages. First, the cross-lingual memory retriever allows abundant monolingual data to be TM. Second, the memory retriever and NMT model can be jointly optimized for the ultimate translation goal. Experiments show that the proposed method obtains substantial improvements. Remarkably, it even outperforms strong TM-augmented NMT baselines using bilingual TM. Owning to the ability to leverage monolingual data, our model also demonstrates effectiveness in low-resource and domain adaptation scenarios.",
}
```
