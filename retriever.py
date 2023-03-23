import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import math
import os, time, random, logging

from transformer import Transformer, SinusoidalPositionalEmbedding, Embedding
from utils import move_to_device, asynchronous_load
from module import label_smoothed_nll_loss, layer_norm, MonoEncoder
from mips import MIPS, augment_query, augment_data, l2_to_ip
from data import BOS, EOS, ListsToTensor, _back_to_txt_for_check
from config import CHECKPOINT_XLMR,SPM_MODEL,ALPHA,ADD,MMR,FIXED,GPU_DEVICE
logger = logging.getLogger(__name__)
import sentencepiece as spm
import warnings
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
spm_model = spm.SentencePieceProcessor(model_file=SPM_MODEL)



def _MMR(embdistrib , X, beta, N):
    """
    Core method using Maximal Marginal Relevance in charge to return the top-N candidates
    :param embdistrib: embdistrib: embedding distributor see @EmbeddingDistributor
    :param text_obj: Input text representation see @InputTextObj
    :param candidates: list of candidates (string)
    :param X: numpy array with the embedding of each candidate in each row
    :param beta: hyperparameter beta for MMR (control tradeoff between informativeness and diversity)
    :param N: number of candidates to extract
    :param use_filtered: if true filter the text by keeping only candidate word before computing the doc embedding
    :return: A tuple with 3 elements :
    1)list of the top-N candidates (or less if there are not enough candidates) (list of string)
    2)list of associated relevance scores (list of float)
    3)list containing for each keyphrase a list of alias (list of list of string)
    """

    # N = min(N, len(candidates))
    # doc_embedd = extract_doc_embedding(embdistrib, text_obj, use_filtered)  # Extract doc embedding
    doc_embedd = embdistrib
    doc_sim = cosine_similarity(X, doc_embedd.reshape(1, -1))

    doc_sim_norm = doc_sim/np.max(doc_sim)
    doc_sim_norm = 0.5 + (doc_sim_norm - np.average(doc_sim_norm)) / np.std(doc_sim_norm)
    
    sim_between = cosine_similarity(X)
    np.fill_diagonal(sim_between, np.NaN)

    sim_between_norm = sim_between/np.nanmax(sim_between, axis=0)
#     print(np.nanstd(sim_between_norm, axis=0))
    sim_between_norm = \
        0.5 + (sim_between_norm - np.nanmean(sim_between_norm, axis=0)) / np.nanstd(sim_between_norm, axis=0)
    selected_candidates = []
    # unselected_candidates = [c for c in range(len(candidates))]
    unselected_candidates = [c for c in range(X.shape[0])]

    j = np.argmax(doc_sim)
    selected_candidates.append(j)
    unselected_candidates.remove(j)

    for _ in range(N - 1):
        selec_array = np.array(selected_candidates)
        unselec_array = np.array(unselected_candidates)

        distance_to_doc = doc_sim_norm[unselec_array, :]
        dist_between = sim_between_norm[unselec_array][:, selec_array]
        if dist_between.ndim == 1:
            dist_between = dist_between[:, np.newaxis]
        j = np.argmax(beta * distance_to_doc - (1 - beta) * np.max(dist_between, axis=1).reshape(-1, 1))
        item_idx = unselected_candidates[j]
        selected_candidates.append(item_idx)
        unselected_candidates.remove(item_idx)

    # Not using normalized version of doc_sim for computing relevance
    # relevance_list = max_normalization(doc_sim[selected_candidates]).tolist()
    # aliases_list = get_aliases(sim_between[selected_candidates, :], candidates, alias_threshold)

    return selected_candidates


class Retriever(nn.Module):
    def __init__(self, vocabs, model, mips, mips_max_norm, mem_pool, mem_feat_or_feat_maker, num_heads, topk, gpuid):
        super(Retriever, self).__init__()
        self.model = model
        self.mem_pool = mem_pool
        self.mem_feat_or_feat_maker = mem_feat_or_feat_maker
        self.num_heads = num_heads
        self.topk = topk
        self.vocabs = vocabs
        self.gpuid = gpuid
        self.mips = mips
        self.mips.index.make_direct_map()
#         if self.gpuid >= 0:
#         self.mips.to_gpu(gpuid=0)
        self.mips_max_norm = mips_max_norm

    @classmethod
    def from_pretrained(cls, num_heads, vocabs, input_dir, nprobe, topk, gpuid, use_response_encoder=False):
        model_args = torch.load(os.path.join(input_dir, 'args'))
        model = MultiProjEncoder.from_pretrained_projencoder(num_heads, vocabs['src'], model_args, os.path.join(input_dir, 'query_encoder'))
#         mem_pool = [line.strip().split() for line in open(os.path.join(input_dir, 'candidates.txt')).readlines()]
        mem_pool = [spm_model.encode(line.strip(), out_type=str, enable_sampling=True, alpha=ALPHA, nbest_size=-1) for line in open(os.path.join(input_dir, f'candidates{ADD}.txt')).readlines()]
        if use_response_encoder:
            mem_feat_or_feat_maker = ProjEncoder.from_pretrained(vocabs['tgt'], model_args, os.path.join(input_dir, 'response_encoder'))
        else:
            mem_feat_or_feat_maker = torch.load(os.path.join(input_dir, f'feat{ADD}.pt'))
        
        mips = MIPS.from_built(os.path.join(input_dir, f'mips_index{ADD}'), nprobe=nprobe)
        mips_max_norm = torch.load(os.path.join(input_dir, f'max_norm{ADD}.pt'))
        retriever = cls(vocabs, model, mips, mips_max_norm, mem_pool, mem_feat_or_feat_maker, num_heads, topk, gpuid)
        return retriever

    def drop_index(self):
        self.mips.reset()
        self.mips = None
        self.mips_max_norm = None

    def update_index(self, index_dir, nprobe):
        self.mips = MIPS.from_built(os.path.join(index_dir, 'mips_index'), nprobe=nprobe)
        if self.gpuid >= 0:
            self.mips.to_gpu(gpuid=self.gpuid)
        self.mips_max_norm = torch.load(os.path.join(index_dir, 'max_norm.pt'))

    def rebuild_index(self, index_dir, batch_size=2048, add_every=1000000, index_type='IVF1024_HNSW32,SQ8', norm_th=999, max_training_instances=1000000, max_norm_cf=1.0, nprobe=64, efSearch=128):
        if not os.path.exists(index_dir):
            os.mkdir(index_dir)
        max_norm = None
        data = [ [' '.join(x), i] for i, x in enumerate(self.mem_pool) ]
        random.shuffle(data)
        used_data = [x[0] for x in data[:max_training_instances]]
        used_ids = np.array([x[1] for x in data[:max_training_instances]])
        logger.info('Computing feature for training')
        used_data, used_ids, max_norm = get_features(batch_size, norm_th, self.vocabs['tgt'], self.mem_feat_or_feat_maker, used_data, used_ids, max_norm_cf=max_norm_cf)
        torch.cuda.empty_cache()
        logger.info('Using %d instances for training', used_data.shape[0])
        mips = MIPS(self.model.output_dim+1, index_type, efSearch=efSearch, nprobe=nprobe) 
        mips.to_gpu()
        mips.train(used_data)
        mips.to_cpu()
        mips.add_with_ids(used_data, used_ids)
        data = data[max_training_instances:]
        torch.save(max_norm, os.path.join(index_dir, 'max_norm.pt'))
        
        cur = 0
        while cur < len(data):
            used_data = [x[0] for x in data[cur:cur+add_every]]
            used_ids = np.array([x[1] for x in data[cur:cur+add_every]])
            cur += add_every
            logger.info('Computing feature for indexing')
            used_data, used_ids, _ = get_features(batch_size, norm_th, vocab, self.mem_feat_or_feat_maker, used_data, used_ids, max_norm)
            logger.info('Adding %d instances to index', used_data.shape[0])
            mips.add_with_ids(used_data, used_ids)
        mips.save(os.path.join(index_dir, 'mips_index'))

    def work(self, inp, allow_hit):
        src_tokens = inp['src_tokens']
        src_feat, src, src_mask = self.model(src_tokens, return_src=True)
        num_heads, bsz, dim = src_feat.size()
        assert num_heads == self.num_heads
        topk = self.topk
        vecsq = src_feat.reshape(num_heads * bsz, -1).detach().cpu().numpy() 
        #retrieval_start = time.time()
        vecsq = augment_query(vecsq)
        if MMR:
            D, I = self.mips.search(vecsq, 21)
        else:
            D, I = self.mips.search(vecsq, topk+1)
        D = l2_to_ip(D, vecsq, self.mips_max_norm) / (self.mips_max_norm * self.mips_max_norm)
        # I, D: (bsz * num_heads x (topk + 1) )
        indices = torch.zeros(topk, num_heads, bsz, dtype=torch.long)
        for i, (Ii, Di) in enumerate(zip(I, D)):
            bid, hid = i % bsz, i // bsz
            tmp_list = []
            tmp = None
            for pred, _ in zip(Ii, Di):
                if allow_hit or self.mem_pool[pred]!=inp['tgt_raw_sents'][bid]:
                    tmp_list.append(pred)
                    tmp = pred   
            if MMR:
                tmp_list = tmp_list[:20]
                if len(tmp_list) < 20:
                    print(inp['tgt_raw_sents'][bid])
                X = []
                for o in tmp_list:
                    X.append(self.mips.reconstruct(int(o)))
                X = np.array(X)
                tmp_list_MMR = _MMR(vecsq[bid] , X, 0.5, topk)
                list_MMR = np.array(tmp_list)[tmp_list_MMR]
            else: 
                tmp_list = tmp_list[:topk]
            if len(tmp_list) < topk:
                print(inp['tgt_raw_sents'][bid])
                print(tmp_list)
                while len(tmp_list) != topk:
                    tmp_list.append(tmp)
            if MMR:
                tmp_list = list_MMR
            assert len(tmp_list) == topk
#             print(list_MMR)
#             print('-'*100)
            indices[:, hid, bid] = torch.tensor(tmp_list)
            
        #retrieval_cost = time.time() - retrieval_start
        #print ('retrieval_cost', retrieval_cost)
        # convert to tensors:
        # all_mem_tokens -> seq_len x ( topk * num_heads * bsz )
        # all_mem_feats -> topk * num_heads * bsz x dim
        all_mem_tokens = []
        for idx in indices.view(-1).tolist():
            #TODO self.mem_pool[idx] +[EOS]
            all_mem_tokens.append([BOS] + self.mem_pool[idx])
        all_mem_tokens = ListsToTensor(all_mem_tokens, self.vocabs['tgt'])
        
        # to avoid GPU OOM issue, truncate the mem to the max. length of 1.5 x src_tokens
        max_mem_len = int(1.5 * src_tokens.shape[0])
        all_mem_tokens = move_to_device(all_mem_tokens[:max_mem_len,:], inp['src_tokens'].device)
       
        if torch.is_tensor(self.mem_feat_or_feat_maker):
            all_mem_feats = self.mem_feat_or_feat_maker[indices].to(src_feat.device)
        else:
            all_mem_feats = self.mem_feat_or_feat_maker(all_mem_tokens).view(topk, num_heads, bsz, dim)

        # all_mem_scores -> topk x num_heads x bsz
        all_mem_scores = torch.sum(src_feat.unsqueeze(0) * all_mem_feats, dim=-1) / (self.mips_max_norm ** 2)
        if FIXED:
            all_mem_scores = all_mem_scores/ all_mem_scores
        mem_ret = {}
        indices = indices.view(-1, bsz).transpose(0, 1).tolist()
        mem_ret['retrieval_raw_sents'] = [ [self.mem_pool[idx] for idx in ind] for ind in indices]
        mem_ret['all_mem_tokens'] = all_mem_tokens
        mem_ret['all_mem_scores'] = all_mem_scores
        return src, src_mask, mem_ret

class BOWModel(nn.Module):
    def __init__(self, tgt_embed):
        ## bag of words autoencoder
        super(BOWModel, self).__init__()
        vocab_size, embed_dim = tgt_embed.weight.shape
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.output_projection = nn.Linear(
                embed_dim,
                vocab_size,
                bias=False,
        )
        self.output_projection.weight = tgt_embed.weight
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.constant_(self.proj.bias, 0.)

    def forward(self, outs, label):
        # bow loss
        bsz, seq_len = label.shape
        label_mask = torch.le(label, 3) # except for PAD UNK BOS EOS
        logits = self.output_projection(self.proj(outs))
        lprobs = F.log_softmax(logits, dim=-1)
        #bsz x vocab
        loss = torch.gather(-lprobs, -1, label).masked_fill(label_mask, 0.)
        loss = loss.sum(dim=-1).mean()

        return loss

