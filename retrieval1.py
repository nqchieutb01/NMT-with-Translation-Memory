class MatchingModel(nn.Module):
    def __init__(self, query_encoder, response_encoder, bow=False):
        super(MatchingModel, self).__init__()
        self.query_encoder = query_encoder
        self.response_encoder = response_encoder
        self.bow = bow
        if self.bow:
            self.query_bow = BOWModel(query_encoder.encoder.src_embed)
            self.response_bow = BOWModel(response_encoder.encoder.src_embed)

    def forward(self, query, response, label_smoothing=0.):
        ''' query and response: [seq_len, batch_size]
        '''
        _, bsz = query.size()
        
        q, q_src, _ = self.query_encoder(query, return_src=True)
        r, r_src, _ = self.response_encoder(response, return_src=True)
        q_src = q_src[0,:,:]
        r_src = r_src[0,:,:]
 
        scores = torch.mm(q, r.t()) # bsz x (bsz + adt)

        gold = torch.arange(bsz, device=scores.device)
        _, pred = torch.max(scores, -1)
        acc = torch.sum(torch.eq(gold, pred).float()) / bsz

        log_probs = F.log_softmax(scores, -1)
        loss, _ = label_smoothed_nll_loss(log_probs, gold, label_smoothing, sum=True)
        loss = loss / bsz

        if self.bow:
            loss_bow_q = self.query_bow(r_src, query.transpose(0, 1))
            loss_bow_r = self.response_bow(q_src, response.transpose(0, 1))
            loss = loss + loss_bow_q + loss_bow_r
        return loss, acc, bsz

    def work(self, query, response):
        ''' query and response: [seq_len x batch_size ]
        '''
        _, bsz = query.size()
        q = self.query_encoder(query)
        r = self.response_encoder(response)

        scores = torch.sum(q * r, -1)
        return scores

    def save(self, model_args, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.query_encoder.state_dict(), os.path.join(output_dir, 'query_encoder'))
        torch.save(self.response_encoder.state_dict(), os.path.join(output_dir, 'response_encoder'))
        torch.save(model_args, os.path.join(output_dir, 'args'))

    @classmethod
    def from_params(cls, vocabs, layers, embed_dim, ff_embed_dim, num_heads, dropout, output_dim, bow):
        query_encoder = ProjEncoder(vocabs['src'], layers, embed_dim, ff_embed_dim, num_heads, dropout, output_dim)
        response_encoder = ProjEncoder(vocabs['tgt'], layers, embed_dim, ff_embed_dim, num_heads, dropout, output_dim)
        model = cls(query_encoder, response_encoder, bow)
        return model
    
    @classmethod
    def from_pretrained(cls, vocabs, input_dir):
        model_args = torch.load(os.path.join(input_dir, 'args'))
        query_encoder = ProjEncoder.from_pretrained(vocabs['src'], model_args, os.path.join(input_dir, 'query_encoder'))
        response_encoder = ProjEncoder.from_pretrained(vocabs['tgt'], model_args, os.path.join(input_dir, 'response_encoder'))
        model = cls(query_encoder, response_encoder)
        return model

class MultiProjEncoder(nn.Module):
    def __init__(self, num_proj_heads, vocab, layers, embed_dim, ff_embed_dim, num_heads, dropout, output_dim):
        super(MultiProjEncoder, self).__init__()
        self.encoder = MonoEncoder(vocab, layers, embed_dim, ff_embed_dim, num_heads, dropout)
        self.proj = nn.Linear(embed_dim, num_proj_heads*output_dim)
        self.num_proj_heads = num_proj_heads
        self.output_dim = output_dim
        self.dropout = dropout

    def forward(self, input_ids, batch_first=False, return_src=False):
        if batch_first:
            input_ids = input_ids.t()
        src, src_mask = self.encoder(input_ids) 
        ret = src[0,:,:]
        ret = F.dropout(ret, p=self.dropout, training=self.training)
        ret = self.proj(ret).view(-1, self.num_proj_heads, self.output_dim).transpose(0, 1)
        ret = layer_norm(F.dropout(ret, p=self.dropout, training=self.training))
        if return_src:
            return ret, src, src_mask
        return ret

    @classmethod
    def from_pretrained_projencoder(cls, num_proj_heads, vocab, model_args, ckpt):
        model = cls(num_proj_heads, vocab, model_args.layers, model_args.embed_dim, model_args.ff_embed_dim, model_args.num_heads, model_args.dropout, model_args.output_dim)
        state_dict = torch.load(ckpt, map_location='cpu')
        model.encoder.load_state_dict({k[len('encoder.'):]:v for k,v in state_dict.items() if k.startswith('encoder.')})
        weight = state_dict['proj.weight'].repeat(num_proj_heads, 1)
        bias = state_dict['proj.bias'].repeat(num_proj_heads)
        model.proj.weight = nn.Parameter(weight)
        model.proj.bias = nn.Parameter(bias)
        return model

class ProjEncoder(nn.Module):
    def __init__(self, vocab, layers, embed_dim, ff_embed_dim, num_heads, dropout, output_dim):
        super(ProjEncoder, self).__init__()
        self.encoder = MonoEncoder(vocab, layers, embed_dim, ff_embed_dim, num_heads, dropout)
        self.proj = nn.Linear(embed_dim, output_dim)
        self.dropout = dropout
        self.output_dim = output_dim
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.constant_(self.proj.bias, 0.)

    def forward(self, input_ids, batch_first=False, return_src=False):
        if batch_first:
            input_ids = input_ids.t()
        src, src_mask = self.encoder(input_ids) 
        ret = src[0,:,:]
        ret = F.dropout(ret, p=self.dropout, training=self.training)
        ret = self.proj(ret)
        ret = layer_norm(ret)
        if return_src:
            return ret, src, src_mask
        return ret

    @classmethod
    def from_pretrained(cls, vocab, model_args, ckpt):
        model = cls(vocab, model_args.layers, model_args.embed_dim, model_args.ff_embed_dim, model_args.num_heads, model_args.dropout, model_args.output_dim)
        model.load_state_dict(torch.load(ckpt, map_location='cpu'))
        return model


def batchify(data, vocab):

    tokens = [[BOS] + x for x in data]

    token = ListsToTensor(tokens, vocab)

    return token

class DataLoader(object):
    def __init__(self, used_data, vocab, batch_size, max_seq_len=256):
        self.vocab = vocab
        self.batch_size = batch_size

        data = []
        for x in used_data:
#             x = x.split()[:max_seq_len]
            x = spm_model.encode(x, out_type=str, enable_sampling=True, alpha=ALPHA, nbest_size=-1)[:max_seq_len]
            data.append(x)
        self.data = data

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        indices = np.arange(len(self))

        cur = 0
        while cur < len(indices):
            data = [self.data[i] for i in indices[cur:cur+self.batch_size]]
            cur += self.batch_size
            yield batchify(data, self.vocab)

@torch.no_grad()
def get_features(batch_size, norm_th, vocab, model, used_data, used_ids, max_norm=None, max_norm_cf=1.0):
    vecs, ids = [], []
#     model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    model = torch.nn.DataParallel(model, device_ids=list(range(1,2)))
    model.eval()
    data_loader = DataLoader(used_data, vocab, batch_size)
    cur, tot = 0, len(used_data)
    for batch in asynchronous_load(data_loader):
        batch = move_to_device(batch, torch.device('cuda', 1)).t()
        bsz = batch.size(0)
        cur_vecs = model(batch, batch_first=True).detach().cpu().numpy()
        valid = np.linalg.norm(cur_vecs, axis=1) <= norm_th
        vecs.append(cur_vecs[valid])
        ids.append(used_ids[cur:cur+batch_size][valid])
        cur += bsz
        logger.info("%d / %d", cur, tot)
    vecs = np.concatenate(vecs, 0)
    ids = np.concatenate(ids, 0)
    out, max_norm = augment_data(vecs, max_norm, max_norm_cf)
    return out, ids, max_norm