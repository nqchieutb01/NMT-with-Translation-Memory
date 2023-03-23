import sentencepiece as spm

# spm.SentencePieceTrainer.train(input='data/kmvi/train.txt', model_prefix='spm_kv', vocab_size=8000,
#                                user_defined_symbols=["<PAD>", "<UNK>", "<BOS>", "<EOS>"], character_coverage=1.0,
#                                model_type="bpe"
#                                )

s = spm.SentencePieceProcessor(model_file='spm_kv.model')
# s = spm.SentencePieceProcessor(model_file='sentencepiece.bpe.model')
tmp = "Hello, toi la Chieu"
tmp = s.encode(tmp,out_type=str)
print(tmp)
print(s.decode(tmp))
# print(s.get_piece_size())
# print(s.piece_to_id('<vi_VN>'))
# print(s.decode(tmp[0]))
# print(s.id_to_piece(7))
