"""
# 만든 모델 불러와서 확인
"""
from transformers import BertForMaskedLM, pipeline

import torch
from transformers.utils.dummy_pt_objects import BertForPreTraining, BertModel

from huggingface_konlpy_master.huggingface_konlpy.tokenizers_konlpy import KoNLPyPreTokenizer
from huggingface_konlpy_master.huggingface_konlpy.transformers_konlpy import KoNLPyPretokBertTokenizer
from konlpy.tag import Komoran

from kobert_tokenizer.tokenization_kobert import KoBertTokenizer

"""
# Tokenizer 정의
"""
komoran_pretok = KoNLPyPreTokenizer(Komoran())

tokenizer = KoNLPyPretokBertTokenizer(
    konlpy_pretok = komoran_pretok,
    vocab_file = 'huggingface_konlpy_master/tokenizers/KomoranBertWordPieceTokenizer/vocab.txt')

tokenizer.add_special_tokens({'mask_token': '[MASK]'})
# tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
my_model = BertForMaskedLM.from_pretrained('KoBert_csse')


# 마스킹된 token을 가지는 시퀀스를 정의하고 단어 대신 tokenizer.mask_token배치
sequence = f"기사는 분류상 {tokenizer.mask_token} 이지만 귀족 중에는 최하급 귀족에 속하며 준귀족적인 성격을 띤다."

# 해당 시퀀스를 ID목록으로 인코딩(파이토치용으로) 해당 목록에서 mask된 token의 position을 찾는다.
input = tokenizer.encode(sequence, return_tensors="pt")
mask_token_index = torch.where(input == tokenizer.mask_token_id)[1]

# 마스크 token의 인덱스에서 예측값을 검색(이 tensor는 어휘와 동일한 크기를 가지며 값은 각 token에 부여된 점수. 모델은 해당 컨텍스트에서 가능성이 있다고 간주되는 token에 더 높은 점수부여)
token_logits = my_model(input)[0]
mask_token_logits = token_logits[0, mask_token_index, :]

# topk 메서드를 사용하여 상위 5개 token을 검색
top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

# mask_token을 위 상위 5개 token으로 교체하고 결과 print
for token in top_5_tokens:
    print(sequence.replace(tokenizer.mask_token, tokenizer.decode([token])))

# nlp_fill = pipeline('fill-mask', top_k=5, model=my_model, tokenizer=tokenizer)
# nlp_fill(sequence)

