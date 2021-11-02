from huggingface_konlpy_master.huggingface_konlpy.tokenizers_konlpy import KoNLPyPreTokenizer
from huggingface_konlpy_master.huggingface_konlpy.transformers_konlpy import KoNLPyPretokBertTokenizer
from konlpy.tag import Komoran

from kobert_tokenizer.tokenization_kobert import KoBertTokenizer


"""
# Tokenizer 정의
"""
komoran_pretok = KoNLPyPreTokenizer(Komoran())

komoran_pretok_berttokenizer = KoNLPyPretokBertTokenizer(
    konlpy_pretok = komoran_pretok,
    vocab_file = 'huggingface_konlpy_master/tokenizers/KomoranBertWordPieceTokenizer/vocab.txt')

tokenizer = komoran_pretok_berttokenizer

# kobert tokenizer 사용할 때
# tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert') 


"""
# 언어모델 정의
"""
from transformers import BertForPreTraining, BertModel
model = BertModel.from_pretrained('monologg/kobert')

config = model.config
model = BertForPreTraining(config = config)



"""
# 데이터셋 정의
"""
from TextDatasetForNextSentencePrediction import TextDatasetForNextSentencePrediction
from transformers import DataCollatorForLanguageModeling
# import datasets
file_path = "../fasttext_data/fasttext_model/data/merged_all.txt"

print("데이터 읽기 시작.")
dataset = TextDatasetForNextSentencePrediction(
    tokenizer=tokenizer,
    file_path=file_path,
    block_size=128,
    overwrite_cache=False,
    short_seq_probability=0.1,
    nsp_probability=0.5,
)
# dataset = datasets.load_dataset(file_path)
print("데이터 읽기 완료.")
data_collator = DataCollatorForLanguageModeling(    # [MASK] 를 씌우는 것은 저희가 구현하지 않아도 됩니다! :-)
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

"""
# trainer 정의
"""
from transformers import Trainer, TrainingArguments

data_collator = DataCollatorForLanguageModeling(    # [MASK] 를 씌우는 것은 저희가 구현하지 않아도 됩니다! :-)
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
   output_dir='model_output',
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=32,
    save_steps=1000, # step 수마다 모델을 저장
    save_total_limit=2, # 마지막 두 모델 빼고 과거 모델은 삭제
    logging_steps=100
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)


"""
# train 시작 및 만든 모델 저장.
"""
print("start train~!!")
trainer.train()
print("train complete")
trainer.save_model("./KoBert_csse")
print("All complete")