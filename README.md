# KoBERT-CSSE

기존 Kobert 모델에 **문맥의존 철자오류 교정**을 적용하기 위해 중앙기사 데이터를 추가로 Pretrain하였고,

Tokenizer로 [Huggingface KoNLPy](https://github.com/lovit/huggingface_konlpy) 를 사용하였습니다.(정확도를 높이기 위해)



## Requirements

* Python >= 3.6

* PyTorch >= 1.7.0

* transformers >= 3.5.0
* konlpy >= 0.5.2
* tokenizers >= 0.8.1
* tqdm >= 4.46.0 
* wandb

----------------------

* MXNet >= 1.4.0

* gluonnlp >= 0.6.0

* sentencepiece >= 0.1.6

* onnxruntime >= 0.3.0

여기는 안깔아도 될듯

------------







## Reference

\- [KoBERT](https://github.com/SKTBrain/KoBERT)

\- [KoBERT-Transformers](https://github.com/monologg/KoBERT-Transformers)

\- [Huggingface Transformers](https://github.com/huggingface/transformers)

\- [Huggingface KoNLPy](https://github.com/lovit/huggingface_konlpy)

