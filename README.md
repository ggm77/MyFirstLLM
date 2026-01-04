# MyFirstLLM

> GPT 스타일의 Decoder-only 트랜스포머 모델입니다.

트랜스포머 모델을 처음부터 구현해보기 위해서 작성되었습니다.
최초 코드는 구글 코랩에서 작성 되었고, 로컬로 이관 되었습니다.

---

### 로컬 개발환경
- Python: 3.11.4
- PyTorch: 2.7.1
- CUDA: 12.8
- datasets: 4.0.0
- transformers: 4.57.3

---

### 데이터셋
영어 데이터셋은 <a href="https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu">FineWeb-Edu</a>, 한글 데이터셋은 <a href="https://huggingface.co/datasets/HuggingFaceFW/fineweb-2">FineWeb2</a>를 사용했습니다.

영어 검증 데이터셋은 <a href="https://huggingface.co/datasets/Salesforce/wikitext">wikitext</a>, 한글 데이터셋은 <a href="https://huggingface.co/datasets/wikimedia/wikipedia">wikipedia</a>를 사용했습니다.

---

### 토크나이저

토크나이저는 GPT2 토크나이저 기반으로 위 데이터셋을 이용하여 추가 학습을 진행했습니다.
