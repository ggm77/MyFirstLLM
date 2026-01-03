import os
from transformers import AutoTokenizer
from datasets import load_dataset, interleave_datasets
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
BASE_MODEL = "gpt2"
VOCAB_SIZE = 32000
SEED = 20260103
SAVE_PATH = BASE_DIR / "MyFirstTokenizer"

def main():
    print("데이터셋 로드 중...")
    en_data = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    kr_data = load_dataset("HuggingFaceFW/fineweb-2", name="kor_Hang", split="train", streaming=True)
    print("데이터셋 로드 완료")

    dataset = interleave_datasets(
        [en_data, kr_data],
        probabilities=[0.7, 0.3],
        seed = SEED,
        stopping_strategy="first_exhausted"
    )

    def batch_iterator(batch_size=1000):
        for i, example in enumerate(dataset):
            yield example["text"]
            if i >= 100000:
                break

    print("새로운 단어 분포 학습 시작...")
    old_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    new_tokenizer = old_tokenizer.train_new_from_iterator(
        batch_iterator(),
        vocab_size=VOCAB_SIZE
    )
    print("토크나이저 학습 완료")

    if new_tokenizer.pad_token is None:
        new_tokenizer.pad_token = new_tokenizer.eos_token

    new_tokenizer.save_pretrained(SAVE_PATH)
    print("토크나이저 저장 완료")

    print("-" * 30)
    print(f"Vocabulary Size: {len(new_tokenizer)}")
    print(f"Pad Token: {new_tokenizer.pad_token} (ID: {new_tokenizer.pad_token_id})")
    print(f"EOS Token: {new_tokenizer.eos_token} (ID: {new_tokenizer.eos_token_id})")

    sample_text = "안녕하세요, 토크나이저 테스트 문장입니다. Hello world!"
    tokens = new_tokenizer.tokenize(sample_text)
    ids = new_tokenizer.encode(sample_text)

    print(f"테스트 문장: {sample_text}")
    print(f"토큰화 결과: {tokens}")
    print(f"인코딩 결과(ID): {ids}")
    print("-" * 30)

if __name__ == "__main__":
    main()