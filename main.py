import torch
import os
from torch.utils.data import DataLoader
from datasets import load_dataset, interleave_datasets
from transformers import AutoTokenizer
import torch.nn as nn
import math
import matplotlib.pyplot as plt
from datetime import datetime
import pytz
from pathlib import Path

from model.myFirstLLM import MyFirstLLM

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_NAME = "MyFirstLLM.pt"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / CHECKPOINT_NAME
TOKENIZER_PATH = BASE_DIR / "tokenizer" / "MyFirstTokenizer"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

# 패딩 토큰이 없는 경우 예외
if tokenizer.pad_token is None:
    raise ValueError("패팅 토큰이 없습니다.")

criterion = nn.CrossEntropyLoss()

VOCAB_SIZE = len(tokenizer) # 어휘 사전 크기
D_MODEL = 512 # 임베딩 차원
N_HEAD = 8 # 멀티 헤드 어텐션 개수
NUM_LAYERS = 6 # 디코더 개수
MAX_LEN = 256 # 입력 문자열 최대 길이

EPOCHS = 1
BATCH_SIZE = 8
SEED = 20260103

kst = pytz.timezone('Asia/Seoul')

def get_device():
  if torch.cuda.is_available():
    return torch.device('cuda')
  elif torch.backends.mps.is_available():
    return torch.device('mps')
  else:
    return torch.device('cpu')

def load_checkpoint(device, checkpoint_name=CHECKPOINT_NAME):
  checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
  if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"--- 체크포인트 발견: Step {checkpoint['step']}부터 재시작 ---")
    return checkpoint['step'], checkpoint['model_state_dict'], checkpoint['optimizer_state_dict'], checkpoint['dataset_state_dict']
  return 0, None, None, None

# 데이터셋을 토크나이즈 하는 함수, 패딩은collate_fn에서
def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True, max_length=MAX_LEN, padding=False)


# DataLoader가 데이터 가지고오면 배치로 포장하는 함수 (여기서 입력 토큰 길이 정리)
def collate_fn(batch):
  # batch의 실제 모습 예시
  # batch = [
  #     {'input_ids': [1, 5, 10], 'attention_mask': [1, 1, 1]},   # 샘플 1 (길이 3)
  #     {'input_ids': [2, 4], 'attention_mask': [1, 1]},         # 샘플 2 (길이 2)
  #     {'input_ids': [7, 8, 9, 3], 'attention_mask': [1, 1, 1, 1]} # 샘플 3 (길이 4)
  # ]
  # input_ids는 입력 받은 토큰의 인덱스 번호가 담긴 리스트
  # attention_mask는 해당 토큰을 학습 해야하는지 여부가 담긴 이진 리스트 (보통 값은 1이지만, 패딩 때문에 0인 경우가 있음)

  # 1. 배치에 들어있는 값을 텐서로 변환
  input_ids = [torch.tensor(item['input_ids'], dtype=torch.long) for item in batch]

  # 2. 패딩 (해당 배치 내에서 최대 길이에 맞춰서 패딩)
  input_ids = torch.nn.utils.rnn.pad_sequence(
      input_ids,
      batch_first=True,
      padding_value=tokenizer.pad_token_id # 패딩할 값은 토크나이저에서 지정한 값
  )

  # 3. attention_mask 생성
  # input_ids.ne(tokenizer.pad_token_id) -> input_ids의 각 요소가 tokenizer.pad_token_id와 다르면 True, 아니면 False (불리언 텐서) 반환
  # long() -> 불리언 텐서 (True/False)를 1과 0으로 캐스팅
  attention_mask = input_ids.ne(tokenizer.pad_token_id).long()

  # 4. labels 생성 (정답 생성)
  label = input_ids.clone() # 값 복사
  label[input_ids == tokenizer.pad_token_id] = -100 # 패딩부분의 정답을 -100으로 해서 Pytorch의 손실함수가 학습하지 않도록 함

  return {
      "input_ids": input_ids[:, :MAX_LEN],
      "attention_mask": attention_mask[:, :MAX_LEN],
      "labels": label[:, :MAX_LEN]
  }

# 인코더를 디코더처럼 사용하기 위해 사용하는 마스크 만드는 함수
def generate_causal_mask(seq_len, device):
  mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
  return mask

def generate_text(model, tokenizer, prompt, max_new_tokens=50, temperature=1.0, device="cpu"):
  model.eval()

  input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

  with torch.no_grad():
    for _ in range(max_new_tokens):
      curr_seq_len = input_ids.size(1)

      inputs = input_ids[:, -MAX_LEN:]
      causal_mask = generate_causal_mask(inputs.size(1), device)

      logits = model(inputs, mask=causal_mask)

      next_token_logits = logits[:, -1, :] / temperature

      next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

      input_ids = torch.cat([input_ids, next_token], dim=-1)

      if next_token.item() == tokenizer.eos_token_id:
        break

  return tokenizer.decode(input_ids[0], skip_special_tokens=True)

def main():
    device = get_device()
    print(f"감지된 시스템: {device}")

    # 1. 모델과 옵티마이저 인스턴스 생성
    model = MyFirstLLM(VOCAB_SIZE, D_MODEL, N_HEAD, NUM_LAYERS, MAX_LEN).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    start_step, model_state, optim_state, dataset_state = load_checkpoint(device)

    # 모델 가중치 복구 (데이터가 있을 경우에만)
    if model_state:
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optim_state)

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

    if dataset_state:
        print("데이터셋 복구중...")
        dataset.load_state_dict(dataset_state)
        print("데이터셋 복구 완료")
    else:
        print("새로운 데이터셋 학습 시작.")

    tokenized_dataset = dataset.map(tokenize_fn, batched=True)

    train_dataloader = DataLoader(tokenized_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    model.train() # 모델을 학습 모드로 바꾸기

    loss_history = []

    # 학습
    for epoch in range(EPOCHS):
        total_loss = 0
        print(f"에포크 {epoch + 1}/{EPOCHS} 시작")
        # enumerate는 (0, batch), (1, batch)... 형태로 반환
        for step, batch in enumerate(train_dataloader):

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 입력과 정답 분리 (Shifted)
            inputs = input_ids[:, :-1]
            targets = labels[:, 1:]

            # 패딩 마스크 생성
            src_key_padding_mask = (attention_mask[:, :-1] == 0)

            curr_seq_len = inputs.size(1)
            causal_mask = generate_causal_mask(curr_seq_len, device)

            # 모델 실행
            logits = model(inputs, mask=causal_mask, src_key_padding_mask=src_key_padding_mask)

            loss = criterion(logits.view(-1, VOCAB_SIZE), targets.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

            if((step + 1) % 100 == 0 and step > 0):
                current_total_step = start_step + step + 1
                loss_history.append(loss.item())
                checkpoint = {
                    'step' : current_total_step,
                    'model_state_dict' : model.state_dict(),
                    'optimizer_state_dict' : optimizer.state_dict(),
                    'dataset_state_dict' : dataset.state_dict(),
                    'loss' : loss.item(),
                }
                if not os.path.exists(CHECKPOINT_DIR):
                    os.makedirs(CHECKPOINT_DIR)

                try:
                  tmp_path = CHECKPOINT_PATH.with_suffix('.tmp')
                  torch.save(checkpoint, tmp_path)

                  os.replace(tmp_path, CHECKPOINT_PATH)
                except Exception as ex:
                   print(f"[Error] 체크포인트 저장중 에러 발생: {ex}")
                   if tmp_path.exists():
                      tmp_path.unlink()
                   raise ex

                now = datetime.now(kst).strftime('%Y-%m-%d %H:%M:%S')
                print(f"[{now}] 체크포인트 저장: Step {current_total_step} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / (step + 1)
        try:
            ppl = math.exp(avg_loss)
        except OverflowError:
            ppl = float('inf')
        print(f"Epoch {epoch} Average Loss: {avg_loss:.4f} Perplexity: {ppl:.4f}")

    # 학습 종료 후 그래프 출력
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.title("Training Loss")
    plt.xlabel("Step(x100)")
    plt.ylabel("Loss")
    plt.show()

    test_prompt = """
    The future of AI is
    """

    generated = generate_text(model, tokenizer, test_prompt, device=device)
    print(f"Prompt: {test_prompt}")
    print(f"Generated: {generated}")

if __name__ == "__main__":
   main()