import torch
import torch.nn as nn
from transformers import AutoTokenizer
import os
import pytz
from pathlib import Path
from model.myFirstLLM import MyFirstLLM

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_NAME = "MyFirstLLM.pt"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
TOKENIZER_PATH = BASE_DIR / "tokenizer" / "MyFirstTokenizer"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

# 패딩 토큰이 없는 경우 예외
if tokenizer.pad_token is None:
    raise ValueError("패팅 토큰이 없습니다.")

criterion = nn.CrossEntropyLoss()

VOCAB_SIZE = len(tokenizer)  # 어휘 사전 크기
D_MODEL = 512  # 임베딩 차원
N_HEAD = 8  # 멀티 헤드 어텐션 개수
NUM_LAYERS = 6  # 디코더 개수
MAX_LEN = 256  # 입력 문자열 최대 길이

EPOCHS = 1
BATCH_SIZE = 8
SEED = 20260103

kst = pytz.timezone("Asia/Seoul")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_checkpoint(device, checkpoint_name=CHECKPOINT_NAME):
    checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print(f"--- 체크포인트 발견: Step {checkpoint['step']}부터 재시작 ---")
        return (
            checkpoint["step"],
            checkpoint["model_state_dict"],
            checkpoint["optimizer_state_dict"],
            checkpoint["dataset_state_dict"],
            checkpoint["best_val_loss"],
        )
    return 0, None, None, None, float("inf")


def generate_causal_mask(seq_len, device):
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    return mask

# dataset_state를 출력해주는 함수
def print_dataset_summary(state):

    epoch = state["epoch"]

    ex_iters = state['examples_iterable']['examples_iterable']['ex_iterables']

    en_info, kr_info = ex_iters

    en_shard, en_idx = en_info['shard_idx'], en_info['shard_example_idx']
    kr_shard, kr_idx = kr_info['shard_idx'], kr_info['shard_example_idx']

    rng_info = state['examples_iterable']['examples_iterable']['bit_generator_state']
    bit_gen_name = rng_info['bit_generator']
    bit_gen_state = rng_info['state']

    print(f"\n=== [Dataset State Summary (Epoch {epoch})] ===")

    print(f"[0] en_data | Shard: {en_shard:3} | Index: {en_idx:,}")
    print(f"[1] kr_data | Shard: {kr_shard:3} | Index: {kr_idx:,}")

    print(f"▶ RNG Type: {bit_gen_name}")
    print(f"▶ RNG State (Internal): {bit_gen_state}")

    print("=" * 40 + "\n")

def generate_text(
    model, tokenizer, prompt, max_new_tokens=50, temperature=1.0, device="cpu"
):
    model.eval()

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

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

    model = MyFirstLLM(VOCAB_SIZE, D_MODEL, N_HEAD, NUM_LAYERS, MAX_LEN).to(device)

    start_step, model_state, optim_state, dataset_state, best_val_loss = load_checkpoint(device)
    if model_state:
        model.load_state_dict(model_state)
    else:
        print("경고! 불러올 model_state가 없습니다.")
        
    print(f"[start_step]: {start_step}")
    print(f"[best_val_loss]: {best_val_loss}")
    print_dataset_summary(dataset_state)

    print("[ Prompt ]")
    test_prompt = input(">> ").strip()

    generated = generate_text(model, tokenizer, test_prompt, device=device)
    
    print("[ Generated ]")
    print(f">> {generated[len(test_prompt):].strip()}")


if __name__ == "__main__":
    main()
