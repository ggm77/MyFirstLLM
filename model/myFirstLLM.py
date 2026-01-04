import torch
import torch.nn as nn

class MyFirstLLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, num_layers, max_len):
        super().__init__()

        # 1. 임베딩 층
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # 2. 포지셔널 인코딩
        self.pos_embedding = nn.Embedding(max_len, d_model)

        # 3. 디코더층 (전통 디코더에서 크로스 어텐션 연산 빼기 위해서 인코더로 만들고 나중에 마스킹 추가)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_model * 4,  # 피드포워드 네트워크 차원수
            dropout=0.1,
            batch_first=True,
            norm_first=True,  # 층 정규화를 먼저 하냐 여부
            activation="gelu",  # 활성화 함수
        )
        self.transformer = nn.TransformerEncoder(
            decoder_layer, num_layers=num_layers
        )  # num_layer만큼 쌓기

        # 4. 출력 레이어
        self.lm_head = nn.Linear(
            d_model, vocab_size
        )  # 밀집층 (입력 차원, 출력 차원 = 은닉 벡터, 토큰)

    def forward(self, x, mask=None, src_key_padding_mask=None):
        """
        x: [batch_size, seq_len]
        mask: [seq_len, seq_len] (Causal Mask)
        src_key_padding_mask: [batch_size, seq_len] (Padding Mask, True for padded positions)
        """

        batch_size, seq_len = x.shape
        device = x.device

        # 위치 인덱스 생성
        pos = torch.arange(0, seq_len).unsqueeze(0).to(device)

        # 토큰 + 위치 임베딩 합치기
        out = self.token_embedding(x) + self.pos_embedding(pos)

        # 디코더 통과 (인코더에 마스킹 추가해서 디코더로 사용)
        # src_key_padding_mask를 전달하여 패딩 토큰에 어텐션하지 않도록 함
        out = self.transformer(
            out, mask=mask, src_key_padding_mask=src_key_padding_mask, is_causal=True
        )

        # 로짓 계산 (은닉 벡터를 토큰으로)
        logits = self.lm_head(out)

        return logits
