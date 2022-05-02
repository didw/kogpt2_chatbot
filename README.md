# KoGPT2 Chatbot

# Dataset
- 한국어 챗봇 데이터
- https://github.com/songys/Chatbot_data


# Usage
``` bash
# install requirements
pip install -r requirements.txt
# clone dataset
git clone https://github.com/songys/Chatbot_data
# Train
python main.py --train
# Chat
python main.py --chat
```

## DEMO
```text
User: 배고프다
Bot:   맛있는 거 드세요.
User: 날씨가 너무 시원하다
Bot:   시원한 곳으로 가세요.
User: 핸드폰 샀다!
Bot:   연락해보세요.
User: 연락할 사람이 없다
Bot:   연락을 안하는게 더 나을 수도 있어요.
User: 남친이랑 헤어졌어
Bot:   잘 지내고 있길 바랄게요.
User: 슬프다
Bot:   저도 사랑합니다.
User: 슬프다고
Bot:   저도 슬퍼요.
```


## Reference
- https://github.com/SKT-AI/KoGPT2
- https://towardsdatascience.com/how-to-fine-tune-gpt-2-for-text-generation-ae2ea53bc272
- https://github.com/haven-jeon/KoGPT2-chatbot
- https://github.com/gyunggyung/KoGPT2-FineTuning
