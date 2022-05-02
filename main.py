import argparse
import logging
import os

from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import torch.multiprocessing
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
torch.multiprocessing.set_sharing_strategy('file_system')


parser = argparse.ArgumentParser(description='GPT2 Chatbot')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

parser.add_argument('--train',
                    action='store_true',
                    default=False,
                    help='train chatbot from gpt2 model')

parser.add_argument('--chat',
                    action='store_true',
                    default=False,
                    help='chat using fine-tuned chatbot model')


class ChatbotDataset(Dataset):  
    def __init__(self, tokenizer):

        filename = "Chatbot_data/ChatbotData.csv"
        df = pd.read_csv(filename)
        _data = [[row['Q'], row['A']] for _, row in df.iterrows()]

        self.tokenizer = tokenizer

        self.chatbot_data = []

        for q, a in _data:
            encoded = self.tokenizer.encode(
                text=f"<usr>{q}<sys>{a}</s>",
                return_tensors = 'pt',
            )
            self.chatbot_data.append(encoded)
        self.chatbot_count = len(self.chatbot_data)
        
    def __len__(self):
        return self.chatbot_count

    def __getitem__(self, item):
        return self.chatbot_data[item]


class GPT2Chatbot():
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def generate(self, text=None, device='cpu'):
        self.model.eval()
        if text is None:
            text = "집에서 일하고 싶어"
        if "<usr>" not in text:
            text = f"<usr>{text}<sys>",
        if isinstance(text, tuple):
            text = text[0]
        encoded = self.tokenizer.encode(
            text=text,
            return_tensors='pt',
        )
        encoded = encoded.to(device)
        self.model = self.model.to(device)
        outputs = self.model.generate(encoded,
                           max_length=128,
                           repetition_penalty=2.0,
                           do_sample=True,
                           temperature=0.9,
                           pad_token_id=self.tokenizer.pad_token_id,
                           eos_token_id=self.tokenizer.eos_token_id,
                           bos_token_id=self.tokenizer.bos_token_id,
                           use_cache=True
                        )

        generated_text = self.tokenizer.decode(outputs[0])
        generated_text = " ".join(generated_text.split("<sys>")[1].split('\n')[:2])

        def normalize_text(text):
            text = text.replace("<usr>", "")
            text = text.replace("<sys>", "")
            text = text.replace("</s>", "")
            text = text.replace("<unk>", "")
            return text

        def print_chat(side, text):
            logger.info(side, text)
        
        text = normalize_text(text)
        generated_text = normalize_text(generated_text)
        print_chat("User: ", text)
        print_chat("Bot: ", generated_text)
        return generated_text

    def train(self,
            dataset,
            batch_size=64, epochs=5, lr=2e-5,
            warmup_steps=200,
            output_dir="checkpoint", output_prefix="kogpt2-chatbot",
            save_model_on_epoch=True,
        ):
        print_every = 500
        device = torch.device("cuda")

        model = self.model.cuda()
        model.train()
        optimizer = AdamW(model.parameters(), lr=lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=-1
        )

        train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        accumulating_batch_count = 0
        input_tensor = None
        avg_loss = 0
        for epoch in range(epochs):
            for idx, entry in tqdm(enumerate(train_dataloader)):
                input_tensor = entry.to(device)
                outputs = model(input_tensor, labels=input_tensor)
                loss = outputs.loss
                if avg_loss == 0:
                    avg_loss = loss.detach().cpu().numpy()
                else:
                    avg_loss = 0.9*avg_loss + 0.1*(loss.detach().cpu().numpy())
                loss.backward()

                if (accumulating_batch_count % batch_size) == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    model.zero_grad()

                if idx % print_every == 0:
                    print("loss: ", avg_loss)
                    user_message = self.tokenizer.decode(input_tensor[0][0])
                    user_message = user_message.split("<sys>")[0] + "<sys>"
                    self.generate(user_message, device)
                    user_message = "얼른 퇴근하고 싶다."
                    self.generate(user_message, device)
                    model.train()

                accumulating_batch_count += 1
                input_tensor = None
            logger.info(f"Training epoch {epoch}, loss: {avg_loss}")
            if save_model_on_epoch:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                torch.save(
                    model.state_dict(),
                    os.path.join(output_dir, f"{output_prefix}-{epoch}-{avg_loss:.2f}.pt"),
                )
        return model


def train():
    tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                                                         bos_token='</s>', 
                                                         eos_token='</s>', 
                                                         unk_token='<unk>',
                                                         pad_token='<pad>', 
                                                         mask_token='<unused0>'
                                                       )
    model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

    dataset = ChatbotDataset(tokenizer)
    gpt2_chatbot = GPT2Chatbot(tokenizer, model)
    gpt2_chatbot.train(dataset)


def chat():
    import glob
    def get_best_model():
        filelist = glob.glob('checkpoint/*')
        filelist = sorted(filelist, key=lambda x: float(x.split('-')[3].split('.pt')[0]))
        return filelist[0]
    try:
        filename = get_best_model()
    except IndexError:
        logger.warning("train first before chat")
        return

    tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                                                         bos_token='</s>', 
                                                         eos_token='</s>', 
                                                         unk_token='<unk>',
                                                         pad_token='<pad>', 
                                                         mask_token='<unused0>'
                                                       )
    model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")
    model.load_state_dict(torch.load(filename))
    gpt2_chatbot = GPT2Chatbot(tokenizer, model)
    while True:
        text = input("User: ")
        gen = gpt2_chatbot.generate(text)
        print("Bot: ", gen)


def main():
    args = parser.parse_args()
    if args.train:
        train()
    if args.chat:
        chat()
    if not (args.train or args.chat):
        print("use train or chat")


if __name__ == '__main__':
    main()
