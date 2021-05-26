from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification, Trainer, TrainingArguments
from transformers import AdamW
from loader import Loader
import torch
from tqdm import tqdm
from tokenizer_model_factory import TokenizerModelFactory
import sys

model_name = sys.argv[1]

def train_model(tokenizer, model, n_epochs=1000):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.train()

    l = Loader()

    l.load_files()

    optim = AdamW(model.parameters(), lr=5e-5)

    max_len = tokenizer.max_model_input_sizes[model_name] if model_name != 'lstm' else 512
    print("max_len", max_len)

    for epoch_i in tqdm(list(range(n_epochs))):
        print(epoch_i)
        n_res = 0
        sum_res = 0
        for epoch in l.next_epoch(batch_size=16, simulate=True):
            batch = epoch[0]
            labels = epoch[1]
            optim.zero_grad()
            tokenized = tokenizer(list(batch), padding=True, truncation=True, is_split_into_words=True, return_length=True, max_length=max_len)
            input_ids = torch.tensor(tokenized["input_ids"]).to(device)
            attention_mask = torch.tensor(tokenized["attention_mask"]).to(device)
            labels_tensor = torch.tensor([list(label) + [-100]*(length - len(label)) if len(label) <= max_len else list(label)[:max_len] for label, length in zip(labels, tokenized["length"])]).to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels_tensor)
            logits = outputs.logits.detach().max(axis=-1)[1]
            mask = labels_tensor != -100
            n_res += mask.float().sum()
            sum_res += ((logits == labels_tensor) & mask).float().sum()
            loss = outputs[0]
            loss.backward()
            optim.step()

        if epoch_i % 50 == 0:
            model.save_pretrained("models/multilabel/%s" % model_name)
            model.eval()
            print(labels_tensor)
            print(logits)
            train2_acc, dev_acc = benchmark(tokenizer, model, l)
            model.train()

        train_acc = sum_res/n_res
        print("train_acc", train_acc)

    model.eval()
    return model, train_acc, l


def benchmark(tokenizer, model, loader):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Dont need this
    # loader = Loader()

    # loader.load_files()

    n_res = 0
    sum_res = 0
    for batch, labels,crises in loader.next_epoch(batch_size=16, simulate=True, dataset="train"):
        tokenized = tokenizer(list(batch), padding=True, is_split_into_words=True, return_length=True)
        input_ids = torch.tensor(tokenized["input_ids"]).to(device)
        attention_mask = torch.tensor(tokenized["attention_mask"]).to(device)
        labels_tensor = torch.tensor([list(label) + [-100]*(length - len(label)) for label, length in zip(labels, tokenized["length"])]).to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits.max(axis=-1)[1]
        mask = labels_tensor != -100
        n_res += mask.float().sum()
        sum_res += ((logits == labels_tensor) & mask).float().sum()

    train_acc = sum_res/n_res

    print("train_acc", train_acc)

    # l_dev = Loader()

    # l_dev.load_files()

    n_res = 0
    sum_res = 0
    for batch, labels,crises in loader.next_epoch(batch_size=16, simulate=True, dataset="dev"):
        tokenized = tokenizer(list(batch), padding=True, is_split_into_words=True, return_length=True)
        input_ids = torch.tensor(tokenized["input_ids"]).to(device)
        attention_mask = torch.tensor(tokenized["attention_mask"]).to(device)
        labels_tensor = torch.tensor([list(label) + [-100]*(length - len(label)) for label, length in zip(labels, tokenized["length"])]).to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits.max(axis=-1)[1]
        mask = labels_tensor != -100
        n_res += mask.float().sum()
        sum_res += ((logits == labels_tensor) & mask).float().sum()

    dev_acc = sum_res/n_res

    print("dev_acc", dev_acc)
    return train_acc, dev_acc


tokenizermodelfactory = TokenizerModelFactory()
tokenizer, model = tokenizermodelfactory.makeTokenizerModel(model_name, unilabel=False, num_labels=10)

model, train_acc, l = train_model(tokenizer, model)

model.save_pretrained("models/multilabel/%s" % model_name)

train2_acc, dev_acc = benchmark(tokenizer, model, l)

