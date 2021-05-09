from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from transformers import AdamW
from loader import Loader
import torch
from tqdm import tqdm

def train_model(tokenizer, model, n_epochs=1):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.train()

    l = Loader()

    l.load_files()

    optim = AdamW(model.parameters(), lr=5e-5)

    for epoch_i in tqdm(list(range(n_epochs))):
        print(epoch_i)

        n_res = 0
        sum_res = 0
        for epoch in l.next_epoch(batch_size=32, simulate = False, dataset = 'train'):
            batch = epoch[0]
            labels = epoch[1]
            optim.zero_grad()
            tokenized = tokenizer([' '.join(s) for s in batch], padding=True)
            labels_tensor = torch.tensor([l[0] for l in labels]).to(device)
            input_ids = torch.tensor(tokenized["input_ids"]).to(device)
            attention_mask = torch.tensor(tokenized["attention_mask"]).to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels_tensor)
            loss = outputs[0]

            logits = outputs.logits.detach().max(axis=1)[1]
            n_res += logits.shape[0]
            sum_res += (logits == labels_tensor).float().sum()

            loss.backward()
            optim.step()
            del loss
            del attention_mask
            del input_ids
            del labels_tensor

        train_acc = sum_res/n_res

        print("train_acc", train_acc)

    model.eval()
    return model, train_acc, l


def benchmark(tokenizer, model, loader):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # loader = Loader()

    # loader.load_files()

    n_res = 0
    sum_res = 0
    for batch, labels in loader.next_epoch(batch_size=32, simulate=False, dataset="train"):
        tokenized = tokenizer([' '.join(s) for s in batch], padding=True)
        labels_tensor = torch.tensor([l[0] for l in labels]).to(device)
        input_ids = torch.tensor(tokenized["input_ids"]).to(device)
        attention_mask = torch.tensor(tokenized["attention_mask"]).to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits.max(axis=1)[1]
        n_res += logits.shape[0]
        sum_res += (logits == labels_tensor).float().sum()

    train_acc = sum_res/n_res

    print("train_acc", train_acc)

    # l_dev = Loader()

    # l_dev.load_files()
    # print(l_dev.unilabel_df)

    n_res = 0
    sum_res = 0
    for batch, labels in loader.next_epoch(batch_size=32, simulate=False, dataset="dev"):
        tokenized = tokenizer([' '.join(s) for s in batch], padding=True)
        labels_tensor = torch.tensor([l[0] for l in labels]).to(device)
        input_ids = torch.tensor(tokenized["input_ids"]).to(device)
        attention_mask = torch.tensor(tokenized["attention_mask"]).to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits.max(axis=1)[1]
        n_res += logits.shape[0]
        sum_res += (logits == labels_tensor).float().sum()

    dev_acc = sum_res/n_res

    print("dev_acc", train_acc)
    return train_acc, dev_acc

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=10)

model, train_acc, loader = train_model(tokenizer, model)
train_acc, dev_acc = benchmark(tokenizer, model, loader)
