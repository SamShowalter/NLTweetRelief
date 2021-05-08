from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from transformers import AdamW
from loader import Loader
import torch

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=10)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.train()

l = Loader()

l.load_files()

print(l.train_corpus.columns)
# print(l.test_corpus.shape)
# print(l.train_corpus.shape)
# print(l.dev_corpus.shape)
# print(l.dev_crises)
# print(l.train_dict.keys())

optim = AdamW(model.parameters(), lr=5e-5)

for epoch_i in range(3):
    print(epoch_i)
    for epoch in l.next_epoch(batch_size=32, simulate = False):
        batch = epoch[0]
        labels = epoch[1]
        optim.zero_grad()
        tokenized = tokenizer([' '.join(s) for s in batch], padding=True)
        labels_tensor = torch.tensor([l[0] for l in labels]).to(device)
        input_ids = torch.tensor(tokenized["input_ids"]).to(device)
        attention_mask = torch.tensor(tokenized["attention_mask"]).to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels_tensor)
        loss = outputs[0]
        loss.backward()
        optim.step()
        del loss
        del attention_mask
        del input_ids
        del labels_tensor

model.eval()
