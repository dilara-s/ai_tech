import re
import torch
import torch.nn as nn

try:
    with open("saltan.txt", encoding="utf-8") as file:
        text = file.read()
except UnicodeDecodeError:
    with open("saltan.txt", encoding="cp1251") as file:
        text = file.read()

text = text.lower()
symbols = re.findall("[а-яё \n]", text)

res = "".join(symbols)
res = res.replace("\n", " ")
for i in range(5):
    res = res.replace("  ", " ")

words = res.split()
vocab = sorted(set(words))
print("слов в тексте:", len(words), "| уникальных слов:", len(vocab))
if not words:
    raise SystemExit("В saltan.txt нет русских букв: проверь файл и его кодировку")


class RNN(nn.Module):
    def __init__(self, vocab, embed_size=32, hidden_size=64):
        super(RNN, self).__init__()
        self.vocab = vocab
        self.word2idx = {w: i for i, w in enumerate(vocab)}
        self.hidden_size = hidden_size
        self.len = 5
        self.epochs = 10

        self.embedding = nn.Embedding(len(vocab), embed_size)
        self.model = nn.Linear(embed_size + hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.output = nn.Linear(hidden_size, len(vocab))

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, ids):
        hidden = torch.zeros(self.hidden_size)
        vectors = self.embedding(torch.tensor(ids))
        for x in vectors:
            hidden = self.activation(self.model(torch.cat((x, hidden))))
        return self.output(hidden)

    def fit(self, words):
        self.train()
        ids = [self.word2idx[w] for w in words]
        total = len(ids) - self.len
        losses = []
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for i in range(total):
                window = ids[i:i + self.len]
                target = torch.tensor([ids[i + self.len]])

                logits = self(window)
                loss = self.criterion(logits.unsqueeze(0), target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() / total
            losses.append(epoch_loss)
            print(f"эпоха {epoch + 1}/{self.epochs}, loss {epoch_loss:.4f}")
        return losses

    def predict(self, words):
        self.eval()
        ids = [self.word2idx[w] for w in words[-self.len:]]
        with torch.no_grad():
            logits = self(ids)
        return self.vocab[torch.argmax(logits).item()]


model = RNN(vocab)
model.fit(words)
print("контекст:", " ".join(words[-5:]))
print("следующее слово:", model.predict(words))
