import random
import spacy
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext import data
from torchtext.datasets import IMDB
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 定义数据预处理
TEXT = data.Field(tokenize='spacy', include_lengths=True, tokenizer_language='en_core_web_sm')
LABEL = data.LabelField(dtype=torch.float)

# 加载IMDB数据集，创建训练集和测试集以及验证集
train_data, test_data = IMDB.splits(TEXT, LABEL)
train_data, valid_data = train_data.split(random_state=random.seed(22))

# 构建词汇表
MAX_VOCAB_SIZE = 25000
TEXT.build_vocab(train_data, max_size=MAX_VOCAB_SIZE, vectors='glove.6B.100d')
LABEL.build_vocab(train_data)

# 设置设备，创建训练集、测试集、验证集的迭代器
BATCH_SIZE = 32
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    sort_within_batch=True,
    device=device
)


# 定义RNN模型
class RNN(nn.Module):
    def __init__(self, inputs, embedding, hidden, outputs):
        super().__init__()
        self.embedding = nn.Embedding(inputs, embedding)
        self.rnn = nn.RNN(embedding, hidden)
        self.fc1 = nn.Linear(hidden, hidden // 2)
        self.fc2 = nn.Linear(hidden // 2, outputs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, text):
        embedded = self.embedding(text)
        output, hidden = self.rnn(embedded)
        predictions = self.relu(self.fc1(hidden.squeeze(0)))
        predictions = self.fc2(predictions)
        return predictions


class LSTM(nn.Module):
    def __init__(self, inputs, embedding, hidden, outputs):
        super().__init__()
        self.embedding = nn.Embedding(inputs, embedding)
        self.lstm = nn.LSTM(embedding, hidden)
        self.fc1 = nn.Linear(hidden, hidden // 2)
        self.fc2 = nn.Linear(hidden // 2, outputs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, cell) = self.lstm(embedded)
        predictions = self.relu(self.fc1(hidden[-1]))  # Use the last hidden state
        predictions = self.fc2(predictions)
        return predictions


# 初始化模型和优化器
INPUT = len(TEXT.vocab)
EMBEDDING = 100
HIDDEN = 128
OUTPUT = 1
# model = RNN(INPUT, EMBEDDING, HIDDEN, OUTPUT)
model = LSTM(INPUT, EMBEDDING, HIDDEN, OUTPUT)
optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()
# 将模型和损失函数移至设备
model = model.to(device)
criterion = criterion.to(device)


# 计算准确率
def binary_accuracy(predicts, y):
    rounded_predicts = torch.round(torch.sigmoid(predicts))
    correct = (rounded_predicts == y).float()
    accuracy = correct.sum() / len(correct)
    return accuracy


# 训练
def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    epoch_accuracy = 0
    for batch in tqdm(iterator, desc=f'Epoch [{epoch + 1}/{EPOCHS}]', delay=0.1):
        optimizer.zero_grad()
        predictions = model(batch.text[0]).squeeze(1)
        loss = criterion(predictions, batch.label)
        accuracy = binary_accuracy(predictions, batch.label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_accuracy += accuracy.item()
    return epoch_loss / len(iterator), epoch_accuracy / len(iterator)


# 验证
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    epoch_accuracy = 0
    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text[0]).squeeze(1)
            loss = criterion(predictions, batch.label)
            accuracy = binary_accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_accuracy += accuracy.item()
    return epoch_loss / len(iterator), epoch_accuracy / len(iterator)


# 训练模型
EPOCHS = 10
for epoch in range(EPOCHS):
    train_loss, train_accuracy = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_accuracy = evaluate(model, valid_iterator, criterion)
    print(f'\tTrain Loss: {train_loss:.3f} | Train Accuracy: {train_accuracy * 100:.2f}%')
    print(f'\tValidation Loss: {valid_loss:.3f} | Validation Accuracy: {valid_accuracy * 100:.2f}%')

# 测试模型
test_loss, test_accuracy = evaluate(model, test_iterator, criterion)
print(f'\nTest Loss: {test_loss:.3f} | Test Acc: {test_accuracy * 100:.2f}%')
nlp = spacy.load('en_core_web_sm')


# 使用模型进行预测
def predict_sentiment(model, sentence, threshold=0.5):
    model.eval()
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    prediction = torch.sigmoid(model(tensor))
    print(f'Proba: {prediction.item()}')
    return 'positive' if prediction.item() > threshold else 'negative'


# 示例预测
positive_comment = "What a fantastic film!"
negative_comment = "I was disappointed by this movie."
print(f'Comment: {positive_comment}\nPrediction: {predict_sentiment(model, positive_comment)}')
print(f'Comment: {negative_comment}\nPrediction: {predict_sentiment(model, negative_comment)}')
