import time
import numpy as np
import pickle as pkl
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

"""
The CSI model is implemented using LSTM

The SAFE model is implemented using CNN

"""



torch.manual_seed(777)

def preprocess_text(review):
	"""
	Clean the review text
	:param review: a single review
	:return:
	"""
	review = re.sub('[^a-zA-Z]', ' ', review)
	review = re.sub(r"\s+[a-zA-Z]\s+", ' ', review)
	review = re.sub(r'\s+', ' ', review)

	return review


def data_loader(data, seq_len, batch_size, train_per, val_per, pretrain=True, cuda=True):

	# load the GloVe vectors
	embed_dict = {}
	with open('data/glove.6B.300d.txt', 'r', encoding='utf-8') as f:
		for line in f:
			values = line.split()
			word = values[0]
			vector = np.asarray(values[1:], "float32")
			embed_dict[word] = vector

	# load the data
	with open(f'data/{data[:3]}_news_text.pkl', 'rb') as f:
		news = pkl.load(f)

	# create train, validation and testing dataset
	X = [preprocess_text(review) for review in news]
	y = np.array([0]*(len(news)//2) + [1]*(len(news)//2))
	# tokenizing the data
	tokenizer = Tokenizer(num_words=5000)
	tokenizer.fit_on_texts(X)
	X = tokenizer.texts_to_sequences(X)
	# total number of individual words in the dataset
	vocab_size = len(tokenizer.word_index) + 1
	# sentence padding
	X = pad_sequences(X, padding='post', maxlen=seq_len)
	# train_test splitting
	X = torch.LongTensor(X)
	y = torch.LongTensor(y)

	if cuda:
		X = X.cuda()
		y = y.cuda()

	dataset = TensorDataset(X, y)

	num_training = int(len(dataset) * train_per)
	num_val = int(len(dataset) * val_per)
	num_test = len(dataset) - (num_training + num_val)

	training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])

	train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)
	test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

	vectors = torch.zeros((vocab_size, 300))
	for word, index in tokenizer.word_index.items():
		glove_vector = embed_dict.get(word)
		if glove_vector is not None:
			vectors[index] = torch.FloatTensor(glove_vector)
	if cuda:
		vectors = vectors.cuda()

	embedding = nn.Embedding(vocab_size, 300)
	if cuda:
		embedding = embedding.cuda()
	if pretrain:
		# initialize the embedding with pre_trained GloVe vectors
		embedding.weight = nn.Parameter(vectors, requires_grad=False)
	else:
		# initialize the embedding using xavier method
		embedding.weight = nn.Parameter(torch.FloatTensor(vocab_size, 300), requires_grad=True)
		nn.init.xavier_uniform_(embedding.weight)

	return [train_loader, val_loader, test_loader], embedding


def train(model, embedding, data_loader, optimizer, loss_func):
	epoch_loss, epoch_acc = 0, 0

	model.train()

	for _, batch_data in enumerate(data_loader):
		inputs, labels = embedding(batch_data[0]), batch_data[1]
		optimizer.zero_grad()
		predictions = model(inputs).squeeze(1)

		loss = loss_func(predictions, labels.float())
		epoch_acc += accuracy_score(labels.cpu(), torch.round(torch.sigmoid(predictions)).detach().cpu())

		loss.backward()
		optimizer.step()

		epoch_loss += loss.item()

	return epoch_loss / len(data_loader), epoch_acc / len(data_loader)


def evaluate(model, embedding, data_loader, loss_func, test=False):
	epoch_loss, epoch_acc = 0, 0
	if test:
		epoch_f1, epoch_recall, epoch_precision = 0, 0, 0

	model.eval()

	with torch.no_grad():
		for _, batch_data in enumerate(data_loader):
			inputs, labels = embedding(batch_data[0]), batch_data[1]
			predictions = model(inputs).squeeze(1)

			epoch_loss += loss_func(predictions, labels.float()).item()

			if test:
				epoch_acc += accuracy_score(labels.cpu(), torch.round(torch.sigmoid(predictions)).detach().cpu())
				epoch_f1 += f1_score(labels.cpu(), torch.round(torch.sigmoid(predictions)).detach().cpu(), average='macro')
				epoch_recall += recall_score(labels.cpu(), torch.round(torch.sigmoid(predictions)).detach().cpu(), average='macro')
				epoch_precision += precision_score(labels.cpu(), torch.round(torch.sigmoid(predictions)).detach().cpu(), average='macro')
			else:
				epoch_acc += accuracy_score(labels.cpu(), torch.round(torch.sigmoid(predictions)).detach().cpu())

		if test:
			return epoch_loss / len(data_loader), epoch_acc / len(data_loader), epoch_f1 / len(data_loader), epoch_recall / len(
				data_loader), epoch_precision / len(data_loader)
		else:
			return epoch_loss / len(data_loader), epoch_acc / len(data_loader)


class RNN(nn.Module):
	def __init__(self, embedding_dim, hidden_dim, output_dim, cuda=True, ele_max=False):
		super(RNN, self).__init__()
		self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
		self.fc = nn.Linear(hidden_dim, output_dim)
		self.max = ele_max
		if cuda:
			self.rnn.cuda()
			self.fc.cuda()

	def forward(self, text):

		output, hidden = self.rnn(text)

		if self.max:
			#  return the element-wise max of all hidden states of input sequence
			return self.fc(torch.max(output, dim=1)[0])
		else:
			return self.fc(hidden.squeeze(0))


class LSTM(nn.Module):
	def __init__(self, embedding_dim, hidden_dim, output_dim, cuda=True, ele_max=False):
		super(LSTM, self).__init__()
		self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
		self.fc = nn.Linear(hidden_dim, output_dim)
		self.max = ele_max
		if cuda:
			self.lstm.cuda()
			self.fc.cuda()

	def forward(self, text):

		output, (hidden, cell) = self.lstm(text)

		if self.max:
			#  return the element-wise max of all hidden states of input sequence
			return self.fc(torch.max(output, dim=1)[0])
		else:
			return self.fc(hidden.squeeze(0))


class GRU(nn.Module):
	def __init__(self, embedding_dim, hidden_dim, output_dim, cuda=True, ele_max=False):
		super(GRU, self).__init__()
		self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
		self.fc = nn.Linear(hidden_dim, output_dim)
		self.max = ele_max
		if cuda:
			self.gru.cuda()
			self.fc.cuda()

	def forward(self, text):

		output, hidden = self.gru(text)

		if self.max:
			#  return the element-wise max of all hidden states of input sequence
			return self.fc(torch.max(output, dim=1)[0])
		else:
			return self.fc(hidden.squeeze(0))


class CNN(nn.Module):

	def __init__(self, embedding_dim, hidden_dim, output_dim, cuda=True):
		super(CNN, self).__init__()
		kernel_wins = [3,4,5]
		# Convolutional Layers with different window size kernels
		self.convs = nn.ModuleList([nn.Conv2d(1, hidden_dim, (w, embedding_dim)) for w in kernel_wins])
		# Dropout layer
		self.dropout = nn.Dropout(0.5)

		# FC layer
		self.fc = nn.Linear(len(kernel_wins) * hidden_dim, output_dim)

		if cuda:
			self.convs = self.convs.cuda()
			self.dropout = self.dropout.cuda()
			self.fc = self.fc.cuda()

	def forward(self, x):

		emb_x = x.unsqueeze(1)

		con_x = [conv(emb_x) for conv in self.convs]

		pool_x = [F.max_pool1d(x.squeeze(-1), x.size()[2]) for x in con_x]

		fc_x = torch.cat(pool_x, dim=1)

		fc_x = fc_x.squeeze(-1)

		fc_x = self.dropout(fc_x)
		logit = self.fc(fc_x)
		return logit

if __name__ == "__main__":

	# hyper-parameters
	data = 'politifact'  # politifact, gossipcop
	model_name = 'CNN'  # RNN, GRU, LSTM, CNN
	ele_max = True  # using the element-wise max of all hidden states or using the last hidden state
	pretrain = True  # whether use the pretrained GloVe vectors as input
	input_dim = 300  # the dimension of input
	hidden_dim = 128  # the dimension of the hidden layer
	output_dim = 1  # the dimension of the output layer
	learning_rate = 1e-1  # learning rate
	epochs = 300  # number of training epochs
	seq_len = 512  # sequence length
	batch_size = 128  # the batch size
	train_per = 0.2  # training data percentage
	val_per = 0.1  # validation data percentage

	cuda = torch.cuda.is_available()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# load the data
	[train_loader, val_loader, test_loader], embedding = data_loader(data, seq_len, batch_size, train_per, val_per, cuda)

	# initialize the model
	if model_name == 'RNN':
		model = RNN(input_dim, hidden_dim, output_dim, cuda, ele_max)
	elif model_name == 'LSTM':
		model = LSTM(input_dim, hidden_dim, output_dim, cuda, ele_max)
	elif model_name == 'GRU':
		model = GRU(input_dim, hidden_dim, output_dim, cuda, ele_max)
	elif model_name == 'CNN':
		model = CNN(input_dim, hidden_dim, output_dim, cuda)

	optimizer = optim.SGD(model.parameters(), lr=learning_rate)
	loss_func = nn.BCEWithLogitsLoss()
	model = model.to(device)
	loss_func = loss_func.to(device)

	# train the model
	loss_log, acc_log = [], []
	for e in range(epochs):

		start_time = time.time()

		train_loss, train_acc = train(model, embedding, train_loader, optimizer, loss_func)
		val_loss, val_acc = evaluate(model, embedding, val_loader, loss_func, test=False)

		end_time = time.time()

		print(f'Epoch: {e + 1} | Epoch Time: {end_time-start_time:.2f}s')
		print(f'\tTrain Loss: {train_loss: .3f} | Train Acc: {train_acc * 100:.2f}%')
		print(f'\tValid Loss: {val_loss: .3f} | Valid Acc: {val_acc * 100:.2f}%')

		loss_log.append((train_loss, val_loss))
		acc_log.append((train_acc, val_acc))

	# test the model
	test_loss, test_acc, test_f1, test_recall, test_precision = evaluate(model, embedding, test_loader, loss_func, test=True)

	print(f'\nTest Loss: {test_loss: .3f} | Test ACC: {test_acc * 100:.2f}% | Test F1: {test_f1 * 100:.2f}%')
	print(f'Test Recall: {test_recall * 100:.2f}%  | Test Precision: {test_precision * 100:.2f}%')

