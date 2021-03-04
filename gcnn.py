import argparse
import time
from tqdm import tqdm
import copy as cp

import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.data import DataLoader, DataListLoader
from torch_geometric.nn import DataParallel
from torch.nn import Linear
from torch_geometric.nn import global_mean_pool, GATConv

from data_loader import *
from eval_helper import *


"""

The GCNFN implementation

"""


parser = argparse.ArgumentParser()

# original model parameters
parser.add_argument('--seed', type=int, default=777, help='random seed')
parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')

# hyper-parameters
parser.add_argument('--dataset', type=str, default='gossipcop', help='[politifact, gossipcop]')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')
parser.add_argument('--nhid', type=int, default=128, help='hidden size')
parser.add_argument('--epochs', type=int, default=50, help='maximum number of epochs')
parser.add_argument('--concat', type=bool, default=False, help='whether concat news embedding and graph embedding')
parser.add_argument('--no_feature', type=bool, default=False, help='whether including node feature')
parser.add_argument('--multi_gpu', type=bool, default=True, help='multi-gpu mode')
parser.add_argument('--feature', type=str, default='content', help='feature type, [hand, glove, bert, content]')

args = parser.parse_args()
torch.manual_seed(args.seed)
if torch.cuda.is_available():
	torch.cuda.manual_seed(args.seed)

dataset = FNNDataset(root='data', feature=args.feature, empty=False, name=args.dataset, transform=ToUndirected())

if args.no_feature:
	dataset.data.x = torch.ones(dataset.data.x.shape)

args.num_classes = dataset.num_classes
args.num_features = dataset.num_features

print(args)

train_edge_slices = dataset.slices['edge_index']
train_node_slices = dataset.slices['x']
graph_list = []
for index, (edge_slice, node_slice) in enumerate(zip(train_edge_slices[:-1], train_node_slices[:-1])):
	data = cp.copy(dataset.data)
	data.edge_index = data.edge_index[:, edge_slice:train_edge_slices[index + 1]]
	# extract self-loops
	mask = data.edge_index[0, :] == data.edge_index[1, :]
	data.edge_index = torch.masked_select(data.edge_index, mask).reshape(2, -1)
	# data.edge_index = torch.empty(2, 1).type(torch.LongTensor)
	data.num_nodes = data.num_nodes[index]
	data.x = data.x[node_slice:train_node_slices[index+1], :]
	data.y = data.y[index]
	graph_list.append(data)

num_training = int(len(dataset) * 0.2)
num_val = int(len(dataset) * 0.1)
num_test = len(dataset) - (num_training + num_val)
training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])

if args.multi_gpu:
	loader = DataListLoader
else:
	loader = DataLoader

train_loader = loader(training_set, batch_size=args.batch_size, shuffle=True)
val_loader = loader(validation_set, batch_size=args.batch_size, shuffle=False)
test_loader = loader(test_set, batch_size=args.batch_size, shuffle=False)


class Net(torch.nn.Module):
	def __init__(self):
		super(Net, self).__init__()

		self.num_features = dataset.num_features
		self.nhid = args.nhid

		self.conv1 = GATConv(self.num_features, self.nhid * 2)
		self.conv2 = GATConv(self.nhid * 2, self.nhid * 2)

		self.fc1 = Linear(self.nhid * 2, self.nhid)
		self.fc2 = Linear(self.nhid, dataset.num_classes)

	def forward(self, data):
		x, edge_index, batch = data.x, data.edge_index, data.batch

		x = F.selu(self.conv1(x, edge_index))
		x = F.selu(self.conv2(x, edge_index))
		x = F.selu(global_mean_pool(x, batch))
		x = F.selu(self.fc1(x))
		x = F.dropout(x, p=0.5, training=self.training)
		x = self.fc2(x)
		return F.log_softmax(x, dim=-1)


model = Net().to(args.device)
if args.multi_gpu:
	model = DataParallel(model)
model = model.to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def compute_test(loader, verbose=False):
	model.eval()
	loss_test = 0.0
	out_log = []
	with torch.no_grad():
		for data in loader:
			if not args.multi_gpu:
				data = data.to(args.device)
			out = model(data)
			if args.multi_gpu:
				y = torch.cat([d.y.unsqueeze(0) for d in data]).squeeze().to(out.device)
			else:
				y = data.y
			if verbose:
				print(F.softmax(out, dim=1).cpu().numpy())
			out_log.append([F.softmax(out, dim=1), y])
			loss_test += F.nll_loss(out, y).item()
	return eval_deep(out_log), loss_test


if __name__ == '__main__':
	# Model training
	out_log = []

	t = time.time()
	model.train()
	for epoch in tqdm(range(args.epochs)):
		loss_train = 0.0
		correct = 0
		for i, data in enumerate(train_loader):
			optimizer.zero_grad()
			if not args.multi_gpu:
				data = data.to(args.device)
			out = model(data)
			if args.multi_gpu:
				y = torch.cat([d.y.unsqueeze(0) for d in data]).squeeze().to(out.device)
			else:
				y = data.y
			loss = F.nll_loss(out, y)
			loss.backward()
			optimizer.step()
			loss_train += loss.item()
			out_log.append([F.softmax(out, dim=1), y])
		acc_train, _, _, _, recall_train, auc_train, _ = eval_deep(out_log)
		[acc_val, _, _, _, recall_val, auc_val, _], loss_val = compute_test(val_loader)
		print(f'loss_train: {loss_train:.4f}, acc_train: {acc_train:.4f},'
			  f' recall_train: {recall_train:.4f}, auc_train: {auc_train:.4f},'
			  f' loss_val: {loss_val:.4f}, acc_val: {acc_val:.4f},'
			  f' recall_val: {recall_val:.4f}, auc_val: {auc_val:.4f}')

	[acc, f1_macro, f1_micro, precision, recall, auc, ap], test_loss = compute_test(test_loader, verbose=False)
	print(f'Test set results: acc: {acc:.4f}, f1_macro: {f1_macro:.4f}, f1_micro: {f1_micro:.4f},'
		  f'precision: {precision:.4f}, recall: {recall:.4f}, auc: {auc:.4f}, ap: {ap:.4f}')