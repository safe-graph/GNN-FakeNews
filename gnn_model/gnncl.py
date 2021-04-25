import argparse
from tqdm import tqdm
from math import ceil

import torch
import torch.nn.functional as F
from torch_geometric.data import DenseDataLoader
import torch_geometric.transforms as T
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool
from torch.utils.data import random_split

from utils.data_loader import *
from utils.eval_helper import *

"""

The GNN-CL is implemented using DiffPool as the graph encoder and profile feature as the node feature 

Paper: Graph Neural Networks with Continual Learning for Fake News Detection from Social Media
Link: https://arxiv.org/pdf/2007.03316.pdf

"""


class GNN(torch.nn.Module):
	def __init__(self, in_channels, hidden_channels, out_channels,
				 normalize=False, lin=True):
		super(GNN, self).__init__()
		self.conv1 = DenseSAGEConv(in_channels, hidden_channels, normalize)
		self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
		self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
		self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
		self.conv3 = DenseSAGEConv(hidden_channels, out_channels, normalize)
		self.bn3 = torch.nn.BatchNorm1d(out_channels)

		if lin is True:
			self.lin = torch.nn.Linear(2 * hidden_channels + out_channels,
									   out_channels)
		else:
			self.lin = None

	def bn(self, i, x):
		batch_size, num_nodes, num_channels = x.size()

		x = x.view(-1, num_channels)
		x = getattr(self, 'bn{}'.format(i))(x)
		x = x.view(batch_size, num_nodes, num_channels)
		return x

	def forward(self, x, adj, mask=None):
		batch_size, num_nodes, in_channels = x.size()

		x0 = x
		x1 = self.bn(1, F.relu(self.conv1(x0, adj, mask)))
		x2 = self.bn(2, F.relu(self.conv2(x1, adj, mask)))
		x3 = self.bn(3, F.relu(self.conv3(x2, adj, mask)))

		x = torch.cat([x1, x2, x3], dim=-1)

		if self.lin is not None:
			x = F.relu(self.lin(x))

		return x


class Net(torch.nn.Module):
	def __init__(self, in_channels=3, num_classes=6):
		super(Net, self).__init__()

		num_nodes = ceil(0.25 * max_nodes)
		self.gnn1_pool = GNN(in_channels, 64, num_nodes)
		self.gnn1_embed = GNN(in_channels, 64, 64, lin=False)

		num_nodes = ceil(0.25 * num_nodes)
		self.gnn2_pool = GNN(3 * 64, 64, num_nodes)
		self.gnn2_embed = GNN(3 * 64, 64, 64, lin=False)

		self.gnn3_embed = GNN(3 * 64, 64, 64, lin=False)

		self.lin1 = torch.nn.Linear(3 * 64, 64)
		self.lin2 = torch.nn.Linear(64, num_classes)

	def forward(self, x, adj, mask=None):
		s = self.gnn1_pool(x, adj, mask)
		x = self.gnn1_embed(x, adj, mask)

		x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)

		s = self.gnn2_pool(x, adj)
		x = self.gnn2_embed(x, adj)

		x, adj, l2, e2 = dense_diff_pool(x, adj, s)

		x = self.gnn3_embed(x, adj)

		x = x.mean(dim=1)
		x = F.relu(self.lin1(x))
		x = self.lin2(x)
		return F.log_softmax(x, dim=-1), l1 + l2, e1 + e2


def train():
	model.train()
	loss_all = 0
	out_log = []
	for data in train_loader:
		data = data.to(device)
		optimizer.zero_grad()
		out, _, _ = model(data.x, data.adj, data.mask)
		out_log.append([F.softmax(out, dim=1), data.y])
		loss = F.nll_loss(out, data.y.view(-1))
		loss.backward()
		loss_all += data.y.size(0) * loss.item()
		optimizer.step()
	return eval_deep(out_log, train_loader), loss_all / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
	model.eval()

	loss_test = 0
	out_log = []
	for data in loader:
		data = data.to(device)
		out, _, _ = model(data.x, data.adj, data.mask)
		out_log.append([F.softmax(out, dim=1), data.y])
		loss_test += data.y.size(0) * F.nll_loss(out, data.y.view(-1)).item()
	return eval_deep(out_log, loader), loss_test


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=777, help='random seed')
# hyper-parameters
parser.add_argument('--dataset', type=str, default='politifact', help='[politifact, gossipcop]')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
parser.add_argument('--nhid', type=int, default=128, help='hidden size')
parser.add_argument('--epochs', type=int, default=60, help='maximum number of epochs')
parser.add_argument('--feature', type=str, default='profile', help='feature type, [profile, spacy, bert, content]')

args = parser.parse_args()
torch.manual_seed(args.seed)
if torch.cuda.is_available():
	torch.cuda.manual_seed(args.seed)

if args.dataset == 'politifact':
	max_nodes = 500
else:
	max_nodes = 200 


dataset = FNNDataset(root='data', feature=args.feature, empty=False, name=args.dataset,
					 transform=T.ToDense(max_nodes), pre_transform=ToUndirected())

print(args)

num_training = int(len(dataset) * 0.2)
num_val = int(len(dataset) * 0.1)
num_test = len(dataset) - (num_training + num_val)
training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])

train_loader = DenseDataLoader(training_set, batch_size=args.batch_size, shuffle=True)
val_loader = DenseDataLoader(validation_set, batch_size=args.batch_size, shuffle=False)
test_loader = DenseDataLoader(test_set, batch_size=args.batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(in_channels=dataset.num_features, num_classes=dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


for epoch in tqdm(range(args.epochs)):
	[acc_train, _, _, _, recall_train, auc_train, _], loss_train = train()
	[acc_val, _, _, _, recall_val, auc_val, _], loss_val = test(val_loader)
	print(f'loss_train: {loss_train:.4f}, acc_train: {acc_train:.4f},'
		  f' recall_train: {recall_train:.4f}, auc_train: {auc_train:.4f},'
		  f' loss_val: {loss_val:.4f}, acc_val: {acc_val:.4f},'
		  f' recall_val: {recall_val:.4f}, auc_val: {auc_val:.4f}')

[acc, f1_macro, f1_micro, precision, recall, auc, ap], test_loss = test(test_loader)
print(f'Test set results: acc: {acc:.4f}, f1_macro: {f1_macro:.4f}, f1_micro: {f1_micro:.4f}, '
	  f'precision: {precision:.4f}, recall: {recall:.4f}, auc: {auc:.4f}, ap: {ap:.4f}')
