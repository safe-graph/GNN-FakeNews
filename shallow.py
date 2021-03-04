import argparse

from torch.utils.data import random_split
from torch_geometric.data import DataLoader

from data_loader import *
from eval_helper import *

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import sklearn.neural_network as nn


"""

The base classifiers using news textual embeddings as imputs

BERT+MLP, GloVe+MLP

"""

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777, help='random seed')

# hyper-parameters
parser.add_argument('--dataset', type=str, default='gossipcop', help='[politifact, gossipcop]')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--feature', type=str, default='bert', help='feature type, [glove, bert, hand]')
parser.add_argument('--model', type=str, default='mlp', help='model type, [lr, svm, rf, mlp]')

args = parser.parse_args()
torch.manual_seed(args.seed)
if torch.cuda.is_available():
	torch.cuda.manual_seed(args.seed)

dataset = FNNDataset(root='data', feature=args.feature, empty=False, name=args.dataset, pre_filter=None, transform=ToUndirected())

args.num_classes = dataset.num_classes
args.num_features = dataset.num_features

print(args)

num_training = int(len(dataset) * 0.2)
num_val = int(len(dataset) * 0.1)
num_test = len(dataset) - (num_training + num_val)
training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])

train_loader = DataLoader(training_set, batch_size=len(training_set), shuffle=True)
val_loader = DataLoader(validation_set, batch_size=len(validation_set), shuffle=False)
test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False)

if args.model == 'svm':
	clf = svm.SVC(kernel='linear', probability=True)
elif args.model == 'lr':
	clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=1000)
elif args.model == 'rf':
	clf = RandomForestClassifier(max_depth=100, random_state=0)
elif args.model == 'mlp':
	clf = nn.MLPClassifier(hidden_layer_sizes=(256, 128), activation='relu', solver='adam', learning_rate_init=0.001,
						   power_t=0.5, max_iter=100, shuffle=True, random_state=None)

# train the model
idx_train = dataset.slices['x'][training_set.indices] + 1
X_train = dataset.data.x[idx_train].data.cpu().numpy()
y_train = dataset.data.y[training_set.indices].data.cpu().numpy()
clf.fit(X_train, y_train)

# test the model
idx_test = dataset.slices['x'][test_set.indices] + 1
X_test = dataset.data.x[idx_test].data.cpu().numpy()
y_test = dataset.data.y[test_set.indices].data.cpu().numpy()
prob_y = clf.predict_proba(X_test)[:, 1]
pred_y = clf.predict(X_test)

acc, f1_macro, f1_micro, precision, recall, auc, ap = eval_shallow(pred_y, prob_y, y_test)

print(f'Test results: acc: {acc:.4f}, f1_macro: {f1_macro:.4f}, f1_micro: {f1_micro:.4f},'
	  f'precision: {precision:.4f}, recall: {recall:.4f}, '
	  f'auc: {auc:.4f}, ap: {ap:.4f}')