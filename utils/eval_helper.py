from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, roc_auc_score, average_precision_score


"""
	Utility functions for evaluating the model performance
"""


def eval_deep(log):

	accuracy, f1_macro, f1_micro, precision, recall = 0, 0, 0, 0, 0

	prob_log, label_log = [], []

	for batch in log:
		pred_y, y = batch[0].data.cpu().numpy().argmax(axis=1), batch[1].data.cpu().numpy().tolist()
		prob_log.extend(batch[0].data.cpu().numpy()[:, 1].tolist())
		label_log.extend(y)

		accuracy += accuracy_score(y, pred_y)
		f1_macro += f1_score(y, pred_y, average='macro')
		f1_micro += f1_score(y, pred_y, average='micro')
		precision += precision_score(y, pred_y, zero_division=0)
		recall += recall_score(y, pred_y, zero_division=0)

	auc = roc_auc_score(label_log, prob_log)
	ap = average_precision_score(label_log, prob_log)

	return accuracy/len(log), f1_macro/len(log), f1_micro/len(log), precision/len(log), recall/len(log), auc, ap


def eval_shallow(pred_y, prob_y, y):

	f1_macro = f1_score(y, pred_y, average='macro')
	f1_micro = f1_score(y, pred_y, average='micro')
	accuracy = accuracy_score(y, pred_y)
	precision = precision_score(y, pred_y, zero_division=0)
	recall = recall_score(y, pred_y, zero_division=0)
	auc = roc_auc_score(y, prob_y)
	ap = average_precision_score(y, prob_y)

	return accuracy, f1_macro, f1_micro, precision, recall, auc, ap