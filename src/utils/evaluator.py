import statistics
import torch.nn.functional as F

from collections import defaultdict


class Evaluator:
    def __init__(self, loss, logger):
        self.loss = loss
        self.logger = logger

    @staticmethod
    def _get_accuracy(output, labels):
        preds = output.max(1)[1].type_as(labels)
        correct = preds.eq(labels).double()
        correct = correct.sum()
        return correct / len(labels)

    def _get_loss_variance(self, output, labels):
        res = list()
        for i in range(labels.shape[0]):
            label = labels[i]
            if self.loss == "negative_log_likelihood":
                nll = -output[i, label].item()
            else:
                nll = -F.log_softmax(output, dim=1)[i, label].item()
            res.append(nll)
        if len(res) > 1:
            return statistics.variance(res)
        else:
            return 0

    def _get_bias(self, output, labels, idx, raw_graph):
        deg = raw_graph.sum(axis=0)
        loss_by_deg = defaultdict(list)
        deg_test = deg[0, idx.cpu().numpy()]
        if self.loss == "negative_log_likelihood":
            loss_mat = -output
        else:
            loss_mat = -F.log_softmax(output, dim=1)
        for i in range(idx.shape[0]):
            degree = int(deg_test[0, i])
            label = labels[i]
            loss_val = loss_mat[i, label].item()
            loss_by_deg[degree].append(loss_val)
        res = [statistics.mean(losses) for degree, losses in loss_by_deg.items()]
        return statistics.variance(res)

    def eval(self, output, labels, idx, raw_graph, stage):
        if self.loss == "negative_log_likelihood":
            loss_value = F.nll_loss(output, labels)
        else:
            loss_value = F.cross_entropy(output, labels)
        accuracy = self._get_accuracy(output, labels)
        bias = self._get_bias(output, labels, idx, raw_graph)
        info = "{stage} - loss: {loss}\taccuracy: {accuracy}\tbias:{bias}".format(
            stage=stage,
            loss=loss_value,
            accuracy=accuracy,
            bias=bias,
        )
        if stage in ("validation", "test"):
            info += "\n"
        self.logger.info(info)
        return info
