import torch
from queue import PriorityQueue
import torch.nn as nn
import torch.nn.functional as F



class BeamSearchNode(object):
    """ Beam search node class """
    def __init__(self, previous_node, char_id, logProb, length):
        """
        Args:
            previous_node (obj:`BeamSearchNode`): node in queue
            char_id (dict): character id
            logProb (float): word probability
            length (int): word length
        """
        self.prev_node = previous_node
        self.char_id = char_id
        self.logp = logProb
        self.leng = length

    def eval(self):
        """ Calculate beam search path score

        Returns:
            float: beam search path score
        """
        return self.logp / float(self.leng - 1 + 1e-6)

    def __lt__(self, other):
        """
        Args:
            self (obj:`BeamSearchNode`): beam search node
            other (obj:`BeamSearchNode`): beam search node
        """
        if self.eval() < other.eval():
            return False
        else:
            return True


def seq_modeling(encoder_outputs, alpha=0.2,path_thred=0.1,beam_width=3, topk=6):
    device = encoder_outputs.device
    max_len =  min(26,encoder_outputs.shape[1])
    N,T,C = encoder_outputs.shape
    batch_prob_graph = []
    length = torch.zeros(N).to(device)
    alpha_weight = torch.zeros_like(encoder_outputs)
    for idx in range(N):
        decoder_input = torch.tensor(-1, device=device)
        endnodes = [] # for back-trace
        # starting node -  previous node, char id, logp, length
        node = BeamSearchNode(None, decoder_input, 1.0, 1)
        nodes = PriorityQueue()
        nodes.put(node)
        qsize = 1
        while True:
            if qsize > 200:
                break
            priority_node = nodes.get()
            if priority_node.char_id == 0 or priority_node.leng == max_len+1:
                endnodes.append(priority_node)
                if len(endnodes) >= topk:
                    break
                else:
                    continue
            prob, indexes = torch.topk(encoder_outputs[idx][priority_node.leng-1], beam_width)
            for new_k in range(beam_width):
                decoded_t = indexes[new_k]
                log_p = prob[new_k].item()
                node = BeamSearchNode(priority_node, decoded_t, priority_node.logp * log_p, priority_node.leng + 1)
                nodes.put(node)
            qsize += beam_width - 1

        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(topk)]

        # generate the prob-map
        prob_graph = torch.zeros(T,C,device=encoder_outputs.device)
        total_prob = 0.
        for endnode in sorted(endnodes, key=lambda x: x.eval()):
            single_path = []
            single_path.append(endnode.char_id)
            length[idx]=endnode.leng-1
            cur_prob = torch.tensor(endnode.logp,device=device)

            # back trace
            while endnode.prev_node != None:
                endnode = endnode.prev_node
                single_path.append(endnode.char_id)

            single_path = torch.stack(single_path[::-1][1:])
            path_len = single_path.shape[0]
            single_path = torch.cat((single_path, torch.zeros(max_len-path_len,device=device)), dim=0).long()
            total_prob += cur_prob
            prob_graph[torch.arange(T),single_path] += cur_prob

        if cur_prob>=path_thred:
            alpha_weight[idx] = alpha
        prob_graph = prob_graph / total_prob
        batch_prob_graph.append(prob_graph)

    return torch.stack(batch_prob_graph), length.long(), alpha_weight # N,26,37


