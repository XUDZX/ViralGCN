import torch
from torch import nn
import random
import pickle


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
with open('emb.pickle', 'rb') as emb:
    embeddings = pickle.load(emb)
    emb.close()

def random_pick(some_list, probabilities,num_sample):
    x = random.uniform(0,1)

    samplelist=[]
    while len(samplelist)<num_sample:
        cumulative_probability = 0.0
        for item, item_probability in zip(some_list, probabilities):
            cumulative_probability += item_probability
            if x < cumulative_probability:
                break

        samplelist.append(item)
        index=some_list.index(item)
        del some_list[index]
        del probabilities[index]
        probsum=sum(probabilities)
        probabilities=list(map(lambda x:x/probsum,probabilities))
    return samplelist



class Sagelayer(nn.Module):
    def __init__(self,layer_size,output_size,gru=False):
        super(Sagelayer,self).__init__()
        self.layer_size=layer_size
        self.output_size=output_size
        self.gru=gru
        self.weight=nn.Parameter(torch.FloatTensor(self.layer_size*3,self.output_size))
        self.init_params()

    def init_params(self):
        for param in self.parameters():
            nn.init.xavier_uniform_(param)

    def forward(self, node_feats, out_neighs,in_neighs):

        if not self.gru:
            combined = torch.cat([node_feats, out_neighs,in_neighs], dim=2)
        else:
            pass
        combined = torch.nn.functional.relu(torch.matmul(combined,self.weight))
        return combined

class temporal_layer(nn.Module):
    def __init__(self,input_size,output_size):
        super(temporal_layer, self).__init__()

        self.gru=torch.nn.GRU(input_size,output_size,batch_first=True)

    def forward(self,time_masks,nodes_states):
        temporal_states=torch.matmul(time_masks,nodes_states)
        outputs=self.gru(temporal_states)
        outputs=torch.nn.functional.relu(outputs[0])
        return outputs



class ViralGCN(nn.Module):
    def __init__(self,num_layers,input_size,hidden_size,output_size,num_time_intervals,aggr_func='mean',num_sam_nodes=500,gru=False):
        super(ViralGCN,self).__init__()
        self.num_layers=num_layers
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.node_embs=nn.Embedding.from_pretrained(embeddings,freeze=True)
        self.node_embs.weight.requires_grad = False
        self.num_intervals = num_time_intervals
        self.aggr_func=aggr_func
        self.num_sam_nodes=100+100*num_sam_nodes
        self.decay_effect = nn.Parameter(torch.FloatTensor(self.num_intervals))#.to(device)
        self.dense_layer1=nn.Linear(self.output_size,12)#.to(device)
        self.dense_layer2=nn.Linear(12,1)#.to(device)


        for i in range(num_layers):
            layer_size = hidden_size if i != 0 else input_size
            setattr(self,'sagelayer'+str(i+1),Sagelayer(layer_size,hidden_size,gru))#.to(device))

        self.temporal_layer=temporal_layer(self.hidden_size,self.output_size)#.to(device)

        self.init_params()

    def init_params(self):
        for param in self.parameters():
            nn.init.uniform_(param)

    def forward(self,adj_dic_list):
        nodes=[]
        masks_out=[]
        masks_in=[]
        time_masks=[]
        for adj_dic in adj_dic_list:
            samp_nodes,mask_out,mask_in,time_mask=self.sample_nodes(adj_dic,self.num_sam_nodes,self.num_intervals)
            nodes.append(samp_nodes)
            masks_out.append(mask_out)
            masks_in.append(mask_in)
            time_masks.append(time_mask)
        nodes=torch.stack(nodes,0)
        masks_out=torch.stack(masks_out,0)
        masks_in=torch.stack(masks_in,0)
        time_masks=torch.stack(time_masks,0)
        nodes_state=self.node_embs(nodes)

        # masks_in=masks_in.to(device)
        # masks_out=masks_out.to(device)
        # nodes_state=nodes_state.to(device)
        # time_masks=time_masks.to(device)
        for i in range(self.num_layers):
            out_neighs,in_neighs=self.aggregate(nodes_state,masks_out,masks_in,self.aggr_func)
            # out_neighs=out_neighs.to(device)
            # in_neighs=in_neighs.to(device)
            sage_layer = getattr(self, 'sagelayer' + str(i+1))
            nodes_state=sage_layer(nodes_state,out_neighs,in_neighs)
        gru_outputs=self.temporal_layer(time_masks,nodes_state)
        temporal_states=torch.reshape(gru_outputs,(-1,self.output_size,self.num_intervals))
        temporal_states=torch.nn.functional.relu(torch.matmul(temporal_states,self.decay_effect))

        popularity_growth=torch.nn.functional.relu(self.dense_layer1(temporal_states))
        popularity_growth=torch.nn.functional.relu(self.dense_layer2(popularity_growth))
        return popularity_growth,self.num_sam_nodes
        # return nodes,nodes_state,popularity_growth, temporal_states

    def sample_nodes(self, adj_dic,num_sam_nodes,num_intervals):
        root_nodes=list(adj_dic.keys())
        if len(root_nodes)<=num_sam_nodes:
            pad = torch.nn.ConstantPad1d((0, num_sam_nodes - len(root_nodes)), 0)
            root_nodes=pad(torch.tensor(root_nodes,dtype=torch.int32))
        else:
            degree=[]
            for node in root_nodes:
                out_degree=len(adj_dic[node])
                degree.append(out_degree)
            degree_sum=sum(degree)
            prob = list(map(lambda x: x / degree_sum, degree))
            root_nodes=torch.tensor(random_pick(root_nodes,prob,num_sam_nodes),dtype=torch.int32)

        sample_nodes=root_nodes.tolist()
        mask_out=torch.zeros(num_sam_nodes,num_sam_nodes)
        row_indices=[]
        column_indices=[]
        for node_i in sample_nodes:
            if node_i!=0:
                i_index=sample_nodes.index(node_i)
                for node_j in sample_nodes:
                    if node_j!=0:
                        j_index=sample_nodes.index(node_j)
                        if node_j in adj_dic[node_i][1:]:
                            row_indices.append(i_index)
                            column_indices.append(j_index)
        mask_out[row_indices,column_indices]=1
        mask_in=mask_out.t()

        time_mask=torch.zeros(num_intervals,num_sam_nodes)
        row_indices=[]
        column_indices=[]
        time_interval=10800/num_intervals
        for interval in range(num_intervals):
            for node in sample_nodes:
                if node!=0:
                    if time_interval*interval <= adj_dic[node][0] < time_interval*(interval+1):
                        row_indices.append(interval)
                        index=sample_nodes.index(node)
                        column_indices.append(index)
        time_mask[row_indices,column_indices]=1



        return root_nodes,mask_out,mask_in,time_mask

    def aggregate(self,nodes_state,masks_out,masks_in,aggr_func):
        if aggr_func=='mean':
            masks_out_s=masks_out.sum(2,True)
            masks_out=masks_out.div(masks_out_s)
            masks_out=torch.where(torch.isnan(masks_out), torch.full_like(masks_out, 0), masks_out)
            out_neighs=torch.matmul(masks_out,nodes_state)
            masks_in_s = masks_in.sum(2, True)
            masks_in = masks_in.div(masks_in_s)
            masks_in = torch.where(torch.isnan(masks_in), torch.full_like(masks_in, 0), masks_in)
            in_neighs = torch.matmul(masks_in, nodes_state)
            return out_neighs,in_neighs
        if aggr_func=='max':
            i = 0
            out_aggre = []
            for mask_out in masks_out:
                indexs = [x.nonzero() for x in mask_out == 1]

                for feat in [nodes_state[i][x.squeeze()] for x in indexs]:
                        if len(feat.size()) == 1:
                            out_aggre.append(feat.view(1, -1))
                        elif len(feat) == 0:
                            out_aggre.append(torch.zeros((1, feat.size()[1]), dtype=torch.float32))
                        else:
                            out_aggre.append(torch.max(feat, 0)[0].view(1, -1))
                i += 1
            out_neighs = torch.cat(out_aggre, 0)
            batch_size=masks_out.size()[0]
            num_sam_nodes=masks_out.size()[1]
            out_neighs = torch.reshape(out_neighs, (batch_size, num_sam_nodes, -1))
            i = 0
            in_aggre = []
            for mask_out in masks_out:
                indexs = [x.nonzero() for x in mask_out == 1]

                for feat in [nodes_state[i][x.squeeze()] for x in indexs]:
                    if len(feat.size()) == 1:
                        in_aggre.append(feat.view(1, -1))
                    elif len(feat) == 0:
                        in_aggre.append(torch.zeros((1, feat.size()[1]), dtype=torch.float32))
                    else:
                        in_aggre.append(torch.max(feat, 0)[0].view(1, -1))
                i += 1
            in_neighs = torch.cat(out_aggre, 0)
            in_neighs = torch.reshape(in_neighs, (batch_size, num_sam_nodes, -1))
            return out_neighs,in_neighs




