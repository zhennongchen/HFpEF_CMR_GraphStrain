import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, GCNConv

def edge_index_cal():
    ### build edge index
    edge_index_raw = [[1,2],[1,6],[1,7],
                    [2,1],[2,3],[2,8],
                    [3,2],[3,4],[3,9],
                    [4,3],[4,5],[4,10],
                    [5,4],[5,6],[5,11],
                    [6,1],[6,5],[6,12],
                    [7,1],[7,8],[7,12],[7,13],
                    [8,2],[8,7],[8,9],[8,13],[8,14],
                    [9,3],[9,8],[9,10],[9,14],[9,15],
                    [10,4],[10,9],[10,11],[10,15],
                    [11,5],[11,10],[11,12],[11,15],[11,16],
                    [12,6],[12,7],[12,11],[12,13],[12,16],
                    [13,7],[13,8],[13,12],[13,14],[13,16],
                    [14,8],[14,9],[14,13],[14,15],
                    [15,9],[15,10],[15,11],[15,14],[15,16],
                    [16,11],[16,12],[16,13],[16,15]]
    edge_index_raw_0based = [[i-1, j-1] for i, j in edge_index_raw]

    index_chosen = []
    source_nodes = []; target_nodes = []
    for index in range(0,len(edge_index_raw_0based)):
        if index in index_chosen:
            continue
        edge_pair = edge_index_raw_0based[index]
        counter_pair = [edge_pair[1], edge_pair[0]]
        # what is the index of counter_pair in edge_index_raw_0based
        counter_index = edge_index_raw_0based.index(counter_pair)
        index_chosen.append(index); index_chosen.append(counter_index)
        source_nodes.append(edge_pair[0]); source_nodes.append(counter_pair[0])
        target_nodes.append(edge_pair[1]); target_nodes.append(counter_pair[1])
    edge_index = torch.LongTensor([source_nodes, target_nodes])
    print(edge_index.shape)
    edge_index_ecc = torch.clone(edge_index)

    ## also build edge index for Err
    edge_index_np = edge_index.cpu().numpy()  

    mask = (edge_index_np[0] <= 11) & (edge_index_np[1] <= 11)
    filtered_edge_index_np = edge_index_np[:, mask]

    edge_index_err = torch.LongTensor(filtered_edge_index_np)
    print(edge_index_err.shape)

    return edge_index_ecc, edge_index_err


class STGCN(nn.Module):
    def __init__(self, strain ,GCN_type = 'ChebConv', Cheb_K = 2, aha_num = 16, tf_num = 25, temporal_out = 32, hidden_size=128, dropout_rate=0.3, edge_index_ecc=None , edge_index_err = None, get_latent_layer = False):
        super().__init__()
        self.strain = strain
        aha_num = aha_num if self.strain != 'Err' else aha_num - 4
        # determine which GCN to use
        self.GCN_type = GCN_type; self.Cheb_K = Cheb_K
        # set the edge index matrix
        if self.strain == 'Ecc' or self.strain == 'pad':
            self.edge_index = edge_index_ecc
        elif self.strain == 'Err':
            self.edge_index = edge_index_err
        else:
            ValueError('strain should be either Ecc or Err or pad')

        if self.strain == 'pad': # anneal Ecc and Err in each node
            tf_num = tf_num * 2
            
        # determine whether we need temporal_conv
        self.temporal_conv = nn.Conv1d(in_channels=1, out_channels=temporal_out, kernel_size=3, padding=1) if temporal_out > 0 else nn.Identity()
        gcn_out = 64 if temporal_out > 0 else 32
        temporal_out_channels = temporal_out * tf_num if temporal_out > 0 else tf_num

        # GCN layer
        self.gcn1 = GCNConv(temporal_out_channels, gcn_out) if self.GCN_type == 'GCNConv' else ChebConv(temporal_out_channels, gcn_out, K=self.Cheb_K)

        self.fc = nn.Linear(aha_num * gcn_out, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.final_layer = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
        self.get_latent_layer = get_latent_layer

    def forward(self, x):  # x: [B, 16, 25]
        B, V, T = x.shape
        self.edge_index = self.edge_index.to(x.device)

        # Step 1: Temporal conv → each node independently
        x = x.unsqueeze(2)  # [B, 16, 1, 25]
        x = x.reshape(B * V, 1, T)  # [B*16, 1, 25]
        x = self.temporal_conv(x)  # [B*16, 32, 25]
        x = x.reshape(B, V, -1)  # Flatten last two dims: [B, 16, 32*25=800]
        # Step 2: GCN over 16 nodes
        out = []
        for i in range(B):
            h = self.gcn1(x[i], self.edge_index)  # [16, 800] → [16, gcn_out]
            out.append(h)

        x = torch.stack(out, dim=0)  # [B, 16, gcn_out]
        x = x.view(B, -1)  # flatten → [B, 16 * gcn_out]

        latent_layer = self.fc(x)
        x = self.relu(latent_layer)
        if self.dropout.p > 0:
            x = self.dropout(x)
        x = self.final_layer(x)

        if self.get_latent_layer == False:
            return self.sigmoid(x)
        else:
            return self.sigmoid(x), latent_layer
    

class DualSTGCN(nn.Module):
    def __init__(self, 
                 fusion_method, # 'gated', 'concat'
                 GCN_type = 'ChebConv', 
                 Cheb_K = 2,
                 aha_num=16,
                tf_num=25,
                 temporal_out=32,
                 hidden_size=128, 
                 dropout_rate=0.3, 
                 edge_index_ecc=None, 
                 edge_index_err=None,
                 get_latent_layer=False):
        super().__init__()
        self.strain = 'both'
        # determine which fusion method to use
        self.fusion_method = fusion_method
        # determine which GCN to use
        self.GCN_type = GCN_type; self.Cheb_K = Cheb_K
        # set the edge index matrix
        self.edge_index_ecc = edge_index_ecc
        self.edge_index_err = edge_index_err
        
        # Temporal Conv1D for Ecc and Err
        self.temporal_conv_ecc = nn.Conv1d(1, temporal_out, kernel_size=3, padding=1) if temporal_out > 0 else nn.Identity()
        self.temporal_conv_err = nn.Conv1d(1, temporal_out, kernel_size=3, padding=1) if temporal_out > 0 else nn.Identity()
        gcn_out = 64 if temporal_out > 0 else 32
        temporal_out_channels = temporal_out * tf_num if temporal_out > 0 else tf_num

        # GCN layers for Ecc and Err
        self.gcn_ecc = GCNConv(temporal_out_channels, gcn_out) if self.GCN_type == 'GCNConv' else ChebConv(temporal_out_channels, gcn_out, K=self.Cheb_K)
        self.gcn_err = GCNConv(temporal_out_channels, gcn_out) if self.GCN_type == 'GCNConv' else ChebConv(temporal_out_channels, gcn_out, K=self.Cheb_K)

        # for Cross-Attention: Ecc queries Err
        # if self.fusion_method == 'cross_attention':
        #     self.query_proj = nn.Linear(gcn_out, gcn_out)
        #     self.key_proj = nn.Linear(gcn_out, gcn_out)
        #     self.value_proj = nn.Linear(gcn_out, gcn_out)
        if self.fusion_method == 'gated':
            self.ecc_proj = nn.Linear(aha_num * gcn_out, hidden_size * 2)
            self.err_proj = nn.Linear((aha_num-4) * gcn_out, hidden_size * 2)
            self.attn_score = nn.Linear(hidden_size*2, 1)
        elif self.fusion_method == 'concat':
            input_dim = aha_num * gcn_out + (aha_num-4) * gcn_out

        # Classifier
        # if self.fusion_method == 'cross_attention':
        #     self.fc1 = nn.Linear(aha_num * gcn_out, hidden_size)
        if self.fusion_method == 'gated':
            self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        elif self.fusion_method == 'concat':
            self.fc1 = nn.Linear(input_dim, hidden_size)
        else:
            ValueError('fusion_method should be either cross_attention, gated, or concat')
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, 1) if self.fusion_method != 'gated' else nn.Linear(hidden_size * 2, 1)
        self.sigmoid = nn.Sigmoid()

        self.get_latent_layer = get_latent_layer

    def forward(self, ecc, err):  # ecc: [B, 16, 25], err: [B, 12, 25]
        B, V_ecc, T = ecc.shape

        # ----- Ecc Branch -----
        ecc = ecc.unsqueeze(2).reshape(B * V_ecc, 1, T)                      # [B*16, 1, 25]
        ecc = self.temporal_conv_ecc(ecc).reshape(B, V_ecc, -1)              # [B, 16, 800]
        edge_index_ecc = self.edge_index_ecc.to(ecc.device)
        ecc_out = torch.stack([
            self.gcn_ecc(ecc[i], edge_index_ecc) for i in range(B)
        ], dim=0)                                                         # [B, 16, gcn_out]

        # ----- Err Branch -----
        _,V_err,_ = err.shape   
        err = err.unsqueeze(2).reshape(B * V_err, 1, T)                      # [B*12, 1, 25]
        err = self.temporal_conv_err(err).reshape(B, V_err, -1)              # [B, 12, 800]
        edge_index_err = self.edge_index_err.to(err.device)
        err_out = torch.stack([
            self.gcn_err(err[i], edge_index_err) for i in range(B)
        ], dim=0)                                                         # [B, 12, gcn_out]

        # ----- Fusion -----
        # if self.fusion_method == 'cross_attention':
        #     Q = self.query_proj(ecc_out)                                      # [B, 16, d]
        #     K = self.key_proj(err_out)                                        # [B, 12, d]
        #     V = self.value_proj(err_out)                                      # [B, 12, d]

        #     attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.shape[-1] ** 0.5)  # [B, 16, 12]
        #     attn_weights = F.softmax(attn_scores, dim=-1)                     # [B, 16, 12]
        #     fused = torch.matmul(attn_weights, V)                             # [B, 16, d]
        #     fused = fused.reshape(B, -1)                                      # [B, 16*d]
        if self.fusion_method == 'gated':
            ecc_global = self.ecc_proj(ecc_out.reshape(B, -1))                 # [B, 16*d] → [B, 2*hidden_size]
            err_global = self.err_proj(err_out.reshape(B, -1))                 # [B, 12*d] → [B, 2*hidden_size]
            attn =  torch.sigmoid(self.attn_score(torch.tanh(ecc_global + err_global)))  # [B, 1]
            fused = attn * ecc_global + (1 - attn) * err_global               # [B, 2*hidden_size]
        elif self.fusion_method == 'concat':
            fused = torch.cat([ecc_out.reshape(B, -1), err_out.reshape(B, -1)], dim=1)
        
        
        # ----- Classifier -----
        if self.fusion_method == 'cross_attention' or self.fusion_method == 'concat':
            latent_layer = self.fc1(fused)   
        elif self.fusion_method == 'gated':
            latent_layer = fused                                            

        x = F.relu(latent_layer)
        x = self.dropout(x)
        x = self.fc2(x)
        if self.get_latent_layer == False:
            return self.sigmoid(x)
        else:
            return self.sigmoid(x), latent_layer

class DualSTGCN_w_EHR(nn.Module):
    def __init__(self, 
                 fusion_method, # 'gated', 'concat'
                 ehr_dim,
                 ehr_out = 64,
                 GCN_type = 'ChebConv', 
                 Cheb_K = 2,
                 aha_num=16,
                tf_num=25,
                 temporal_out=32,
                 hidden_size=128, 
                 dropout_rate=0.3, 
                 edge_index_ecc=None, 
                 edge_index_err=None,
                 get_latent_layer=False):
        super().__init__()
        self.strain = 'both'
        # determine which fusion method to use
        self.fusion_method = fusion_method
        # determine which GCN to use
        self.GCN_type = GCN_type; self.Cheb_K = Cheb_K
        # set the edge index matrix
        self.edge_index_ecc = edge_index_ecc
        self.edge_index_err = edge_index_err
        
        # Temporal Conv1D for Ecc and Err
        self.temporal_conv_ecc = nn.Conv1d(1, temporal_out, kernel_size=3, padding=1) if temporal_out > 0 else nn.Identity()
        self.temporal_conv_err = nn.Conv1d(1, temporal_out, kernel_size=3, padding=1) if temporal_out > 0 else nn.Identity()
        gcn_out = 64 if temporal_out > 0 else 32
        temporal_out_channels = temporal_out * tf_num if temporal_out > 0 else tf_num

        # GCN layers for Ecc and Err
        self.gcn_ecc = GCNConv(temporal_out_channels, gcn_out) if self.GCN_type == 'GCNConv' else ChebConv(temporal_out_channels, gcn_out, K=self.Cheb_K)
        self.gcn_err = GCNConv(temporal_out_channels, gcn_out) if self.GCN_type == 'GCNConv' else ChebConv(temporal_out_channels, gcn_out, K=self.Cheb_K)

        if self.fusion_method == 'gated':
            self.ecc_proj = nn.Linear(aha_num * gcn_out, hidden_size * 2)
            self.err_proj = nn.Linear((aha_num-4) * gcn_out, hidden_size * 2)
            self.attn_score = nn.Linear(hidden_size*2, 1)
        elif self.fusion_method == 'concat':
            input_dim = aha_num * gcn_out + (aha_num-4) * gcn_out

        # EHR branch
        self.ehr_proj = nn.Sequential(
            nn.Linear(ehr_dim, ehr_out),
            nn.ReLU(),
            nn.Dropout(dropout_rate))

        # Classifier
        if self.fusion_method == 'gated':
            self.fc1 = nn.Linear(hidden_size * 2 + ehr_out, hidden_size)
        elif self.fusion_method == 'concat':
            self.fc1 = nn.Linear(input_dim + ehr_out, hidden_size)
        else:
            ValueError('fusion_method should be either cross_attention, gated, or concat')
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, 1) if self.fusion_method != 'gated' else nn.Linear(hidden_size * 2 + ehr_out, 1)
        self.sigmoid = nn.Sigmoid()

        self.get_latent_layer = get_latent_layer

    def forward(self, ecc, err, ehr):  # ecc: [B, 16, 25], err: [B, 12, 25]
        B, V_ecc, T = ecc.shape

        # ----- Ecc Branch -----
        ecc = ecc.unsqueeze(2).reshape(B * V_ecc, 1, T)                      # [B*16, 1, 25]
        ecc = self.temporal_conv_ecc(ecc).reshape(B, V_ecc, -1)              # [B, 16, 800]
        edge_index_ecc = self.edge_index_ecc.to(ecc.device)
        ecc_out = torch.stack([
            self.gcn_ecc(ecc[i], edge_index_ecc) for i in range(B)
        ], dim=0)                                                         # [B, 16, gcn_out]

        # ----- Err Branch -----
        _,V_err,_ = err.shape   
        err = err.unsqueeze(2).reshape(B * V_err, 1, T)                      # [B*12, 1, 25]
        err = self.temporal_conv_err(err).reshape(B, V_err, -1)              # [B, 12, 800]
        edge_index_err = self.edge_index_err.to(err.device)
        err_out = torch.stack([
            self.gcn_err(err[i], edge_index_err) for i in range(B)
        ], dim=0)                                                         # [B, 12, gcn_out]

        # ----- Fusion -----
        if self.fusion_method == 'gated':
            ecc_global = self.ecc_proj(ecc_out.reshape(B, -1))                 # [B, 16*d] → [B, 2*hidden_size]
            err_global = self.err_proj(err_out.reshape(B, -1))                 # [B, 12*d] → [B, 2*hidden_size]
            attn =  torch.sigmoid(self.attn_score(torch.tanh(ecc_global + err_global)))  # [B, 1]
            fused = attn * ecc_global + (1 - attn) * err_global               # [B, 2*hidden_size]
        elif self.fusion_method == 'concat':
            fused = torch.cat([ecc_out.reshape(B, -1), err_out.reshape(B, -1)], dim=1)

        # ----- EHR branch -----
        ehr_proj = self.ehr_proj(ehr)

         # ----- Combine -----
        combined = torch.cat([fused, ehr_proj], dim=1)  
        
        # ----- Classifier -----
        if self.fusion_method == 'concat':
            latent_layer = self.fc1(combined)   
        elif self.fusion_method == 'gated':
            latent_layer = combined                                          

        x = F.relu(latent_layer)
        x = self.dropout(x)
        x = self.fc2(x)
        if self.get_latent_layer == False:
            return self.sigmoid(x)
        else:
            return self.sigmoid(x), latent_layer
