import torch
from dataloader import BasicDataset
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import SGConv
import odeblock as ode
import random
import scipy.sparse as sp
from layer import BiGNNLayer, SparseDropout
from init import xavier_normal_initialization

import ode as ode
import ode1 as ode1

class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError
    
class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError
    
class PureMF(BasicModel):
    def __init__(self, 
                 args,
                 dataset:BasicDataset):
        super(PureMF, self).__init__()
        self.args = args
        self.num_users  = dataset.n_users
        self.num_items  = dataset.m_items
        self.latent_dim = self.args.recdim #config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.__init_weight()
        
    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        print("using Normal distribution N(0,1) initialization for PureMF")
        
    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)
    
    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb   = self.embedding_item(pos.long())
        neg_emb   = self.embedding_item(neg.long())
        pos_scores= torch.sum(users_emb*pos_emb, dim=1)
        neg_scores= torch.sum(users_emb*neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) + 
                          pos_emb.norm(2).pow(2) + 
                          neg_emb.norm(2).pow(2))/float(len(users))
        return loss, reg_loss
        
    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb*items_emb, dim=1)
        return self.f(scores)


class SGCN(nn.Module):
    def __init__(self,
                 args,
                 dataset:BasicDataset):
        super(SGCN, self).__init__()
        self.args = args
        self.num_users  = dataset.n_users
        self.num_items  = dataset.m_items
        train_ui_mat = dataset.UserItemNet
        userind_arr, itemind_arr = train_ui_mat.nonzero()
        itemind_arr_added = [itemind+self.num_users for itemind in itemind_arr]
        edge_index = torch.tensor([np.concatenate((userind_arr, itemind_arr_added)),
                                np.concatenate((itemind_arr_added, userind_arr))], dtype=torch.long)

        self.latent_dim = self.args.recdim #config['latent_dim_rec']
        self.__init_weight()
        all_emb = torch.cat([self.embedding_user.weight, self.embedding_item.weight])
        self.graph_data = Data(x=all_emb, edge_index=edge_index).to(self.args.device)
        self.sgcn_conv = SGConv(in_channels=self.latent_dim, out_channels=self.latent_dim, K=args.layer, cached=True, add_self_loops=False).to(self.args.device)

    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim).to(self.args.device)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim).to(self.args.device)

    def computer(self):
        sgcn_out = self.sgcn_conv(self.graph_data.x, self.graph_data.edge_index)
        users, items = torch.split(sgcn_out, [self.num_users, self.num_items])
        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = torch.matmul(users_emb, items_emb.t())
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) +
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss

        

class LightGCN(BasicModel):
    def __init__(self, 
                 args,
                 dataset:BasicDataset):
        super(LightGCN, self).__init__()
        self.args = args
        self.dataset : dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.args.recdim #self.config['latent_dim_rec']
        self.n_layers = self.args.layer #self.config['lightGCN_n_layers']
        self.keep_prob = self.args.keepprob #self.config['keep_prob']
        self.A_split = False #self.config['A_split']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph(self.dataset.UserItemNet)

    def reset_all(self):
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        self.Graph = self.dataset.getSparseGraph(self.dataset.UserItemNet)

    def reset_all_uuii(self):
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        self.Graph = self.dataset.getSparseGraph(self.dataset.UserItemNet, include_uuii=True)


    def reset_graph(self):
        self.Graph = self.dataset.getSparseGraph(self.dataset.UserItemNet)

        # print("save_txt")
    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    def computer(self):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.args.dropout:
            if self.training:
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph    
        
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getUsersUsers(self, users):
        all_users, _ = self.computer()
        users_emb = all_users[users.long()]

        return self.f(torch.matmul(users_emb, all_users.t()))

    def getItemsItems(self, items):
        _, all_items = self.computer()
        items_emb = all_items[items.long()]
        
        return self.f(torch.matmul(items_emb, all_items.t()))

    def getItemsRating(self, items):
        all_users, all_items = self.computer()
        items_emb = all_items[items.long()]
        users_emb = all_users
        item_rating = self.f(torch.matmul(items_emb, users_emb.t()))
        return item_rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
       
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss
       
    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        #all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma



class UltraGCN(nn.Module):
    def __init__(self, args, dataset, constraint_mat, ii_constraint_mat, ii_neighbor_mat):
        super(UltraGCN, self).__init__()
        self.args = args
        self.user_num  = dataset.n_users
        self.item_num  = dataset.m_items
        self.embedding_dim = self.args.recdim
        self.w1 = self.args.w1
        self.w2 = self.args.w2
        self.w3 = self.args.w3
        self.w4 = self.args.w4

        self.negative_weight = self.args.negative_weight
        self.gamma = self.args.decay
        self.lambda_ = self.args.lambda_Iweight

        self.user_embeds = nn.Embedding(self.user_num, self.embedding_dim)
        self.item_embeds = nn.Embedding(self.item_num, self.embedding_dim)

        self.constraint_mat = constraint_mat
        self.ii_constraint_mat = ii_constraint_mat
        self.ii_neighbor_mat = ii_neighbor_mat


    def get_omegas(self, users, pos_items, neg_items):
        device = self.args.device
        #self.constraint_mat['beta_uD'] = self.constraint_mat['beta_uD'].to(device)
        #self.constraint_mat['beta_iD'] = self.constraint_mat['beta_iD'].to(device)
        if self.w2 > 0:
            pos_weight = torch.mul(self.constraint_mat['beta_uD'][users], self.constraint_mat['beta_iD'][pos_items]).to(device)
            pow_weight = self.w1 + self.w2 * pos_weight
        else:
            pos_weight = self.w1 * torch.ones(len(pos_items)).to(device)

        # users = (users * self.item_num).unsqueeze(0)
        if self.w4 > 0:
            neg_weight = torch.mul(self.constraint_mat['beta_uD'][users], self.constraint_mat['beta_iD'][neg_items]).to(device)
            neg_weight = self.w3 + self.w4 * neg_weight
        else:
            neg_weight = self.w3 * torch.ones(neg_items.size(0) * neg_items.size(1)).to(device)


        weight = torch.cat((pow_weight, neg_weight))
        return weight

    def cal_loss_L(self, users, pos_items, neg_items, omega_weight):
        device = self.args.device
        user_embeds = self.user_embeds(users.to(device))
        pos_embeds = self.item_embeds(pos_items.to(device))
        neg_embeds = self.item_embeds(neg_items.to(device))

        pos_scores = (user_embeds * pos_embeds).sum(dim=-1) # batch_size
        user_embeds = user_embeds.unsqueeze(1)
        neg_scores = (user_embeds * neg_embeds).sum(dim=-1) # batch_size * negative_num

        neg_labels = torch.zeros(neg_scores.size()).to(device)
        neg_loss = F.binary_cross_entropy_with_logits(neg_scores, neg_labels, weight = omega_weight[len(pos_scores):], reduction='none').mean(dim = -1)

        pos_labels = torch.ones(pos_scores.size()).to(device)
        pos_loss = F.binary_cross_entropy_with_logits(pos_scores, pos_labels, weight = omega_weight[:len(pos_scores)], reduction='none')

        loss = pos_loss + neg_loss * self.negative_weight

        return loss.sum()

    def cal_loss_I(self, users, pos_items):
        device = self.args.device
        #self.ii_neighbor_mat = self.ii_neighbor_mat.to(device)
        neighbor_embeds = self.item_embeds(self.ii_neighbor_mat[pos_items].to(device))    # len(pos_items) * num_neighbors * dim
        #self.ii_constraint_mat = self.ii_constraint_mat.to(device)
        sim_scores = self.ii_constraint_mat[pos_items].to(device)     # len(pos_items) * num_neighbors
        user_embeds = self.user_embeds(users.to(device)).unsqueeze(1)

        loss = -sim_scores * (user_embeds * neighbor_embeds).sum(dim=-1).sigmoid().log()

        # loss = loss.sum(-1)
        return loss.sum()

    def norm_loss(self):
        loss = 0.0
        for parameter in self.parameters():
            loss += torch.sum(parameter ** 2)
        return loss / 2

    def forward(self, users, pos_items, neg_items):
        omega_weight = self.get_omegas(users, pos_items, neg_items)

        loss = self.cal_loss_L(users, pos_items, neg_items, omega_weight)
        loss += self.gamma * self.norm_loss()
        loss += self.lambda_ * self.cal_loss_I(users, pos_items)
        return loss

    def getUsersRating(self, users):
        users_emb = self.user_embeds(users.long())
        items_emb = self.item_embeds.weight
        return users_emb.mm(items_emb.t())



class LTOCF(BasicModel):
    def __init__(self, 
                 args,
                 dataset:BasicDataset):
        super(LTOCF, self).__init__()
        self.args = args
        self.dataset : dataloader.BasicDataset = dataset
        self.__init_weight()
        self.__init_ode()


    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.args.recdim
        self.n_layers = self.args.layer
        self.keep_prob = self.args.keepprob
        self.A_split = False
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        if self.args.pretrain == 0:
#             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
#             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
#             print('use xavier initilizer')
# random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
        else:
            # self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            # self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph(self.dataset.UserItemNet)
    
    def __init_ode(self):
        self.time_split = self.args.time_split # init the number of time split
        if self.args.learnable_time == True:
            
            self.odetimes = ode.ODETimeSetter(self.time_split, self.args.K)
            self.odetime_1 = [self.odetimes[0]]
            self.odetime_2 = [self.odetimes[1]]
            self.odetime_3 = [self.odetimes[2]]
            self.ode_block_test_1 = ode.ODEBlockTimeFirst(ode.ODEFunction(self.Graph),self.time_split, self.args.solver)
            self.ode_block_test_2 = ode.ODEBlockTimeMiddle(ode.ODEFunction(self.Graph),self.time_split, self.args.solver)
            self.ode_block_test_3 = ode.ODEBlockTimeMiddle(ode.ODEFunction(self.Graph),self.time_split, self.args.solver)
            self.ode_block_test_4 = ode.ODEBlockTimeLast(ode.ODEFunction(self.Graph),self.time_split, self.args.solver)
        else:
            self.odetime_splitted = ode.ODETimeSplitter(self.time_split, self.args.K)
            self.ode_block_1 = ode.ODEBlock(ode.ODEFunction(self.Graph), self.args.solver, 0, self.odetime_splitted[0])
            self.ode_block_2 = ode.ODEBlock(ode.ODEFunction(self.Graph), self.args.solver, self.odetime_splitted[0], self.odetime_splitted[1])
            self.ode_block_3 = ode.ODEBlock(ode.ODEFunction(self.Graph), self.args.solver, self.odetime_splitted[1], self.odetime_splitted[2])
            self.ode_block_4 = ode.ODEBlock(ode.ODEFunction(self.Graph), self.args.solver, self.odetime_splitted[2], self.args.K)

    def get_time(self):
        ode_times=list(self.odetime_1)+ list(self.odetime_2)+ list(self.odetime_3)
        return ode_times
        
    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    def computer(self):
        """
        propagate methods for LT-NCF
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]

        """
        layers
        """
        if self.args.learnable_time == True:
            out_1 = self.ode_block_test_1(all_emb, self.odetime_1)
            if self.args.dual_res == False:
                out_1 = out_1 - all_emb
            embs.append(out_1)

            out_2 = self.ode_block_test_2(out_1, self.odetime_1, self.odetime_2)
            if self.args.dual_res == False:
                out_2 = out_2 - out_1
            embs.append(out_2)

            out_3 = self.ode_block_test_3(out_2, self.odetime_2, self.odetime_3)
            if self.args.dual_res == False:
                out_3 = out_3 - out_2
            embs.append(out_3)            

            out_4 = self.ode_block_test_4(out_3, self.odetime_3)
            if self.args.dual_res == False:
                out_4 = out_4 - out_3
            embs.append(out_4)
            
        elif self.args.learnable_time == False:
            all_emb_1 = self.ode_block_1(all_emb)
            all_emb_1 = all_emb_1 - all_emb
            embs.append(all_emb_1)
            all_emb_2 = self.ode_block_2(all_emb_1)
            all_emb_2 = all_emb_2 - all_emb_1
            embs.append(all_emb_2)
            all_emb_3 = self.ode_block_3(all_emb_2)
            all_emb_3 = all_emb_3 - all_emb_2
            embs.append(all_emb_3)
            all_emb_4 = self.ode_block_4(all_emb_3)
            all_emb_4 = all_emb_4 - all_emb_3
            embs.append(all_emb_4)

        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)        

        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss
       
    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma
    



class LayerGCN(BasicModel):
    def __init__(self, args, dataset):
        super(LayerGCN, self).__init__()
        self.args = args
        self.dataset : dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        # load dataset info
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.interaction_matrix = self.dataset.UserItemNet.tocoo()
        self.f = nn.Sigmoid()
        # load parameters info
        self.latent_dim = self.args.recdim  # int type:the embedding size of lightGCN
        self.n_layers = self.args.layer  # int type:the layer num of lightGCN
        self.reg_weight = 0.0001  # float32 type: the weight decay for l2 normalizaton
        self.dropout = self.args.keepprob

        self.n_nodes = self.num_users + self.num_items

        # define layers and loss
        self.user_embeddings = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.num_users, self.latent_dim)))
        self.item_embeddings = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.num_items, self.latent_dim)))

        # normalized adj matrix
        self.Graph = self.dataset.getSparseGraph(self.dataset.UserItemNet)
        self.masked_adj = None
        self.forward_adj = None
        self.pruning_random = False

        # edge prune
        self.edge_indices, self.edge_values = self.get_edge_info()
        self.edge_indices = self.edge_indices.to('cuda')


    def pre_epoch_processing(self):
        if self.dropout <= .0:
            self.masked_adj = self.Graph
            return
        keep_len = int(self.edge_values.size(0) * (1. - self.dropout))
        if self.pruning_random:
            # pruning randomly
            keep_idx = torch.tensor(random.sample(range(self.edge_values.size(0)), keep_len)).to('cuda')
        else:
            # pruning edges by pro
            keep_idx = torch.multinomial(self.edge_values, keep_len)         # prune high-degree nodes
        self.pruning_random = True ^ self.pruning_random
        keep_indices = self.edge_indices[:, keep_idx]
        # norm values
        keep_values = self._normalize_adj_m(keep_indices, torch.Size((self.num_users, self.num_items)))
        all_values = torch.cat((keep_values, keep_values))
        # update keep_indices to users/items+self.n_users
        keep_indices[1] += self.num_users
        all_indices = torch.cat((keep_indices, torch.flip(keep_indices, [0])), 1)
        self.masked_adj = torch.sparse.FloatTensor(all_indices, all_values, self.Graph.shape).to('cuda')

    def _normalize_adj_m(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        col_sum = 1e-7 + torch.sparse.sum(adj.t(), -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        c_inv_sqrt = torch.pow(col_sum, -0.5)
        cols_inv_sqrt = c_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return values

    def get_edge_info(self):
        rows = torch.from_numpy(self.interaction_matrix.row)
        cols = torch.from_numpy(self.interaction_matrix.col)
        edges = torch.stack([rows, cols]).type(torch.LongTensor)
        # edge normalized values
        values = self._normalize_adj_m(edges, torch.Size((self.num_users, self.num_items)))
        return edges, values

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.
        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        ego_embeddings = torch.cat([self.user_embeddings, self.item_embeddings], 0)
        return ego_embeddings

    def forward(self):
        ego_embeddings = self.get_ego_embeddings()
        all_embeddings = ego_embeddings
        embeddings_layers = []

        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.forward_adj, all_embeddings)
            _weights = F.cosine_similarity(all_embeddings, ego_embeddings, dim=-1)
            all_embeddings = torch.einsum('a,ab->ab', _weights, all_embeddings)
            embeddings_layers.append(all_embeddings)

        ui_all_embeddings = torch.sum(torch.stack(embeddings_layers, dim=0), dim=0)
        user_all_embeddings, item_all_embeddings = torch.split(ui_all_embeddings, [self.num_users, self.num_items])
        return user_all_embeddings, item_all_embeddings

    def bpr_loss(self, user, pos_item, neg_item):
        self.forward_adj = self.masked_adj
        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        posi_embeddings = item_all_embeddings[pos_item]
        negi_embeddings = item_all_embeddings[neg_item]

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, posi_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, negi_embeddings).sum(dim=1)
        m = torch.nn.LogSigmoid()
        bpr_loss = torch.sum(-m(pos_scores - neg_scores))
        #mf_loss = self.mf_loss(pos_scores, neg_scores)
        u_ego_embeddings = self.user_embeddings[user]
        posi_ego_embeddings = self.item_embeddings[pos_item]
        negi_ego_embeddings = self.item_embeddings[neg_item]
        reg_loss = (1/2)*(u_ego_embeddings.norm(2).pow(2) + 
                         posi_ego_embeddings.norm(2).pow(2)  +
                         negi_ego_embeddings.norm(2).pow(2))/float(len(user))
        
        return bpr_loss, reg_loss

    def getUsersRating(self, users):
        self.forward_adj = self.Graph
        all_users, all_items = self.forward()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getUsersUsers(self, users):
        self.forward_adj = self.Graph
        all_users, _ = self.forward()
        users_emb = all_users[users.long()]

        return self.f(torch.matmul(users_emb, all_users.t()))

    def getItemsItems(self, items):
        self.forward_adj = self.Graph
        _, all_items = self.forward()
        items_emb = all_items[items.long()]
        
        return self.f(torch.matmul(items_emb, all_items.t()))

    def getItemsRating(self, items):
        self.forward_adj = self.Graph
        all_users, all_items = self.forward()
        items_emb = all_items[items.long()]
        users_emb = all_users
        item_rating = self.f(torch.matmul(items_emb, users_emb.t()))
        return item_rating
    

class IMP_GCN(BasicModel):
    def __init__(self, args, dataset):
        super(IMP_GCN, self).__init__()
        self.args = args
        self.dataset : dataloader.BasicDataset = dataset
        self.__init_weight()
    
    def __init_weight(self):
        self.device = 'cuda'
        self.n_fold = 20
        # load parameters info
        self.n_users  = self.dataset.n_users
        self.n_items  = self.dataset.m_items
        self.groups = self.args.groups
        self.emb_dim = self.args.recdim  # int type:the embedding size of lightGCN
        self.reg_weight = 0.0001  # float32 type: the weight decay for l2 normalizaton
        self.n_layers = self.args.layer
        self.batch_size = self.args.bpr_batch
        self.f = nn.Sigmoid()

        self.interaction_matrix = self.dataset.UserItemNet.tocoo()
        # generate intermediate data
        self.norm_adj = self.get_norm_adj_mat(self.interaction_matrix)

        # init parameters
        initializer = nn.init.xavier_uniform_
        self.embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.n_users, self.emb_dim))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.n_items, self.emb_dim)))
        })
        self.W_gc_1 = nn.Parameter(initializer(torch.empty(self.emb_dim, self.emb_dim)))
        self.b_gc_1 = nn.Parameter(initializer(torch.empty(1, self.emb_dim)))
        self.W_gc_2 = nn.Parameter(initializer(torch.empty(self.emb_dim, self.emb_dim)))
        self.b_gc_2 = nn.Parameter(initializer(torch.empty(1, self.emb_dim)))
        self.W_gc = nn.Parameter(initializer(torch.empty(self.emb_dim, self.groups)))
        self.b_gc = nn.Parameter(initializer(torch.empty(1, self.groups)))

        self.A_fold_hat = self._split_A_hat(self.norm_adj)
        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

    def pre_epoch_processing(self):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

    def get_norm_adj_mat(self, interaction_matrix):
        adj_mat = sp.dok_matrix((self.n_users + self.n_items,
                           self.n_users + self.n_items), dtype=np.float32)
        inter_M = interaction_matrix
        inter_M_t = interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col+self.n_users),
                             [1]*inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row+self.n_users, inter_M_t.col),
                                  [1]*inter_M_t.nnz)))
        adj_mat._update(data_dict)
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()

        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj = d_mat_inv.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat_inv)
        #print('generate pre adjacency matrix.')
        pre_adj_mat = norm_adj.tocsr()
        return pre_adj_mat

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col])#.transpose()
        indices = torch.from_numpy(indices).type(torch.LongTensor)
        data = torch.from_numpy(coo.data)
        return torch.sparse.FloatTensor(indices, data, torch.Size((coo.shape[0], coo.shape[1])))

    def _split_A_hat(self, X):
        A_fold_hat = []
        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]).to(self.device))
        return A_fold_hat

    def sparse_dense_mul(self, s, d):
        i = s._indices()
        v = s._values()
        dv = d[i[0, :], i[1, :]]  # get values from relevant entries of dense matrix
        ret_tensor = torch.sparse.FloatTensor(i, v * dv, s.size())
        return ret_tensor.to(self.device)

    def _split_A_hat_group(self, X, group_embedding):
        group_embedding = group_embedding.T
        A_fold_hat_group = []
        A_fold_hat_group_filter = []
        A_fold_hat = self.A_fold_hat

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for k in range(0, self.groups):
            A_fold_item_filter = []
            A_fold_hat_item = []

            # n folds in per group (filter user)
            for i_fold in range(self.n_fold):
                start = i_fold * fold_len
                if i_fold == self.n_fold - 1:
                    end = self.n_users + self.n_items
                else:
                    end = (i_fold + 1) * fold_len

                temp_g = self.sparse_dense_mul(A_fold_hat[i_fold], group_embedding[k].expand(A_fold_hat[i_fold].shape))
                temp_slice = self.sparse_dense_mul(temp_g, torch.unsqueeze(group_embedding[k][start:end], dim=1).expand(temp_g.shape))
                #A_fold_hat_item.append(A_fold_hat[i_fold].__mul__(group_embedding[k]).__mul__(torch.unsqueeze(group_embedding[k][start:end], dim=1)))
                A_fold_hat_item.append(temp_slice)
                item_filter = torch.sparse.sum(A_fold_hat_item[i_fold], dim=1).to_dense()
                item_filter = torch.where(item_filter > 0., torch.ones_like(item_filter), torch.zeros_like(item_filter))
                A_fold_item_filter.append(item_filter)

            A_fold_item = torch.concat(A_fold_item_filter, dim=0)
            A_fold_hat_group_filter.append(A_fold_item)
            A_fold_hat_group.append(A_fold_hat_item)

        return A_fold_hat_group, A_fold_hat_group_filter

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.
        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        return ego_embeddings

    def forward(self):
        # _create_imp_gcn_embed in original IMP_GCN/IMP_GCN.py
        A_fold_hat = self.A_fold_hat
        ego_embeddings = self.get_ego_embeddings()
        # group users
        temp_embed = []
        for f in range(self.n_fold):
            temp_embed.append(torch.sparse.mm(A_fold_hat[f], ego_embeddings))
        user_group_embeddings_side = torch.concat(temp_embed, dim=0) + ego_embeddings

        user_group_embeddings_hidden_1 = F.leaky_relu(torch.matmul(user_group_embeddings_side, self.W_gc_1) + self.b_gc_1)
        user_group_embeddings_hidden_d1 = F.dropout(user_group_embeddings_hidden_1, 0.6)

        user_group_embeddings_sum = torch.matmul(user_group_embeddings_hidden_d1, self.W_gc) + self.b_gc
        # user 0-1
        a_top, a_top_idx = torch.topk(user_group_embeddings_sum, 1, sorted=False)
        user_group_embeddings = torch.eq(user_group_embeddings_sum, a_top).type(torch.float32)
        u_group_embeddings, i_group_embeddings = torch.split(user_group_embeddings, [self.n_users, self.n_items], 0)
        i_group_embeddings = torch.ones_like(i_group_embeddings)
        user_group_embeddings = torch.concat([u_group_embeddings, i_group_embeddings], dim=0)
        # Matrix mask
        A_fold_hat_group, A_fold_hat_group_filter = self._split_A_hat_group(self.norm_adj, user_group_embeddings)
        # embedding transformation
        all_embeddings = [ego_embeddings]
        temp_embed = []
        for f in range(self.n_fold):
            temp_embed.append(torch.sparse.mm(A_fold_hat[f], ego_embeddings))

        side_embeddings = torch.concat(temp_embed, dim=0)
        all_embeddings += [side_embeddings]

        ego_embeddings_g = []
        for g in range(0, self.groups):
            ego_embeddings_g.append(ego_embeddings)
        ego_embeddings_f = []
        for k in range(1, self.n_layers):
            for g in range(0, self.groups):
                temp_embed = []
                for f in range(self.n_fold):
                    temp_embed.append(torch.sparse.mm(A_fold_hat_group[g][f], ego_embeddings_g[g]))
                side_embeddings = torch.concat(temp_embed, dim=0)
                ego_embeddings_g[g] = ego_embeddings_g[g] + side_embeddings
                temp_embed = []
                for f in range(self.n_fold):
                    temp_embed.append(torch.sparse.mm(A_fold_hat[f], side_embeddings))
                if k == 1:
                    ego_embeddings_f.append(torch.concat(temp_embed, dim=0))
                else:
                    ego_embeddings_f[g] = torch.concat(temp_embed, dim=0)
            ego_embeddings = torch.sum(torch.stack(ego_embeddings_f, dim=0), dim=0)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, 1)
        all_embeddings = torch.sum(all_embeddings, dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings

    def bpr_loss(self, user, pos_item, neg_item):
        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user, :]
        pos_embeddings = item_all_embeddings[pos_item, :]
        neg_embeddings = item_all_embeddings[neg_item, :]
        u_embeddings_pre = self.embedding_dict['user_emb'][user, :]
        pos_embeddings_pre = self.embedding_dict['item_emb'][pos_item, :]
        neg_embeddings_pre = self.embedding_dict['item_emb'][neg_item, :]

        pos_scores = torch.sum(torch.mul(u_embeddings, pos_embeddings), dim=1)
        neg_scores = torch.sum(torch.mul(u_embeddings, neg_embeddings), dim=1)

        regularizer = 1./2*(u_embeddings_pre**2).sum() + 1./2*(pos_embeddings_pre**2).sum() + 1./2*(neg_embeddings_pre**2).sum()
        regularizer = regularizer / self.batch_size

        maxi = F.softplus(-(pos_scores - neg_scores))
        mf_loss = torch.mean(maxi)

        emb_loss = self.reg_weight * regularizer
        #reg_loss = 0.0
        return mf_loss, emb_loss


    def getUsersRating(self, users):

        all_users, all_items = self.forward()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
    
    def getUsersUsers(self, users):

        all_users, _ = self.forward()
        users_emb = all_users[users.long()]

        return self.f(torch.matmul(users_emb, all_users.t()))
    
    def getItemsItems(self, items):

        _, all_items = self.forward()
        items_emb = all_items[items.long()]
        
        return self.f(torch.matmul(items_emb, all_items.t()))
    
    def getItemsRating(self, items):

        all_users, all_items = self.forward()
        items_emb = all_items[items.long()]
        users_emb = all_users
        item_rating = self.f(torch.matmul(items_emb, users_emb.t()))
        return item_rating



class EASE(BasicModel):
    r"""EASE is a linear model for collaborative filtering, which combines the
    strengths of auto-encoders and neighborhood-based approaches.
    """

    def __init__(self, args, dataset):
        super(EASE, self).__init__()
        self.args = args
        self.dataset : dataloader.BasicDataset = dataset

        reg_weight = 0.0001

        # need at least one param
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

        X = self.dataset.UserItemNet
        # just directly calculate the entire score matrix in init
        # (can't be done incrementally)

        # gram matrix
        G = X.T @ X

        # add reg to diagonal
        G += reg_weight * sp.identity(G.shape[0]).astype(np.float32)

        # convert to dense because inverse will be dense
        G = G.todense()

        # invert. this takes most of the time
        P = np.linalg.inv(G)
        B = P / (-np.diag(P))
        # zero out diag
        np.fill_diagonal(B, 0.0)

        # instead of computing and storing the entire score matrix,
        # just store B and compute the scores on demand
        # more memory efficient for a larger number of users
        # but if there's a large number of items not much one can do:
        # still have to compute B all at once
        # S = X @ B
        # self.score_matrix = torch.from_numpy(S).to(self.device)

        # torch doesn't support sparse tensor slicing,
        # so will do everything with np/scipy
        self.item_similarity = B
        self.interaction_matrix = X
        #self.other_parameter_name = ["interaction_matrix", "item_similarity"]
        self.device = 'cuda'

    def forward(self):
        pass

    def calculate_loss(self, interaction):
        return torch.nn.Parameter(torch.zeros(1))

    def getUsersRating(self, users):
        users = users.cpu()
        r = self.interaction_matrix[users, :] @ self.item_similarity
        return torch.from_numpy(r.flatten())
    


class NGCF(BasicModel):
    r"""NGCF is a model that incorporate GNN for recommendation.
    We implement the model following the original author with a pairwise training mode.
    """
    def __init__(self, args, dataset):
        super(NGCF, self).__init__()
        self.args = args
        self.dataset : dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        # load dataset info
        self.interaction_matrix = self.dataset.UserItemNet.tocoo()
        self.n_users  = self.dataset.n_users
        self.n_items  = self.dataset.m_items

        # load parameters info
        self.embedding_size = self.args.recdim 
        #self.hidden_size_list = config['hidden_size_list']
        #self.hidden_size_list = [self.embedding_size] + self.hidden_size_list
        self.n_layers = self.args.layer 
        self.hidden_size_list = [self.embedding_size] * self.n_layers
        self.node_dropout = 0.0
        self.message_dropout = 0.1
        self.reg_weight = 0.0001
        self.device = 'cuda'
        self.f = nn.Sigmoid()
        # define layers and loss
        self.sparse_dropout = SparseDropout(self.node_dropout)
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.GNNlayers = torch.nn.ModuleList()
        for idx, (input_size, output_size) in enumerate(zip(self.hidden_size_list[:-1], self.hidden_size_list[1:])):
            self.GNNlayers.append(BiGNNLayer(input_size, output_size))

        # generate intermediate data
        self.Graph = self.dataset.getSparseGraph(self.dataset.UserItemNet)
        self.eye_matrix = self.get_eye_mat().to(self.device)

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def get_eye_mat(self):
        r"""Construct the identity matrix with the size of  n_items+n_users.
        Returns:
            Sparse tensor of the identity matrix. Shape of (n_items+n_users, n_items+n_users)
        """
        num = self.n_items + self.n_users  # number of column of the square matrix
        i = torch.LongTensor([range(0, num), range(0, num)])
        val = torch.FloatTensor([1] * num)  # identity matrix
        return torch.sparse.FloatTensor(i, val)

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.
        Returns:
            Tensor of the embedding matrix. Shape of (n_items+n_users, embedding_dim)
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self):
        A_hat = self.sparse_dropout(self.Graph) if self.node_dropout != 0 else self.Graph
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]
        for gnn in self.GNNlayers:
            all_embeddings = gnn(A_hat, self.eye_matrix, all_embeddings)
            all_embeddings = nn.LeakyReLU(negative_slope=0.2)(all_embeddings)
            all_embeddings = nn.Dropout(self.message_dropout)(all_embeddings)
            all_embeddings = F.normalize(all_embeddings, p=2, dim=1)
            embeddings_list += [all_embeddings]  # storage output embedding of each layer
        ngcf_all_embeddings = torch.cat(embeddings_list, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(ngcf_all_embeddings, [self.n_users, self.n_items])

        return user_all_embeddings, item_all_embeddings

    def bpr_loss(self, user, pos_item, neg_item):

        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        bpr_loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        reg_loss = (1/2)*(u_embeddings.norm(2).pow(2) + 
                         pos_embeddings.norm(2).pow(2)  +
                         neg_embeddings.norm(2).pow(2))/float(len(user))
        
        return bpr_loss, reg_loss

    def getUsersRating(self, users):
        all_users, all_items = self.forward()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
    
    def getUsersUsers(self, users):
        all_users, _ = self.forward()
        users_emb = all_users[users.long()]

        return self.f(torch.matmul(users_emb, all_users.t()))

    def getItemsItems(self, items):
        _, all_items = self.forward()
        items_emb = all_items[items.long()]
        
        return self.f(torch.matmul(items_emb, all_items.t()))

    def getItemsRating(self, items):
        all_users, all_items = self.forward()
        items_emb = all_items[items.long()]
        users_emb = all_users
        item_rating = self.f(torch.matmul(items_emb, users_emb.t()))
        return item_rating
    
class CDE_CF(BasicModel):
    def __init__(self, 
                 args,
                 dataset:BasicDataset):
        super(CDE_CF, self).__init__()
        self.args = args
        self.dataset : dataloader.BasicDataset = dataset
        self.__init_weight()
        
    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.args.recdim #self.config['latent_dim_rec']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph(self.dataset.UserItemNet)
        self.odeblock = ode.ODEblock(ode.ODEFunc(self.Graph, self.latent_dim, self.args.data_name), t= torch.tensor([0, self.args.t]))
    
    def computer(self):   
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        all_emb_1 = self.odeblock(all_emb)
        users, items = torch.split(all_emb_1, [self.num_users, self.num_items])
        return users, items
    
    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getUsersUsers(self, users):
        all_users, _ = self.computer()
        users_emb = all_users[users.long()]

        return self.f(torch.matmul(users_emb, all_users.t()))

    def getItemsItems(self, items):
        _, all_items = self.computer()
        items_emb = all_items[items.long()]
        
        return self.f(torch.matmul(items_emb, all_items.t()))

    def getItemsRating(self, items):
        all_users, all_items = self.computer()
        items_emb = all_items[items.long()]
        users_emb = all_users
        item_rating = self.f(torch.matmul(items_emb, users_emb.t()))
        return item_rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
       
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss
       
    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma
    
    
class ODE_CF(BasicModel):
    def __init__(self, 
                args,
                dataset:BasicDataset):
        super(ODE_CF, self).__init__()
        self.args = args
        self.dataset : dataloader.BasicDataset = dataset
        self.__init_weight()
        
    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.args.recdim #self.config['latent_dim_rec']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph(self.dataset.UserItemNet)
        self.odeblock = ode1.ODEblock(ode1.ODEFunc(self.Graph, self.latent_dim),t= torch.tensor([0, self.args.t]))

    def computer(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        all_emb_1 = self.odeblock(all_emb)
        users, items = torch.split(all_emb_1, [self.num_users, self.num_items])
        return users, items
    
    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getUsersUsers(self, users):
        all_users, _ = self.computer()
        users_emb = all_users[users.long()]

        return self.f(torch.matmul(users_emb, all_users.t()))

    def getItemsItems(self, items):
        _, all_items = self.computer()
        items_emb = all_items[items.long()]
        
        return self.f(torch.matmul(items_emb, all_items.t()))

    def getItemsRating(self, items):
        all_users, all_items = self.computer()
        items_emb = all_items[items.long()]
        users_emb = all_users
        item_rating = self.f(torch.matmul(items_emb, users_emb.t()))
        return item_rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                        posEmb0.norm(2).pow(2)  +
                        negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
    
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss
    
    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        #all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma