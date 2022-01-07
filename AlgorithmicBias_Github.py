# these are the special functions for matrix factorization
import MF_sim

import os,ast,time,random
import pickle as pk 
import pandas as pd
import numpy as np
import matplotlib
import tensorflow as tf
from glob import glob
from typing import Dict
import copy
from collections import Counter
from scipy.stats import spearmanr
from scipy import sparse
 
# find most popular item
def most_frequent(R_model,top_picks_GT=[]):
    rank_GT = []
    if len(top_picks_GT) > 0:
        # mean model rank of each user's GT top pick
        rank_GT = [(len(line)-1) - np.argsort(line)[tp_GT] for tp_GT,line in zip(top_picks_GT,R_model)]
    # find the most-liked item for each user
    top_picks = np.argmax(R_model,axis=1)
    # return the model's top picks and where the model ranks the ground truth most-liked item 
    return top_picks,rank_GT
 
def model_bias(R_bias,R_avail,top_picks_GT,U,V,beta,r,GT=False,rand_rec=False,epsilon=0.0,real_data=False):
    # making recommendations based on items not yet chosen
    # pick items as a function of intrinsic preferences (with probability 1-beta) and other factors (beta)
    # if GT then we are in the idealized scenario where we know the probabilites each user picks each item
    R_model=R_bias
    if not GT:
        # student model
        R_model = U.numpy() @ V.numpy()
    # top_picks: the model's top picks 
    # rank_GT: where the model ranks the ground truth most-liked item 
    top_picks,rank_GT = most_frequent(R_model,top_picks_GT)
    
    # recommend K items to each user
    rec_indices = np.argsort(R_model, axis=1)
    topr_indices = []
    for i,row in enumerate(rec_indices):
        # find top (last) indices that were not previously chosen
        if r > 0:
            # always show new content
            if rand_rec or np.random.uniform() < epsilon:
                new_indices = row[R_avail[i,row]<0]
                # recommend content completely at random
                random.shuffle(new_indices)
                new_indices = new_indices[-r:]
            else:
                new_indices = row[R_avail[i,row]<0][-r:]
            # if new content, update 
            new = new_indices[R_avail[i,new_indices]<0]
            R_row = R_bias[i] #+ beta*(np.ones(R[i].shape)-R[i])
            if real_data: # synthetic data are ratings, e.g., between 0-1 after normalization
                chosen = R_model[i,new]
            else:
                chosen = np.random.binomial(1,R_row)[new]
            R_avail[i,new] = chosen
    
    return R_avail,[top_picks,rank_GT]


def sim_data(R,fract_available,beta,r,embeddings,m,seed,GT=False,rand_rec=False,uniform_beta = False,epsilon = 0.0,max_time = -1, real_data = False):
    # probability to choose item
    R_bias = np.ones(R.shape)*beta + (1-beta)*R
    # if uniform_beta, choose item due to internal preferences (1-beta) plus other factors (beta)
    # this is strongly different from factorizable matrix, an assumption of MF models
    if uniform_beta:
        b = np.random.uniform(low=0,high=1,size=R.shape)
        print(b.shape)
        R_bias = np.multiply(np.ones(R.shape),b) + np.multiply((np.ones(R.shape)-b),R)
    # top_picks: the model's top picks 
    # rank_GT: where the model ranks the ground truth most-liked item 
    top_picks_GT,rank_GT = most_frequent(R_bias)
    # controlling randomness of simulation
    if seed is not None:
        rng = np.random.seed(seed)
    if real_data:
        # no noise
        R_realization = R
    else:
        # items are chosen with a probability
        R_realization =  np.random.binomial(1,R_bias)
    # create sparsified matrix of initial items chosen
    R_avail = MF_sim.create_Ravail(m,R_realization,fract_available,seed)
    
    U = None
    V = None
    simtime = 0
    loss_over_time = []
    # simulation system
    # while items are not recommended
    while -1 in set(list(R_avail.flatten())):
        # record time
        simtime += 1
        # prepare system to fit student model to data
        R_avail,R_train,R_val = MF_sim.prepare_system(m,R_avail,seed)
        model_err = 0
        min_val = -1
        if not GT:
            # fit student model to data
            U,V,min_val,collect_vals = MF_sim.fit_data(R_train,R_val, embeddings,U=U,V=V,silence=True)
            # find mean squared error between GT probabilities and model estimate
            model_err = np.mean(np.abs(R_bias - (U.numpy() @ V.numpy())).flatten()**2)        
            # spearman correlations
            R_model = U.numpy() @ V.numpy() 
        else:
            R_model = R_bias.copy()
        # legacy:
        # correlation between model and GT data
        s_all,p_all = spearmanr(R_model.flatten(),R_bias.flatten())
        # correlation between model and GT item populatity
        s_item,p_item = spearmanr(np.mean(R_model,axis=0).flatten(),np.mean(R_bias,axis=0).flatten())
        # correlation between model and GT user activity (items chosen)
        s_user,p_user = spearmanr(np.mean(R_model,axis=1).flatten(),np.mean(R_bias,axis=1).flatten())
        # find Brier score (mean squared error) between real data and model estimate
        data_err = np.mean((R_model.flatten()[R_avail.flatten()>=0] - R_avail.flatten()[R_avail.flatten()>=0])**2)
        # find rank, popularity of items
        R_avail,[top_picks,rank_GT] = model_bias(R_bias,R_avail,top_picks_GT,U,V,beta,r,GT,rand_rec,epsilon,real_data)
        # how many times item chosen
        popularity = [np.sum(c[0][c[0]!=-1]) for c in zip(R_avail.T)]
        # save data
        loss_over_time.append([simtime,min_val,len(R_avail.flatten()[R_avail.flatten()==-1]),popularity,[model_err,data_err],[top_picks,rank_GT],[[s_all,p_all],[s_item,p_item],[s_user,p_user]]])
        # stop if model takes too long (e.g., stuck in some infinite loop)           
        if simtime > 10000 or (max_time > 0 & simtime >= max_time):
            break
    loss_over_time = np.array(loss_over_time)
    return R_avail,U,V,loss_over_time,simtime 

# impute missing data
def find_GT(df):
    U = None
    V = None
    embedding_dim = 30
    # shuffle rows
    df = df.sample(frac=1)
    # look at 5000 users for simplicity
    df = df.loc[df['userId'].values < 5000,]
    # dimension of dense matrix
    dim = (np.max(df['userId'].values.astype(int)),np.max(df['movieId'].values.astype(int)))
    # non-zero indices
    indx = df[['userId','movieId']].values.astype(int)-1
    #values in non-zero elements
    vals = (df['rating'].values.astype(np.float64)-0.5)/4.5
    # create training/testing split for imputation
    train_pos = int(len(df)*0.8)
    R_train = tf.SparseTensor(indices=indx[:train_pos], values=vals[:train_pos], dense_shape=dim)
    R_val = tf.SparseTensor(indices=indx[train_pos:], values=vals[train_pos:], dense_shape=dim)
    U,V,min_val,collect_vals = MF_sim.fit_data(R_train,R_val, embedding_dim,U=U,V=V,silence=True)    
    return [df,U,V,min_val,collect_vals]

def collect_AB_sims(n,m,k,outfile,GT,rand_rec,seed=None,ID=None,uniform_beta=False,epsilon=0.0):
    if ID is None:
        P = np.random.uniform(0,1/np.sqrt(k),size=(n,k))
        Q = np.random.uniform(0,1/np.sqrt(k),size=(m,k))
    # continuous values
    else:
        in_file = 'AlgorithmicBiasSimulation_n=4000_m=200_k=4_'+str(ID)+'_beta=0.0_r=1.pkl'
        in_data = pk.load(open(in_file,'rb'))
        P = in_data['P'][0]
        Q = in_data['Q'][0]
        
    R = P @ Q.T
    new_sim = True
    # ordering beta in order of interest
    for beta in np.arange(0.0,1.0,0.1):
       beta = round(beta,2)
       # ignore beta values if we simulate beta values uniformly distributed between 0 and 1
       if uniform_beta and not new_sim:
           continue
       new_sim = False
       # number of items recommended for each user each timestep
       for r in [1]:
            # what we save for each simulation
            all_loss = {'P':[],'Q':[],'n':[],'m':[],'k':[],'beta':[],'r':[],'fract_available':[],'epsilon':[],'embeddings':[],'realization':[],'sim_data':[],'final_R_views':[],'final_U':[],'final_V':[],'gt_U':[],'gt_V':[]}
            # P, Q are factorized matrices of user-item matrix
            all_loss['P'].append(P)
            all_loss['Q'].append(Q)
            # number of agents
            all_loss['n'].append(n)
            # number of items
            all_loss['m'].append(m)
            # number of latent features in teacher model/ground truth data (rank of latent matrices)
            all_loss['k'].append(k)
            # fraction of data we start with before we make recommendations
            for fract_available in [0.001]:
                # embedding dimension for student model
                for embeddings in [2,5]:
                        # result of simulation
                        R_new,U,V,data_over_time,simtime = sim_data(R,fract_available,beta,r,embeddings,m,seed,GT,rand_rec,uniform_beta,epsilon)
                        # value of beta, ignore if uniformly distributed beta
                        all_loss['beta'].append(beta)
                        # number of recommendations at each timestep
                        all_loss['r'].append(r)
                        # fraction of data we start	with before we make recommendations
                        all_loss['fract_available'].append(fract_available)
                        # probability we make random recommendation
                        all_loss['epsilon'].append(epsilon)
                        # dimension of student model embedding
                        all_loss['embeddings'].append(embeddings)
                        # number of realizations (this is legacy)
                        all_loss['realization'].append(1)
                        # time-varying data from simulation
                        all_loss['sim_data'].append(data_over_time)
                        # who picked what items at the end of the simulation
                        all_loss['final_R_views'].append(R_new)
                        # student model at the end of the simulation
                        if U is not None and V is not None:
                            all_loss['final_U'].append(U.numpy())
                            all_loss['final_V'].append(V.numpy())
            # output all data
            file = outfile + '_beta='+str(beta)+'_r='+str(r)+'.pkl'
            if GT:
                file = outfile + '_beta='+str(beta)+'_r='+str(r)+'_GT.pkl'
            if rand_rec:
                file = outfile + '_beta='+str(beta)+'_r='+str(r)+'_rand_rec.pkl'
            pk.dump(all_loss,open(file,'wb'))

# this code is for creating simulations from real data
def semi_synth_sims(infile,outfile,GT,rand_rec,seed=None,epsilon=0.0):
    # fill in missing recommendation data from GT with values imputed from matrix factorization 
    gt_file = '/'.join(infile.split('/')[:-1])+ 'filled_GT.npy'
    if not os.path.exists(gt_file):
        df = pd.read_csv(infile)
        # factorize matrix from real data
        df,U,V,min_val,collect_vals = find_GT(df)
        gt_data = U.numpy() @ V.numpy()
        indx = df[['userId','movieId']].values.astype(int)-1
        vals = (df['rating'].values.astype(np.float64)-0.5)/4.5
        for [x,y],value in zip(indx,vals):
            gt_data[x,y] = value
        np.save(gt_file,gt_data)
    gt_data = np.load(gt_file)
    # number of users (n) and items (m)
    n = gt_data.shape[0]
    m = gt_data.shape[1]
    # Semi-synthetic GT data
    R = gt_data
    new_sim = True
    # number of recommendations for each user per timestep
    for r in [1]:
            # what we save for each simulation
            all_loss = {'P':[],'Q':[],'n':[],'m':[],'beta':[],'r':[],'fract_available':[],'epsilon':[],'embeddings':[],'realization':[],'sim_data':[],'final_R_views':[],'final_U':[],'final_V':[],'gt_U':[],'gt_V':[]}
            # number of users (n) and items (m)
            all_loss['n'].append(n)
            all_loss['m'].append(m)
            # legacy parameters
            beta = 0.0
            uniform_beta = False
            # fraction of semi-synthetic data we begin with
            for fract_available in [0.001]:
                # number of latent features in student model
                for embeddings in [30]:
                        # outcome of simulation
                        R_new,U,V,loss_over_time,simtime = sim_data(R,fract_available,beta,r,embeddings,m,seed,GT,rand_rec,uniform_beta,epsilon,max_time = 30,real_data = True)
                        # descriptions of these outputs discussed in collect_AB_sims
                        all_loss['beta'].append(beta)
                        all_loss['r'].append(r)
                        all_loss['fract_available'].append(fract_available)
                        all_loss['epsilon'].append(epsilon)
                        all_loss['embeddings'].append(embeddings)
                        all_loss['realization'].append(1)#realization)
                        all_loss['sim_data'].append(loss_over_time)
                        all_loss['final_R_views'].append(R_new)
                        if U is not None and V is not None:
                            all_loss['final_U'].append(U.numpy())
                            all_loss['final_V'].append(V.numpy())
            # save data
            file = outfile + '_r='+str(r)+'.pkl'
            if GT:
                file = outfile + '_r='+str(r)+'_GT.pkl'
            if rand_rec:
                file = outfile + '_r='+str(r)+'_rand_rec.pkl'
            pk.dump(all_loss,open(file,'wb'))
def main():

    # semi-synthetic simulations
    semi_synth = True:
    if semi_synth:
        data_in_file = 'datasets/ml-25m/ratings.csv'
        for GT in [True]:
            for eps in [0.0]:#[0.0,0.1,1.0]:
                outfile = 'datasets/ml-25m/Sim_epsilon='+str(eps)
                semi_synth_sims(data_in_file,outfile,GT,rand_rec=False,epsilon=eps)
    else: 
        #number of users
        n=4000
        #number of items
        m=200
        # latent dimensions
        k=4
        # ID = GT probabilities
        ID = 23493#None
        # epsilon: probability to make random recommendations
        eps = 1.0#0.1
        #RNG seed: let the initial conditions be the same, but stochasticity still develop. What happens?
        seed = None#3141
        # uniform beta
        uniform_beta = True#False
        for GT in [False]:
            for rand_rec in [False]:#[True]:
                if rand_rec and GT: continue
                for realization in range(35,40):
                    file_id = ID
                    if ID == None:
                        file_id = np.random.randint(0,100000)
                    # simulate data, save to file
                    algorithmic_bias_file = 'AlgorithmicBiasSimulation_seed='+str(seed)+'_n='+str(n)+'_m='+str(m)+'_k='+str(k)+'_'+str(file_id)+'-'+str(realization)+'_uniform_beta='+str(uniform_beta)+'_epsilon='+str(eps)
                    collect_AB_sims(n,m,k,algorithmic_bias_file,GT,rand_rec,seed,ID,uniform_beta,epsilon=eps)

if __name__ == "__main__":
    main()
