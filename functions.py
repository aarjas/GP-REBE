import torch
from scipy.spatial.distance import pdist, cdist, squareform
from math import sqrt, log
import matplotlib.pyplot as plt
import numpy as np

def likelihoodREML(lv,G,y,X,x,xgrid,C_interp,Cinv):
    n = y.shape[0]
    ngrid = xgrid.shape[0]
    
    legrid = lv[torch.arange(ngrid)]
    lggrid = lv[torch.arange(ngrid,2*ngrid)]
    

    
    le = C_interp @ (Cinv @ legrid)
    lg = C_interp @ (Cinv @ lggrid)
    

    ve = torch.exp(le)
    vg = torch.exp(lg)
    V = (torch.reshape(torch.sqrt(vg),(n,1)) @ torch.reshape(torch.sqrt(vg),(n,1)).T) * G + torch.diag(ve)
    L = torch.linalg.cholesky(V)
    t1 = 2*torch.sum(torch.log(L.diag()))
    
    XVX = torch.linalg.solve_triangular(L,X,upper=False)
    XVX = XVX.T @ XVX
    L2 = torch.linalg.cholesky(XVX)
    t2 = 2*torch.sum(torch.log(L2.diag()))
        
    Ly = torch.linalg.solve_triangular(L,torch.reshape(y,(n,1)),upper=False)
    t3 = torch.sum(Ly**2)
    
    Vy = torch.linalg.solve_triangular(L.T,Ly,upper=True)
    XVy = X.T @ Vy
    XVXXVy = torch.linalg.solve_triangular(L2,XVy,upper=False)
    t4 = torch.sum(XVXXVy**2)

    
    val = -0.5*t1 - 0.5*t2 - 0.5*t3 + 0.5*t4 - 0.5*(legrid.T @ Cinv @ legrid)- 0.5*(lggrid.T @ Cinv @ lggrid)
    return val

def grad_likelihoodREML(lv,G,y,X,x,xgrid,C_interp,Cinv):
    n = y.shape[0]
    ngrid = xgrid.shape[0]    
    legrid = lv[torch.arange(ngrid)]
    lggrid = lv[torch.arange(ngrid,2*ngrid)]    
    le = C_interp @ (Cinv @ legrid)
    lg = C_interp @ (Cinv @ lggrid)
    ve = torch.exp(le)
    vg = torch.exp(lg)
    VGi = (torch.reshape(torch.sqrt(vg),(n,1)) @ torch.reshape(torch.sqrt(vg),(n,1)).T)
    V = VGi * G + torch.diag(ve)
    Vinv = torch.linalg.inv(V)
    VinvX = Vinv @ X
    P = Vinv - VinvX @ torch.linalg.solve(X.T @ VinvX, VinvX.T)
    tr1 = ve * P.diag()
    
    Py = P @ y    
    gr1 = tr1 - ve * Py**2
    gr1 = Cinv @ (C_interp.T @ gr1) +  2 * (Cinv @ legrid)
    
    temp = 0.5 * VGi * G * P
    temp.fill_diagonal_(0)
    temp += torch.diag(torch.sum(temp,0) + vg * torch.diag(G) * torch.diag(P))
    tr2 = torch.sum(temp,0)
    
    
    
    Pymat = torch.reshape(Py,(n,1)) @ torch.reshape(Py,(n,1)).T
    temp2 = 0.5 * VGi * Pymat * G
    temp2.fill_diagonal_(0)
    temp2 += torch.diag(torch.sum(temp2,0) + vg * torch.diag(G) * Py**2)
    gr2 = torch.sum(temp2, 0)
    gr2 = tr2 - gr2
    gr2 = Cinv @ (C_interp.T @ gr2) + 2 * (Cinv @ lggrid)

    gr = -0.5*torch.cat((gr1,gr2))
    return gr

def BAI_reml(lv,G,y,X,x,xgrid,C_interp,Cinv):
    n = y.shape[0]
    ngrid = xgrid.shape[0]    
    legrid = lv[torch.arange(ngrid)]
    lggrid = lv[torch.arange(ngrid,2*ngrid)]    
    le = C_interp @ (Cinv @ legrid)
    lg = C_interp @ (Cinv @ lggrid)
    ve = torch.exp(le)
    vg = torch.exp(lg)
    VGi = (torch.reshape(torch.sqrt(vg),(n,1)) @ torch.reshape(torch.sqrt(vg),(n,1)).T)
    V = VGi * G + torch.diag(ve)
    Vinv = torch.linalg.inv(V)
    VinvX = Vinv @ X
    P = Vinv - VinvX @ torch.linalg.solve(X.T @ VinvX, VinvX.T)
    v = P @ y

    m1 = torch.diag(ve * v)
    m1 = (m1 @ C_interp) @ Cinv
    m2 = 0.5 * VGi * G * v
    m2.fill_diagonal_(0)
    m2 += torch.diag(torch.sum(m2,1) + vg * v * torch.diag(G))
    m2 = (m2 @ C_interp) @ Cinv
    m = torch.cat((m1,m2),1)
    H = m.T @ P @ m

    
    zer = torch.zeros((ngrid,ngrid))
    Hc = torch.cat((torch.cat((Cinv,zer),1),torch.cat((zer,Cinv),1)),0)
    
    Ht = 0.5*H + Hc
    return(Ht)

    


    

def mcmc(G,y,X,x,xgrid,Nsim,step,precond,mh,ngrid,C_interp,Cinv):
    lk = torch.zeros((Nsim,2*ngrid),dtype=torch.double)
    tau = step #Step-size
    itermean = torch.exp(lk[0,:].clone())
    C = torch.linalg.inv(precond)
    R = torch.linalg.cholesky(C)
    
    likek = lambda lk: likelihoodREML(lk,G,y,X,x,xgrid,C_interp,Cinv)
    lki = lk[0,:].requires_grad_()
    likeold = likek(lki)
    likeold.backward()    
    grold = lki.grad
    likelihood = [likeold.detach().cpu()]    
    
    las = 0
    
    for i in range(1,Nsim):
        t1 = C @ grold
        t2 = R @ torch.randn((2*ngrid),dtype=torch.double)
        lkp = lk[i-1,:] + tau*t1 + sqrt(2*tau)*t2
        
        likek = lambda lk: likelihoodREML(lk,G,y,X,x,xgrid,C_interp,Cinv)
        lki = lkp.requires_grad_()
        like = likek(lki)
        like.backward()    
        gr = lki.grad
        #print(like)
        
        if mh:        
            qpnorm = precond @ (sqrt(2*tau)*t2)
            qpnorm = torch.sum((sqrt(2*tau)*t2)*qpnorm)
            
            v1 = lk[i-1,:] - lkp - tau * (C @ gr)
            qonorm = torch.sum(v1 * (precond @ v1))
            
            qr = -0.25/tau * qonorm + 0.25/tau * qpnorm
            
            accept_prob = like - likeold + qr
            
            
            if log(torch.rand(1)) < accept_prob:
                lk[i,:] = lkp.detach().clone()
                likeold = like.clone()
                grold = gr.clone()
                las += 1
            else:
                lk[i,:] = lk[i-1,:].clone()
                
            
            
            if i % 50 == 0:
                delta = min(0.01, 1/sqrt(i))
                if las/50 < 0.574:
                    tau -= delta
                else:
                    tau += delta
                las = 0
        else:
            lk[i,:] = lkp.detach().clone()
            grold = gr.clone()
            likeold = like.clone()
            
        likelihood.append(likeold.detach().cpu())
                
        itermean = 1/(i+1)*torch.exp(lk[i,:]) + i/(i+1)*itermean
        
        
        if i % 100 == 0:
            #plt.figure(figsize=(18, 6), dpi=300)
            plt.subplot(1,2,1)
            plt.plot(lk[0:i,70].detach().cpu())
            plt.subplot(1,2,2)
            plt.plot(xgrid.cpu(),itermean[ngrid:2*ngrid].cpu())
            plt.pause(0.001)
            
    return lk   


def opt(G,y,X,x,xgrid,step,ngrid,C_interp,Cinv):
    lk = torch.zeros((2*ngrid),dtype=torch.double)
    tau = step #Step-size
    sorted, indices = torch.sort(xgrid)
    las = 0
    lkold = lk.clone()
    
    while True:
        likek = lambda lk: likelihoodREML(lk,G,y,X,x,xgrid,C_interp,Cinv)
        #likek = lambda lk: likelihood(lk,HG,y,x,xgrid)
        lki = lk.clone().requires_grad_()
        like = likek(lki)
        like.backward()    
        gr = lki.grad
        H = BAI_reml(lk,G,y,X,x,xgrid,C_interp,Cinv)
        lk = lk + tau*torch.linalg.solve(H,gr)
        #gr = grad_likelihoodREML(lkold,HG,y,X,x,xgrid)
        #t1 = C @ gr[torch.arange(ngrid)]
        #t2 = C @ gr[torch.arange(ngrid,2*ngrid)]
        #lk = lk + tau*torch.cat((t1,t2))
        
        if torch.sum((lk - lkold)**2)/torch.sum(lkold**2) < 1e-6:
            break
        
        lkold = lk.clone()
        
        las += 1
        
        if las % 1 == 0:
            plt.figure(figsize=(18, 6))
            plt.subplot(1,2,1)
            plt.plot(xgrid.cpu(),torch.exp(lk[0:ngrid]).cpu())
            plt.subplot(1,2,2)
            plt.plot(xgrid.cpu(),torch.exp(lk[ngrid:2*ngrid]).cpu())
            plt.pause(0.001)
          
    H = BAI_reml(lk, G, y, X, x, xgrid, C_interp, Cinv)
    return lk, H

def estimate_variances(G,y,X,x,xgrid,step_opt,step_MCMC,ngrid,C_interp,Cinv,Nsim):
    '''    

    Parameters
    ----------
    G : Relationship matrix.
    y : Measuremed phenotype data.
    X : Fixed effects matrix.
    x : Measured covariate values of individuals.
    xgrid : Computational grid.
    step_opt : Initial optimization step length.
    step_MCMC : MCMC step length.
    ngrid : Size of computational grid.
    C_interp : Covariance matrix between computational grid and measure covariate values.
    Cinv : Inverse covariance matrix of computational grid.
    Nsim : Number of MCMC rounds.

    Returns
    -------
    log_var_chains : MCMC chains of log-transformed variance components.

    '''
    lkopt, BAI = opt(G,y,X,x,xgrid,step_opt,ngrid,C_interp,Cinv)
    log_var_chains = mcmc(G,y,X,x,xgrid,Nsim,step_MCMC,BAI,True,ngrid,C_interp,Cinv)

    return log_var_chains