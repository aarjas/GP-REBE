import torch
import functions
from scipy.spatial.distance import pdist, cdist, squareform
from math import sqrt
import matplotlib.pyplot as plt

#Create data
#####################################################################################################################
#Create relationship matrix
n = 1000 #Number of individuals
x = torch.rand(n)
D = torch.tensor(squareform(pdist(x.cpu().reshape(n,1))))
G = torch.exp(-torch.abs(D)*200)

#Create GP covariance matrix to simulate paths
ngrid = 50
xgrid = torch.linspace(x.min(),x.max(),ngrid)
D = torch.tensor(squareform(pdist(xgrid.reshape(ngrid,1).cpu())))
l =  D.max()/3  #Length-scale
C = (1 + sqrt(5)*D/l + 5*D**2/(3*l**2))*torch.exp(-sqrt(5)*D/l)
Cinv = torch.linalg.inv(C)
L = torch.linalg.cholesky(C)
D_interp = torch.tensor(cdist(x.reshape(n,1).cpu(),xgrid.reshape(ngrid,1).cpu()))
C_interp = (1 + sqrt(5)*D_interp/l + 5*D_interp**2/(3*l**2))*torch.exp(-sqrt(5)*D_interp/l)

#Simulate variance functions with GP
legrid = L @ torch.randn(ngrid,dtype=torch.double)
lggrid = L @ torch.randn(ngrid,dtype=torch.double)
le = C_interp @ (Cinv @ legrid)
lg = C_interp @ (Cinv @ lggrid)
ve = torch.exp(le)
vg = torch.exp(lg)
ground_truths = [torch.exp(legrid),torch.exp(lggrid)]
plt.plot(torch.cat((ground_truths[0],ground_truths[1])))

#Create data covariance matrix and simulate data with random intercept
K = (torch.reshape(torch.sqrt(vg),(n,1)) @ torch.reshape(torch.sqrt(vg),(n,1)).T) * G + torch.diag(ve)
R = torch.linalg.cholesky(K)
y = torch.rand(1) + R @ torch.randn(n,dtype=torch.double)

######################################################################################################################

#Estimate variance functions
######################################################################################################################
#Create fixed effects matrix X
X = torch.ones((n,1),dtype=torch.double)

#Create computational grid
ngrid = 50
xgrid = torch.linspace(x.min(),x.max(),ngrid)

#Create GP covariance and interpolation matrices
D = torch.tensor(squareform(pdist(xgrid.reshape(ngrid,1).cpu())))
l =  D.max()/3  #Length-scale
C = (1 + sqrt(3)*D/l)*torch.exp(-sqrt(3)*D/l)
Cinv = torch.linalg.inv(C)
D_interp = torch.tensor(cdist(x.reshape(n,1).cpu(),xgrid.reshape(ngrid,1).cpu()))
C_interp = (1 + sqrt(3)*D_interp/l)*torch.exp(-sqrt(3)*D_interp/l)

step_opt = 1 #Initial optimization step length
step_MCMC = 0.25 #MCMC step length
Nsim = 1000 #Number of MCMC rounds


stdy = y.std().item()
y_scaled = y/(stdy*sqrt(0.5))   
chains = functions.estimate_variances(G,y_scaled,X,x,xgrid,step_opt,step_MCMC,ngrid,C_interp,Cinv,Nsim)

######################################################################################################################

#Plot results
######################################################################################################################
alpha = 0.05 #100*(1-alpha) % credible interval

cm_ve = torch.median(torch.exp(chains[:,0:ngrid])*0.5*stdy**2,0)[0]
cm_vg = torch.median(torch.exp(chains[:,ngrid:2*ngrid])*0.5*stdy**2,0)[0]
crint_ve = torch.quantile(torch.exp(chains[:,0:ngrid])*0.5*stdy**2,torch.tensor([alpha/2,1-alpha/2],dtype=torch.double),0)
crint_vg = torch.quantile(torch.exp(chains[:,ngrid:2*ngrid])*0.5*stdy**2,torch.tensor([alpha/2,1-alpha/2],dtype=torch.double),0)
h2 = torch.exp(chains[:,ngrid:2*ngrid])/(torch.exp(chains[:,0:ngrid]) + torch.exp(chains[:,ngrid:2*ngrid]))
cm_h2 = torch.median(h2,0)[0]
crint_h2 = torch.quantile(h2,torch.tensor([alpha/2,1-alpha/2],dtype=torch.double),0)
vetrue = torch.exp(legrid)
vgtrue = torch.exp(lggrid)

plt.figure(figsize=(18, 6), dpi=300)
plt.subplot(1,3,1)
plt.plot(xgrid.cpu(),cm_ve.cpu(),color="black")
plt.plot(xgrid.cpu(),crint_ve[0,:].cpu(),"--",color="black",linewidth=1)
plt.plot(xgrid.cpu(),crint_ve[1,:].cpu(),"--",color="black",linewidth=1)
plt.plot(xgrid.cpu(),vetrue.cpu(),"--",color="red",linewidth=0.5)
plt.xlabel('Covariate',fontsize=15)
plt.ylabel("Residual variance",fontsize=15)
plt.subplot(1,3,2)
plt.plot(xgrid.cpu(),cm_vg.cpu(),color="black")
plt.plot(xgrid.cpu(),crint_vg[0,:].cpu(),"--",color="black",linewidth=1)
plt.plot(xgrid.cpu(),crint_vg[1,:].cpu(),"--",color="black",linewidth=1)
plt.plot(xgrid.cpu(),vgtrue.cpu(),"--",color="red",linewidth=0.5)
plt.xlabel('Covariate',fontsize=15)
plt.ylabel("Genetic variance",fontsize=15)
plt.subplot(1,3,3)
plt.plot(xgrid.cpu(),cm_h2.cpu(),color="black")
plt.plot(xgrid.cpu(),crint_h2[0,:].cpu(),"--",color="black",linewidth=1)
plt.plot(xgrid.cpu(),crint_h2[1,:].cpu(),"--",color="black",linewidth=1)
plt.plot(xgrid.cpu(),vgtrue.cpu()/(vgtrue.cpu() + vetrue.cpu()),"--",color="red",linewidth=0.5)
plt.xlabel('Covariate',fontsize=15)
plt.ylabel("Heritability",fontsize=15)