#%%
import matplotlib.pyplot as plt
import numpy as np
from qutip import *
import time
start_time = time.time()#checking how long the code takes

# to compute QFI of pure state
# u is state, A is photon number operator
def QFI_pure(u,A):
    aa1 = 4*(u.dag()*A*A*u-(u.dag()*A*u)*(u.dag()*A*u)) #4*(u.dag()*A*A*u-(u.dag()*A*u)**2)
    F = aa1[0,0]
    return F.real

# for cavity
cdim = 100
a = destroy(cdim)

AL = np.linspace(0,4,20) # varying alpha
Nth = np.arange(0,20,1) # varying average photon number
H = a.dag()*a # photon number operator
eta = H#(a.dag()+a) # photon number operator
# H = eta

# initialise for states: coherent, cat, and 0+alpha
F_coh = AL*0
F_0al = AL*0
F_cat = AL*0
N_coh = AL*0
N_0al = AL*0
N_cat = AL*0
NN2 = AL*0
for k in np.arange(0,len(AL)):
    al = AL[k]# alpha

    #the states
    u_coh = coherent(cdim,al)
    u_0al = (coherent(cdim,0)+coherent(cdim,al)).unit()
    u_cat = (coherent(cdim,-al)+coherent(cdim,al)).unit()
    st = coherent(cdim,0)+coherent(cdim,al)
    NN2[k] = 1/(st.dag()*st)[0,0]

    # fisher information
    F_coh[k] = QFI_pure(u_coh,H)
    F_cat[k] = QFI_pure(u_cat,H)
    F_0al[k] = QFI_pure(u_0al,H)

    #average photon number
    N_coh[k] = (u_coh.dag()*H*u_coh)[0,0].real
    N_cat[k] = (u_cat.dag()*H*u_cat)[0,0].real
    N_0al[k] = (u_0al.dag()*H*u_0al)[0,0].real

r_s = np.linspace(0,1.5,15)*np.exp(1j*np.pi*1)
F_sq = r_s*0
N_sq = r_s*0
for k in range(len(r_s)):
    r = r_s[k]
    S = (0.5*(np.conjugate(r)*a*a-r*a.dag()*a.dag())).expm()
    u_sq = S*coherent(cdim,0)
    F_sq[k] = QFI_pure(u_sq,eta)
    N_sq[k] = (u_sq.dag()*H*u_sq)[0,0].real

# initialise for states: 0+n
F_0n = np.zeros([len(Nth),1])
N_0n = F_0n*0
for k in np.arange(0,len(Nth)):
    n = Nth[k]
    u_0n = (basis(cdim,0)+basis(cdim,n)).unit()# the state
    # u_n = (basis(cdim,n)).unit()# the state
    F_0n[k] = QFI_pure(u_0n,eta)# the fisher information
    N_0n[k] = (u_0n.dag()*H*u_0n)[0,0].real #average photon number
    # N_n[k] = (u_n.dag()*H*u_n)[0,0].real #average photon number
    # F_n[k] = QFI_pure(u_n,eta)# the fisher information
    # print(n)

fig = plt.figure(figsize = (7,5))

# plt.plot(N_coh,N_coh*0,'c-',label=r'$|n\rangle$')# for Fock, we know that it's always zero

###for phase
plt.plot(N_coh,F_coh,'b-',label=r'$|\alpha\rangle$')# for coherent state
# plt.plot(N_coh,4*N_coh,'m--',label=r'$|\alpha\rangle$')# for coherent state: analystical calculation

plt.plot(N_0n,F_0n,'k-',label='SFS') # for superpositon of fock state
# plt.plot(N_0n,4*N_0n**2,'sr',label='SFS') # for superpositon of fock state analystical calculation

plt.plot(N_0al,F_0al,'g-',label=r'$\mathcal{N}(|0\rangle+|\alpha\rangle)$')  # for 0+alpha
# plt.plot(N_0al,4*(1-NN2)/NN2*N_0al**2+4*N_0al,'k--',label=r'$\mathcal{N}(|0\rangle+|\alpha\rangle)$') #  for superpositon of fock state analystical calculation

plt.plot(N_sq,F_sq,'r-',label=r'$S|0\rangle$')  # for sq state
# plt.plot(N_sq,8*N_sq**2+8*N_sq,'c--',label=r'$S|0\rangle$') # for sq state for superpositon of fock state

plt.plot(N_sq,N_sq*0,'r-',label=r'$N$') # for fock state
plt.plot(N_cat,F_cat,'k--',label=r'$\mathcal{N}(|\alpha\rangle+|-\alpha\rangle)$')# for cat



###for amplitude
# plt.plot(N_coh,F_coh,'b-',label=r'$|\alpha\rangle$')# for coherent state
# plt.plot(N_coh,N_coh*0+4,'m--',label=r'$|\alpha\rangle$')# for coherent state

# plt.plot(N_0al,F_0al,'g-',label=r'$\mathcal{N}(|0\rangle+|\alpha\rangle)$')
# plt.plot(N_0al,8*N_0al+4,'w--',label=r'$\mathcal{N}(|0\rangle+|\alpha\rangle)$')

# plt.plot(N_sq,F_sq,'r-',label=r'$S|0\rangle$')
# plt.plot(N_sq,8*np.sqrt(N_sq*(N_sq+1))+8*N_sq+4,'c--',label=r'$S|0\rangle$')

# plt.plot(N_0n,F_0n,'k-',label='SFS')
# # plt.plot(N_0n,4*N_0n**2,'sr',label='SFS')

plt.xlabel(r'$\langle N\rangle $')
plt.ylabel('Quantum Fisher information')
plt.legend()
plt.xlim([0,4.5])
plt.ylim([-1,60])

plt.savefig('/Users/panxiaozhou/Downloads/QFI_different_state1.pdf')
# plt.show()

print("")
print("--- %s seconds ---" % (time.time() - start_time))

#%%

Nmax = 10

F1 = np.zeros(Nmax,dtype=np.float_)
for j in range(Nmax):
    uf = fock(cdim,j)
    F1[j] = QFI_pure(uf,eta)

AL = np.linspace(0,np.sqrt(Nmax)*2,10)
F2 = np.zeros(Nmax,dtype=np.float_)
N2 = np.zeros(Nmax,dtype=np.float_)
Nave = np.zeros(Nmax,dtype=np.float_)
for j in range(len(AL)):
    us = (coherent(cdim,0)+coherent(cdim,AL[j])).unit()
    ov = (coherent(cdim,0)+coherent(cdim,AL[j])).dag()*(coherent(cdim,0)+coherent(cdim,AL[j]))
    N2[j] = 1/ov[0,0]
    F2[j] = QFI_pure(us,eta)
    Nave[j] = (us.dag()*H*us)[0,0].real

N = np.arange(0,Nmax)
Ns = AL**2

# plt.plot(Ns,F2,'b',)
# plt.plot(Ns,8*Ns*N2+4,'g--')
plt.plot(Nave,F2,'b:',linewidth=5)
plt.plot(Nave,8*Nave+4,'d')
plt.plot(N,F1,'k')
plt.plot(N,8*N+4,'r--')

# %%
