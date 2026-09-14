import numpy as np
def silu(x): return x/(1+np.exp(-x))
Sg=21.0; xmax=19.5/Sg; vm=xmax*xmax
c=np.array([-52.555831396795135,228.7193911068195,-414.74107047011859,406.34322413539491,-233.55938542700747,80.378386397027882,-16.533501657853961,2.3986479950949571,0.012662442785882201])
gate=np.fromfile(r'.tmp_tok/tail/l26_gate.bin',dtype='<f4').astype(np.float64).reshape(4,6144)
up=np.fromfile(r'.tmp_tok/tail/l26_up.bin',dtype='<f4').astype(np.float64).reshape(4,6144)
Wd=np.fromfile(r'.tmp_tok/tail/w26/down_proj.bin',dtype='<f4').astype(np.float64).reshape(2048,6144)
sil_t=silu(gate)
x=gate/Sg; w=(x*x)/vm
sil_p=(0.5*x+np.polyval(c,w))*Sg
ds=sil_p-sil_t
dact=ds*up
print('sil err: max|ds|=%.4g  sil scale max=%.3g'%(np.abs(ds).max(), np.abs(sil_t).max()))
print('dact: max|dact|=%.4g  act max=%.3g'%(np.abs(dact).max(), np.abs(sil_t*up).max()))
for t in range(4):
    dmlp=dact[t].dot(Wd.T)
    du2=dmlp/1024.0
    print('t%d: max|dmlp|=%.4g -> du2 max=%.4g'%(t,np.abs(dmlp).max(),np.abs(du2).max()))
