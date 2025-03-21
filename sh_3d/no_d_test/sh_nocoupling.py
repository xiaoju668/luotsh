import numpy as np
import input as inp
from functools import reduce
import matplotlib.pyplot as plt

"""这个是没有nac的情况下的sh
   paper:https://doi.org/10.1021/acs.jctc.3c00813"""
   
class SH_no_d():
    def __init__(self):
        self.state = inp.state
        self.ini_afaj, self.ini_xj, self.ini_pj = self.sample()
        self.gailv = []
              
    def sample(self, nj = None):
        '''
        action angle
        '''
        if nj is None : nj = [0] * inp.num_vib
        # np.random.seed(0)
        afaj = [np.random.random() * 2 * np.pi  for i in range(inp.num_vib)]
        nj = np.array(nj , dtype=np.float64)
        afaj = np.array(afaj , dtype=np.float64)
        ini_xj = np.sqrt(2 * nj + 1) * np.sin(afaj)
        ini_pj = np.sqrt(2 * nj + 1) * np.cos(afaj)
        return afaj, ini_xj, ini_pj
    
    def kinetic_eng(self,pj):
        h0 = np.sum(0.5 * inp.omiga * pj**2)
        Tn = np.diag([h0,h0])
        # print(Tn)
        return Tn
    
    def potential_eng(self,xj):
        vee = np.sum(0.5 * inp.omiga * xj**2)
        vne1 = np.sum(inp.kai1 * xj)
        vne2 = np.sum(inp.kai2 * xj)
        v11 = inp.energy1 + vne1 + vee
        v22 = inp.energy2 + vne2 + vee
        v12 = inp.lamda * xj[2]
        v21 = v12
        Hele = np.array([[v11,v12],[v21,v22]])
        # print(Hele)
        return Hele
    
    def dia_hami(self,pj,xj):
        dia_hami = self.kinetic_eng(pj) + self.potential_eng(xj)
        # print(dia_hami)
        return dia_hami
    
    def adi_hami(self, dia_hami, coupling):
        eigval, S = np.linalg.eigh(dia_hami)
        coupling = np.complex128 ((1j) * coupling)
        adi_hami = np.array([[eigval[0], -coupling],
                             [coupling, eigval[1]]])
        return adi_hami
    
    def coupling_gradient_formula(self, xj, pj, last_xj, last_pj):
        deri_delta_v = self.deri_delta_v(xj,pj)
        # print('now',deri_delta_v)
        last_deri_delta_v = self.deri_delta_v(last_xj,last_pj)
        # print('last', last_deri_delta_v)
        deri_deri_delta_v = (deri_delta_v - last_deri_delta_v) / inp.dt
        Hele = self.potential_eng(xj)
        eigval, S = np.linalg.eigh(Hele)
        radicand = deri_deri_delta_v / (eigval[0] - eigval[1])
        if radicand < 0:
            coupling = 0
        else:
            coupling = 0.5 * np.sqrt(radicand)
        return coupling
           
    def deri_all(self,xj):
        '''obtain the first-order adibatic hamiltonian'''
        one1 = inp.omiga * xj + inp.kai1
        one2 = inp.omiga * xj + inp.kai2
        deri_x1 = np.array([[ one1[0] , 0 ],[ 0 , one2[0] ]])
        deri_x2 = np.array([[ one1[1] , 0 ],[ 0 , one2[1] ]])
        deri_x3 = np.array([[ one1[2] , inp.lamda ],[ inp.lamda , one2[2] ]])
        Hele = self.potential_eng(xj)
        eigval , S = np.linalg.eigh(Hele)
        S_diag = np.linalg.inv(S)
        deri_x1 = reduce(np.dot , [S_diag , deri_x1 , S])
        deri_x2 = reduce(np.dot , [S_diag , deri_x2 , S])
        deri_x3 = reduce(np.dot , [S_diag , deri_x3, S])
        deri_all = np.array([deri_x1 , deri_x2 , deri_x3] )
        return deri_all,eigval
    
    def nul_force(self,xj):
        """obtain adibatic force"""
        deri_all = self.deri_all(xj)[0]
        force1s = []
        force2s = []
        for i in range(len(deri_all)):
            force1 =  -deri_all[i][0][0]
            force1s.append(force1)
            force2 =  -deri_all[i][1][1]
            force2s.append(force2)
        return force1s, force2s
    
    def deri_delta_v(self,xj,pj):
        '''gradient formula 的其中一步'''
        force1s, force2s = self.nul_force(xj)
        delta_force = np.array(force1s) - np.array(force2s)
        deri_deta_v = np.sum(delta_force * inp.omiga * pj)
        return deri_deta_v
    
    def ele_diagonal(self,hami,wf):
        eigval, S = np.linalg.eigh(hami)
        S_diag = np.conj(S.T)
        change = np.diag(np.exp(-(1j) * np.array(eigval) * inp.dt))
        change_hami = reduce(np.dot, [S, change, S_diag])
        next_wf = np.dot(change_hami, wf)
        # print(next_wf)
        return next_wf

    def nul_stateforce(self,xj):
        deri_all = self.deri_all(xj)[0]
        force1s = []
        force2s = []
        for i in range(len(deri_all)):
            force1 =  -deri_all[i][0][0]
            force1s.append(force1)
            force2 =  -deri_all[i][1][1]
            force2s.append(force2)
        if self.state == 'state1':
            forces = np.array(force1s)
        elif self.state == 'state2':
            forces = np.array(force2s)
        return forces
    
    def nul_adibatic(self,xj,pj):
        '''
        by rf4,obtain next xj,pj
        '''
        k11 = pj * inp.omiga
        k21 = self.nul_stateforce(xj)
        k12 = (pj + 0.5*inp.dt*k21)*inp.omiga
        xj_1 = xj + 0.5*inp.dt*k11
        k22 = self.nul_stateforce(xj_1)
        k13 = (pj + 0.5*inp.dt*k22)*inp.omiga
        xj_2 = xj + 0.5*inp.dt*k12
        k23 = self.nul_stateforce(xj_2)
        k14 = (pj + inp.dt*k23)*inp.omiga
        xj_3 = xj + inp.dt*k13
        k24 = self.nul_stateforce(xj_3)
        next_xj = xj + (1/6)*inp.dt*(k11 + 2*k12 + 2*k13 + k14)
        next_pj = pj + (1/6)*inp.dt*(k21 + 2*k22 + 2*k23 + k24)
        return next_xj,next_pj
    
    def den_martix(self,wf):
        den_matrix = np.dot(wf,wf.T.conjugate())
        return den_matrix
    
    def hopping_pro(self,den,coupling,xj,pj):
        rand_num = np.random.random()
        if self.state == 'state1':
            pro12 = 2 * (inp.dt * den[0,1] * coupling / den[0,0]).real
            pro12 = max([pro12,0])
            if pro12 > rand_num:
                new_state = 'state2'
                pj = self.vel_adjust(xj,pj,coupling,new_state)
        else:
            pro21 = 2 * (inp.dt * den[1,0] * (-coupling) / den[1,1]).real
            pro21 = max([pro21,0])
            if pro21 > rand_num:
                new_state = 'state1'
                pj = self.vel_adjust(xj,pj,coupling,new_state)
        return pj
    
    def vel_adjust(self,xj,pj,coupling,new_state):
        # print('vvvvvvvvv ajust')
        Hele = self.potential_eng(xj)
        eigval , S = np.linalg.eigh(Hele)
        if new_state == 'state2':
            c12 = np.sum(coupling**2 / (2 * inp.omiga * pj**2) )
            b12 = coupling
            D = b12**2 - 4*c12*(eigval[1] - eigval[0])
            if D>= 0:
                # print('111to2222222222222')
                self.state = 'state2'
                sigama1 = (-b12 + D**0.5) / (2*c12)
                sigama2 = (-b12 - D**0.5) / (2*c12) 
                sigama = min([sigama1,sigama2], key=lambda x: abs(x - 0))
            else:
                sigama = 0
            pj = pj + sigama * (coupling / (inp.omiga * pj))
        else:
            c21 = np.sum((-coupling)**2 / (2 * inp.omiga * pj**2) )
            b21 = -coupling
            D = b21**2 - 4*c21*(eigval[0] - eigval[1])
            if D>= 0:
                # print('222to111111111')
                self.state = 'state1'
                sigama1 = (-b21 + D**0.5) / (2*c21)
                sigama2 = (-b21 - D**0.5) / (2*c21)
                sigama = min([sigama1,sigama2], key=lambda x: abs(x - 0))
            else:
                sigama = 0
            pj = pj + sigama * (-coupling / (inp.omiga * pj))
        return pj
    
    def energy(self,xj,pj):
        Hele = self.potential_eng(xj)
        eigval, S = np.linalg.eigh(Hele)
        if self.state == 'state1':
            poti_eng = eigval[0]
        else:
            poti_eng = eigval[1]
        kine_eng = np.sum(0.5 * inp.omiga * pj**2)
        totl_eng = poti_eng + kine_eng
        return poti_eng,kine_eng,totl_eng
    
    def get_gailv(self):
        if self.state == 'state1':
            return 0
        elif self.state == 'state2':
            return 1 
    
    def run(self):
        xj_s = []
        pj_s = []
        ad_wf_s = []
        tot_draw = []
        afaj,xj,pj = self.ini_afaj, self.ini_xj, self.ini_pj
        wf = inp.wf
        next_xj, next_pj = self.nul_adibatic(xj,pj)
        next_coupling = self.coupling_gradient_formula(next_xj, next_pj, xj, pj)
        dia_hami = self.dia_hami(next_pj,next_xj)
        adi_hami = self.adi_hami(dia_hami, next_coupling)
        for i in range(inp.time):
            xj, pj = next_xj, next_pj
            next_xj, next_pj = self.nul_adibatic(xj,pj)
            xj_s.append(next_xj)
            next_wf = self.ele_diagonal(adi_hami, wf)
            ad_wf_s.append(next_wf)
            next_den = self.den_martix(next_wf)
            next_coupling = self.coupling_gradient_formula(next_xj, next_pj, xj, pj)
            next_pj = self.hopping_pro(next_den, next_coupling, next_xj, next_pj)
            pj_s.append(next_pj)
            poti_eng,kine_eng,totl_eng = self.energy(next_xj,next_pj)
            self.gailv.append(self.get_gailv())
            tot_draw.append(totl_eng)
            wf = next_wf
            dia_hami = self.dia_hami(next_pj,next_xj)
            adi_hami = self.adi_hami(dia_hami, next_coupling)
        return
            
            
def many_traj():
    pj1s_aver = []
    pj6a_aver = []
    gailv_aver = []
    for i in range(inp.traj):
        sh = SH_no_d()
        print(i+1)
        sh.run()
        gailv_aver.append(sh.gailv)
    gailv_aver = np.einsum('ij->j', gailv_aver) / inp.traj
    
    x_draw = np.arange(0,inp.time)
    plt.figure('pigdraw' , figsize = (10 , 5))
    plt.plot(x_draw, gailv_aver)
    plt.savefig('./gailv.jpg' , dpi=600)
             

if __name__ == '__main__':
    # a.run()
    many_traj()