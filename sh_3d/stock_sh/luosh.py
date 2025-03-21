#!/usr/bin/python3
import numpy as np
import input as inp
from functools import reduce
import matplotlib.pyplot as plt

class SH():
    def __init__(self):
        self.state = inp.state
        self.ini_afaj, self.ini_xj, self.ini_pj = self.sample()
        self.gailv = []
              
    def sample(self, nj = None):
        '''
        action angle
        '''
        if nj is None : nj = [0] * inp.num_vib
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
    
    def dia_ele_diagonal(self,wf,hami):
        eigval, S = np.linalg.eigh(hami)
        S_diag = np.linalg.inv(S)
        change = np.diag(np.exp(-(1j) * np.array(eigval) * inp.dt))
        change_hami = reduce(np.dot, [S, change, S_diag])
        next_wf = np.dot(change_hami, wf)
        return next_wf
    
    def dia_to_adwf(self,xj,wf):
        eigval,S = np.linalg.eigh(self.potential_eng(xj))
        S_diag = np.linalg.inv(S)
        ad_wf = np.dot(S_diag, wf)
        return ad_wf
    
    def deri_all(self,xj):
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
    
    def nondiag_fist_coupling(self,xj):
        deri_all, eigval = self.deri_all(xj)
        dE = eigval[0] - eigval[1]
        T12s = []
        T21s = []
        for i in range(len(deri_all)):
            T12 = - deri_all[i][0][1] / dE
            T21 = deri_all[i][1][0] / dE
            T12s.append(T12)
            T21s.append(T21)
        T12s = np.array(T12s)
        T21s = np.array(T21s)
        return T12s,T21s
    
    def hopping_pro(self,den,xj,pj):
        T12s,T21s = self.nondiag_fist_coupling(xj)
        A12 = - np.sum(T12s * inp.omiga * pj)
        A21 = - np.sum(T21s * inp.omiga * pj)
        rand_num = np.random.random()
        if self.state == 'state1':
            pro12 = 2 * (inp.dt * den[0,1] * A21 / den[0,0]).real
            pro12 = max([pro12,0])
            if pro12 > rand_num:
                new_state = 'state2'
                pj = self.vel_adjust(xj,pj,T12s,T21s,new_state)
        else:
            pro21 = 2 * (inp.dt * den[1,0] * A12 / den[1,1]).real
            pro21 = max([pro21,0])
            if pro21 > rand_num:
                new_state = 'state1'
                pj = self.vel_adjust(xj,pj,T12s,T21s,new_state)
        return pj
        
    
    def vel_adjust(self,xj,pj,T12s,T21s,new_state):
        # print('vvvvvvvvv')
        Hele = self.potential_eng(xj)
        eigval , S = np.linalg.eigh(Hele)
        if new_state == 'state2':
            c12 = np.sum(0.5 * inp.omiga * T12s**2)
            b12 = np.sum(inp.omiga * pj * T12s)
            D = b12**2 - 4*c12*(eigval[1] - eigval[0])
            if D>= 0:
                # print('111to2222222222222')
                self.state = 'state2'
                sigama1 = (-b12 + D**0.5) / (2*c12)
                sigama2 = (-b12 - D**0.5) / (2*c12) 
                sigama = min([sigama1,sigama2], key=lambda x: abs(x - 0))
            else:
                sigama = 0
            pj = pj + sigama * T12s
        else:
            c21 = np.sum(0.5 * inp.omiga * T21s**2)
            b21 = np.sum(inp.omiga * pj * T21s)
            D = b21**2 - 4*c21*(eigval[0] - eigval[1])
            if D>= 0:
                # print('222to111111111')
                self.state = 'state1'
                sigama1 = (-b21 + D**0.5) / (2*c21)
                sigama2 = (-b21 - D**0.5) / (2*c21)
                sigama = min([sigama1,sigama2], key=lambda x: abs(x - 0))
            else:
                sigama = 0
            pj = pj + sigama * T21s
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
        wf_s = []
        ad_wf_s = []
        tot_draw = []
        afaj,xj,pj = self.ini_afaj, self.ini_xj, self.ini_pj
        print(afaj)
        wf = inp.wf
        for i in range(inp.time):
            dia_hami = self.dia_hami(pj,xj)
            wf = self.dia_ele_diagonal(wf, dia_hami)
            wf_s.append(wf)
            ad_wf = self.dia_to_adwf(xj,wf)
            den = self.den_martix(ad_wf)
            ad_wf_s.append(ad_wf)
            xj, pj = self.nul_adibatic(xj,pj)
            pj = self.hopping_pro(den,xj,pj)
            self.gailv.append(self.get_gailv())
            xj_s.append(xj)
            pj_s.append(pj)
            poti_eng,kine_eng,totl_eng = self.energy(xj,pj)
            # print(poti_eng,kine_eng,totl_eng)
            tot_draw.append(totl_eng)
        pj_1s = [i[0].real for i in pj_s]
        pj_6as = [i[1].real for i in pj_s]
        return pj_1s, pj_6as
        # for wr in range(len(pj_1s)):
        #     self.output.write(f'{pj_1s[wr]}                   {pj_6as[wr]}\n')
        # self.output.write('\n\n')
        # self.output.close()  
        
        # pj_draw = [i[0].real for i in pj_s]
        # times_draw = np.arange(0,inp.time)
        # plt.plot(times_draw,tot_draw,color='red',linestyle='--')
        # plt.plot(times_draw,pj_draw,color='red',linestyle='--')
        # plt.show()
    
def many_traj():
    pj1s_aver = []
    pj6a_aver = []
    gailv_aver = []
    output = open('good.txt','w')
    for i in range(inp.traj):
        sh = SH()
        print(i+1)
        pj_1s, pj_6as = sh.run()
        pj1s_aver.append(pj_1s)
        pj6a_aver.append(pj_6as)
        gailv_aver.append(sh.gailv)
    pj1s_aver = np.array(pj1s_aver)
    pj6a_aver = np.array(pj6a_aver)
    pj1s_aver = np.einsum('ij->j', pj1s_aver) / inp.traj
    pj6a_aver = np.einsum('ij->j', pj6a_aver) / inp.traj
    gailv_aver = np.einsum('ij->j', gailv_aver) / inp.traj
    
    x_draw = np.arange(0,inp.time)
    plt.figure('pigdraw' , figsize = (10 , 5))
    plt.plot(x_draw, pj1s_aver)
    plt.savefig('./p1s.jpg' , dpi=600)
    plt.clf()
    plt.plot(x_draw, pj6a_aver)
    plt.savefig('./p6a.jpg' , dpi=600)
    plt.clf()
    plt.plot(x_draw, gailv_aver)
    plt.savefig('./gailv.jpg' , dpi=600)
    
    for i in range(inp.time):
        output.write(f'{pj1s_aver[i]}         {pj6a_aver[i]}          {gailv_aver[i]}\n')
    output.write('\n\n')
    output.close()
    
if __name__ == "__main__":
    # c.run()
    many_traj()
    