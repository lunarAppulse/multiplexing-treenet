import numpy as np
from math import exp,sqrt
import pandas as pd
from datetime import datetime
import os
from multiprocessing import Pool

#获取相关系数
def get_correlation(args):   
    phi, sigma, Id0, Is0, Id_base, Is_base, gs, cd, gd, gc, p, tau_d_exit, tau_s_exit = args
    Id=np.zeros(N+1)
    for i in range(N+1):
        k=((i-td0/dt*phi)%(td0/dt))/(td0/dt)
        if k<=low_ratio_d:
            Id[i]=Id_base
        else:
            Id[i]=Id0
    

    Is=np.zeros(N+1)
    for i in range(N+1):
        k=i%(ts0/dt)/(ts0/dt)
        if k<=low_ratio_s:
            Is[i]=Is_base
        else:
            Is[i]=Is0


    #统计：在time时间长度内各个1毫秒内计数  单位：Hz = s^-1
    fireRate=np.zeros(time)			#spike rate
    eventRate=np.zeros(time)		#single spike 加上 burst
    burstRate=np.zeros(time)
    burstFraction=np.zeros(time)   	#burst probability
    
    Voltage_corr = 0

    #seed = np.random.randint(100000,size=neuronNumber)
    for counter in range(0, neuronNumber):
        
        #设置噪声
        np.random.seed()
        z1, z2=np.random.normal(0,sigma*sqrt(2*alpha),(2,N))
    
        noise1, noise2 = np.zeros((2,N+1))
        for i in range(N):
    	    noise1[i+1]=noise1[i]-alpha*noise1[i]*dt+z1[i]*np.sqrt(dt)
    	    noise2[i+1]=noise2[i]-alpha*noise2[i]*dt+z2[i]*np.sqrt(dt)
    
    
        #Euler method
        Vs=np.zeros(N+1)
        Vs[0]=EL
        Vd=np.zeros(N+1)
        Vd[0]=EL
    
        ws=np.zeros(N+1)
        wd=np.zeros(N+1)
    
        S=[]   #spike train
        spikeNumber=0
        Convolution_K_S=0
        i1=0    #右方最靠近(t-2.5)时刻的spike指标
        i2=0    #右方最靠近(t-0.5)时刻的spike指标
        timeInterval1=25   #(t-2.5)与S[i1]的时间间隔/0.1
        timeInterval2=5   #(t-0.5)与S[i2]的时间间隔/0.1
        
        for i in range(1,N+1,1):
            Vd[i]=(-tau_d_exit*(Vd[i-1]-EL)/tau_d+(gc/(1-p)*(Vs[i-1]-Vd[i-1])+gd*f(Vd[i-1])+Id[i-1]+noise1[i-1]-wd[i-1]+cd*Convolution_K_S)/Cd)*dt + Vd[i-1]
            wd[i]=(-wd[i-1]+a*(Vd[i-1]-EL))/tau_wd*dt + wd[i-1]
            if Vs[i-1] >= VT:
                Vs[i] = Vr
                S.append(t[i-1])
                spikeNumber = spikeNumber+1
                ws[i]=-ws[i-1]/tau_ws*dt + b + ws[i-1]
            else:
                Vs[i]=(-tau_s_exit*(Vs[i-1]-EL)/tau_s+(gc/p*(Vd[i-1]-Vs[i-1])+gs*f(Vd[i-1])+Is[i-1]+noise2[i-1]-ws[i-1])/Cs)*dt + Vs[i-1]
                ws[i]=-ws[i-1]/tau_ws*dt + ws[i-1]
            
            #replace convolution
            if i1 != spikeNumber:
                timeInterval1 = timeInterval1 - 1
                if timeInterval1 == 0:
                    Convolution_K_S = Convolution_K_S - 1
                    i1=i1 + 1
                    if i1 == spikeNumber:
                        timeInterval1=25
                    else:
                        timeInterval1=round((S[i1]-(t[i]-2.5))/dt)
                if i2 != spikeNumber:
                    timeInterval2 = timeInterval2 - 1
                    if timeInterval2 == 0:
                        Convolution_K_S=Convolution_K_S+1
                        i2=i2 + 1
                        if i2 == spikeNumber:
                            timeInterval2=5
                        else:
                            timeInterval2=round((S[i2]-(t[i]-0.5))/dt)
        
        #计算电压相关系数    
        Voltage_corr += abs(np.corrcoef(Vs[1000:-1000], Vd[1000:-1000])[0,1])
                            
        #计算四个统计量                     
        B=[]	#burst train
        E=[]	#event train
        i=0
        if spikeNumber>=2:
    	    while i<(spikeNumber-1):
    	        s=S[i]
    	        E.append(s)
    	        if (S[i+1]-s) <= 15:
    	            B.append(s)
    	            i=i+2
    	            while i<(spikeNumber-1):
    	                if (S[i]-s)>15:
    	                    break
    	                i=i+1
    	            continue
    	        i=i+1
    
        for point in S:
            fireRate[int(point)]+=1
        for point in E:
            eventRate[int(point)]+=1
        for point in B:
            burstRate[int(point)]+=1
            
    for i in range(time):
    	if eventRate[i]!=0:
    		burstFraction[i]=burstRate[i]/eventRate[i]
            
    Corr_1 = np.corrcoef(eventRate[100:-100], Is[1005:-1000:10])[0,1]  
    Corr_2 = np.corrcoef(burstFraction[100:-100], Id[1005:-1000:10])[0,1]
    Multiplex_metric = (Corr_1 + Corr_2)/2
    Voltage_corr = Voltage_corr/neuronNumber

    return Voltage_corr, Corr_1, Corr_2, Multiplex_metric




#模型参数
EL=-70    #mV
tau_s=16  #ms
Cs=370    #pF
tau_ws=100  #100ms
b=200       #pA  strength of spike-triggered adaptation,
VT=-50    #mV

tau_d=7   #ms
#gd=1200   #pA
Cd=170    #pF
#cd=2600   #pA
tau_wd=30   #30ms
a=13        #nS=nA/V=pA/mV

Ed=-38    #mV
Dd=6      #mV

Vr=-70    #mV  reset voltage after a spike


#时间精度、长度设置
dt=0.1    #ms
time=600  #ms
N=round(time/dt)
t=np.arange(0,N+1)*dt #t=[0,0.1,0.2,...,N*dt]

#噪声参数设置
alpha=1

#无噪声信号输入设置
td0=200    #ms
low_ratio_d=0.4
ts0=200   #ms
low_ratio_s=0.5

#f function
f = lambda x: 1/(1 + exp((Ed-x)/Dd))



#参数设置集中区***************************************************************
neuronNumber = 8000   #神经元数目，建议1000
pointNum = 5        #检索数据点数，越多越好，但耗时会更长

#注意：程序运行消耗时间正比于以上这两个数乘积

#选择若干变量设置其变动范围
phi =   0.4   #phase shift 0~1  0.4 
sigma = 550  #standard deviation(intensity) for noises  350
Id0 = 650   #pA  340
Is0 = 750   #pA  600 取值必须大于Is_base
Id_base = 0         #0
Is_base = 300       #300
gs = 2500           #1300pA
cd = 2500       #2600pA
gd = 1200       #1200pA
gc = 2.1        #2.1mS,不想引入p和gc项就设为0
p = 0.5         #0<p<1
tau_d_exit = 1  #存在为1，缺席为0，表示tau_d0/tau_d
tau_s_exit = 1  #同上
#参数设置完毕***************************************************************

labels_short=['phi','sigma','Id0','Is0','Id_base','Is_base','gs','cd', 'gd', 'gc', 'p', 'tau_d_exit', 'tau_s_exit']
parameters=[phi, sigma, Id0, Is0, Id_base, Is_base, gs, cd, gd, gc, p, tau_d_exit, tau_s_exit]

index=[]
args=[0]*13
dataframe = pd.DataFrame() 

for i in range(13):
    if isinstance(parameters[i], list):
        
        index.append(i)
        data = np.random.rand(pointNum)
        for j in range(pointNum):
            data[j] = parameters[i][0] + data[j] * (parameters[i][1] - parameters[i][0])
        dataframe[labels_short[i]] = data.copy()
        
    else:
        args[i] = parameters[i]
        
def DataProcess(dataframe, index, args):
    
    
    #多进程并行运算
    process_pool = Pool()
    results = []  
    Args = []
    for i in range(pointNum):
        for j in index:
            args[j] = dataframe[labels_short[j]][i]
        Args.append(args.copy())    
    results = process_pool.map(get_correlation, Args)
     
    process_pool.close()
    process_pool.join()

    #四个系数
    Voltage_corr = np.zeros(pointNum)
    Corr_1 = np.zeros(pointNum)
    Corr_2 = np.zeros(pointNum)
    Multiplex_metric = np.zeros(pointNum)
    for i, result in enumerate(results):
        Voltage_corr[i], Corr_1[i], Corr_2[i], Multiplex_metric[i] = result  

    #结果记录于表格
    dataframe['Corr(Vs,Vd)'] = Voltage_corr
    dataframe['Corr(ER,Is)'] = Corr_1
    dataframe['Corr(BF,Id)'] = Corr_2
    dataframe['Multiplex_metric'] = Multiplex_metric    

    return dataframe
    
if __name__=='__main__':
    
    folder = os.path.exists('Find_Best_Multiplex')
    if not folder:                                      #判断是否存在文件夹如果不存在则创建为文件夹
    		os.makedirs('Find_Best_Multiplex') 
    
    # 获取现在时间
    program_start=datetime.now()
    print('\n开始运行:',program_start,'\n')

    #冲！！
    dataframe = DataProcess(dataframe, index, args)
    
    #对表格重新排序，按多路复用程度降序
    sort_df = dataframe.sort_values(by='Multiplex_metric', ascending=False)
    
    #然后保存至csv文件
    sort_df.to_csv('Find_Best_Multiplex/Multiplex_metric with %(num)d cells.csv'%{"num": neuronNumber},index=False,sep=',')
    
    program_end=datetime.now()
    print('\n\n计算结束:',program_end)
    print('程序运行总时长:',program_end - program_start)
    print('\n\n展示Multiplex_metri最大的前6组数据如下:\n')
    print(sort_df.head(6).to_string(index=False))










