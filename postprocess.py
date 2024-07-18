import sys
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_lsqt_ave(filename,Emax,Emin,start_row,end_row,energy_points):
    data=np.loadtxt(filename)
    if start_row < 0 or end_row > len(data) or start_row >= end_row:
        print("Invalid row numbers.")
        sys.exit(1)

    subset_data = data[start_row:end_row, :]


    energy_array=np.linspace(Emin,Emax,energy_points)

    if len(subset_data) == 0:
        print("No dos data found in the specified range.")
        sys.exit(1)

    average_data = np.mean(np.abs(subset_data), axis=0, dtype=np.float64)

    save_data = np.column_stack((energy_array, average_data))

    np.savetxt('average_'+filename, save_data,fmt='%.6f %.12e')

    print("Average "+filename+" saved to average_"+filename)

    if filename=='lsqt_sigma.out':
        return energy_array,average_data



def df_de(E, mu, kbT):
    x = (E - mu) / kbT
    exp_term = np.exp(x)
    return (1 / kbT) * (exp_term / ((exp_term + 1)**2))

def X_function(average_data_sigma, E_range, mu, kbT,Emax):
    df = df_de(E_range, mu, kbT) * average_data_sigma
    E_size=E_range.size
    step_length=(np.max(E_range)-np.min(E_range))/E_size
    x0 = np.sum(df*step_length)
    x1 = np.sum(df * E_range*step_length)
    x2 = np.sum(df * E_range * E_range*step_length)
    return x0, x1, x2


calculate_transportation=False

if len(sys.argv) == 6:
    print("only calculate dos,velocity,sigma time average")
    start_row =int(sys.argv[1])
    end_row =int(sys.argv[2])
    Emin=float(sys.argv[3])
    Emax=float(sys.argv[4])
    energy_points=int(sys.argv[5])
    get_lsqt_ave("lsqt_dos.out",Emax,Emin,start_row,end_row,energy_points)
    get_lsqt_ave("lsqt_velocity.out",Emax,Emin,start_row,end_row,energy_points)
    get_lsqt_ave("lsqt_sigma.out",Emax,Emin,start_row,end_row,energy_points)
elif(len(sys.argv) == 8):
    start_row =int(sys.argv[1])
    end_row =int(sys.argv[2])
    Emin=float(sys.argv[3])
    Emax=float(sys.argv[4])
    Temp=float(sys.argv[5])
    energy_points=int(sys.argv[6])
    kl=float(sys.argv[7])
    calculate_transportation=True
    Temp_inv=1/Temp
    kb=8.617333262145e-5
    kbT=kb*Temp
    
    
    if not glob.glob("average_lsqt_dos.out"):
        get_lsqt_ave("lsqt_dos.out",Emax,Emin,start_row,end_row,energy_points)
        get_lsqt_ave("lsqt_velocity.out",Emax,Emin,start_row,end_row,energy_points)
        energy_array,average_data_sigma=get_lsqt_ave("lsqt_sigma.out",Emax,Emin,start_row,end_row,energy_points)
    else:
        energy_array,average_data_sigma=get_lsqt_ave("lsqt_sigma.out",Emax,Emin,start_row,end_row,energy_points)

    ###post-processing of transport properties
    sigma=[]
    seeback=[]
    kel=[]
    ZT=[]
    if calculate_transportation:
        integral_range=np.linspace(Emin,Emax,energy_points, dtype=np.float64)
        E_range=np.linspace(Emin,Emax,energy_points, dtype=np.float64)

        for mu in integral_range:
            x0,x1,x2= X_function(average_data_sigma,E_range,mu,kbT,Emax)
            
            seeback_val=-Temp_inv*(x1/x0-mu) 

            kel_val=Temp_inv*(x2-(x1*x1)/x0)
            
            ZT_val=(seeback_val**2*x0*Temp)/(kl+kel_val)
            
            
            sigma.append(x0)    #S/m
            seeback.append(seeback_val)#V/K
            kel.append(kel_val) # W/mk
            ZT.append(ZT_val)
            
            
            
        save_sigma = np.column_stack((energy_array, np.array(sigma)))
        save_seeback = np.column_stack((energy_array, np.array(seeback)))
        save_kel = np.column_stack((energy_array, np.array(kel)))
        
        np.savetxt('postprocess_sigma.out', save_sigma,fmt='%.6f %.12e')
        np.savetxt('postprocess_seeback.out', save_seeback,fmt='%.6f %.12e')
        np.savetxt('postprocess_kel.out', save_kel,fmt='%.6f %.12e')
        print("All Postprocess Done")

        plt.figure(figsize=(15, 10))

        
        plt.subplot(2, 2, 1)
        plt.plot(energy_array, sigma, color='blue')
        plt.title('Sigma')
        plt.xlabel('Energy')
        plt.ylabel('Value')
        #plt.xlim(-1, 1)
        #plt.ylim(0, 4e6)

        
        plt.subplot(2, 2, 2)
        plt.plot(energy_array, seeback, color='red')
        plt.title('Seeback')
        plt.xlabel('Energy')
        plt.ylabel('Value')
        #plt.xlim(-1, 1)
        #plt.ylim(-0.001, 0.001)

        
        plt.subplot(2, 2, 3)
        plt.plot(energy_array, kel, color='green')
        plt.title('Kel')
        plt.xlabel('Energy')
        plt.ylabel('Value')
        #plt.xlim(-1, 1)
        #plt.ylim(0, 15)
        
        plt.subplot(2, 2, 4)
        plt.plot(energy_array, ZT, color='orange')
        plt.title('ZT')
        plt.xlabel('Energy')
        plt.ylabel('Value')
        #plt.xlim(-1, 1)
        #plt.ylim(0, 15)

        
        plt.tight_layout()
        plt.savefig('result.png',dpi=800)
        
        #plt.show()
else:
    print("Wrong input info")
    sys.exit(1)

