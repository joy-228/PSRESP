import numpy as np
import random as ran
import numpy.random as npr
import numpy.fft as nft
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
import matplotlib.colors as mcl
from scipy.weave import inline

c1=mcm.Dark2(np.linspace(0,1,9))
c2=mcm.Set2(np.linspace(0,1,9))

#################################################################################################
###### This section is about reading NoF number of lightcurves stored in a folder ############### 
###### named objct located in the same directory. The paths of these lightcurves  ###############
###### are listed in the path.txt located inside objct. The lbl and instru can be ###############
###### used to record the time this observation took place and the telescope used.###############
#################################################################################################  

objct=raw_input('Enter object name:')
paths=np.genfromtxt(objct+'/path.txt',dtype=str,usecols=0)
lbl=np.genfromtxt(objct+'/path.txt',dtype=str,usecols=2)
instru=np.genfromtxt(objct+'/path.txt',dtype=str,usecols=1)
NoF=len(paths)

########################################################################
###### M (should be 2**n) is the length of the the "super"  ############
###### lightcurve from which R lightcurves are divided into ############ 
###### using the the Timmer-Kroenig algorithm. M should  ###############  
###### depend on R(=200 when I use) and typical number   ###############  
###### of data points in each lightcurve. A large value ################
###### like M=2097152 can be used but that can make     ################
###### things inefficient. I used trial&error to narrow ################
###### to M=524288 for the lightcurves in my study.     ################
########################################################################

#M=65536
#M=131072
#M=262144
M=524288
#M=1048576
#M=2097152
R=200

######################################################
###### Some important lists and variables ############
##### chi-stores chi-square values           #########
##### hdr-stores best-fit parameters         #########
##### guess-parameter space for searching    #########
#####       best PSD normalization           #########
##### pl_ind-parameter search for searching  #########
#####        best fit PSD slope              #########
##### sucssf-stores success fractions        #########
##### max_ind-stores index of best fit slope #########
######################################################

chi=np.zeros((NoF,R+1))
hdr=np.empty((NoF,6))
guess=np.arange(-6,1,0.007)
J=len(guess)
pl_ind=np.arange(1.0,3.0,0.05)
K=len(pl_ind)
sucssf=np.zeros((NoF,K))
max_ind=np.zeros(NoF)

def least_div(X,N):
    V=np.zeros(N-1)
    for i in range(N-1): V[i]=X[i+1]-X[i]
    dt=min(V)
    nabla=(np.ceil(time[-1]) - np.floor(time[0])) /dt
    return int(np.ceil(nabla)),dt
    
########################################################
###### Calculation of PSD for SIMULATED     ############
###### lightcurve without rms normalization ############
##### X-input light curve            #######
##### T-corresponding time array     #######
##### F-frequency array              #######
#####        RETURNS:                #######
##### Y-PSD for X                    #######
############################################

def powspec(X,T,F):
        L = len(X)
        O = len(F)
        Y = np.zeros(O)
               
        code='''
        #include <math.h>
      
        int i,k,j;
        float Q,R,S,pi2;
        double mean,nmu,normlz;
        pi2=2*M_PI;
        S=0;

        for(i=0;i<L;i++) S=S+X[i];
        mean=S/L;
        for(i=0;i<L;i++) X[i]=X[i] - mean;
        nmu=L*L;
        normlz=2.0*(T[L-1]-T[0])/nmu;

        for(k=0;k<O;k++) {
            Q=0.0;
            R=0.0;
            for(j=0;j<L;j++) {

                Q=Q+ X[j] * cos(pi2*F[k]*T[j]);
                R=R+ X[j] * sin(pi2*F[k]*T[j]); 
                           
            }
            Y[k]= (Q*Q + R*R)*normlz;
          }
        '''
        inline(code,['X','T','F','Y','L','O'])
        return Y
        
########################################################
###### Calculation of PSD for OBSERVED      ############
###### lightcurve with rms normalization ############
##### X-input lightcurve         ########
##### T-corresponding time array ########
##### F-frequency array          ########
#####        RETURNS:            ########
##### Y-PSD for X                ########
#########################################

def powspec_obs(X,T,F):
        L = len(X)
        O = len(F)
        Y = np.zeros(O)
               
        code='''
        #include <math.h>
      
        int i,k,j;
        float Q,R,S,pi2;
        double mean,nmu,normlz;
        pi2=2*M_PI;
        S=0;

        for(i=0;i<L;i++) S=S+X[i];
        mean=S/L;
        for(i=0;i<L;i++) X[i]=X[i] - mean;
        nmu=L*L*mean*mean;
        normlz=2.0*(T[L-1]-T[0])/nmu;

        for(k=0;k<O;k++) {
            Q=0.0;
            R=0.0;
            for(j=0;j<L;j++) {

                Q=Q+ X[j] * cos(pi2*F[k]*T[j]);
                R=R+ X[j] * sin(pi2*F[k]*T[j]); 
                           
            }
            Y[k]= (Q*Q + R*R)*normlz;
          }
        '''
        inline(code,['X','T','F','Y','L','O'])
        return Y

###############################################################
########Generates shape of simulated PSD#######################
##### x-frequency                    ########
##### a-simple power law slope       ########

#####        RETURNS:         #######
##### S-PSD at frequency x    #######
#####################################
'''
def ps_shape(x,a):
   if x< 3.2e-6:  
     return    (3.2e-6)**(-a[0])
   if x>=3.2e-6:  
     return    x**(-a[0])
'''
def ps_shape(x,a):
    c=1e-6
    return (x**(-1.0)) /(1 + (x/c)**(-1.0+a[0]))


###################################################################################
#########Bins and interpolates between gaps of the observed light curve ########### 
##### X-observed light curve             ######## 
##### X-corresponding time array         ######## 
##### Q-new time interval for resampling ########
############       RETURNS:      #############
########### U-masking array         ##########
########### V-resampled light curve ##########
########### Y-time intervals for Y  ##########
########### L- size of U,V,Y        ##########
##############################################     

def bin_intp(X,T,dt):
      Y=np.arange(np.floor(T[0]),np.ceil(T[-1]),dt)
      L=len(Y)
      U=np.zeros(L)
      V=np.zeros(L)

      j=0
      for i in range(L-1):
           s=0
           c=0           
           while T[j]>=Y[i] and T[j]<Y[i+1]:
                   s+=X[j]
                   c+=1.0
                   j+=1
           if c!=0: 
              V[i]=s/c
              U[i]=1
      
      V[-1]=X[-1]
      U[-1]=1
      
      
      for i in range(1,L-1):
          a=V[i-1]
          if U[i]==0:
              k=1
              while (i+k)<L:
                   if U[i+k]!=0:
                       b=V[i+k]
                       break
                   k+=1
              m=(b-a)/(k+2)
              for l in range(i,i+k):  V[l]=a+m*(l-i+1)
              i=k
     
      Y+=0.5*dt
      '''
      plt.plot(T,X)
      plt.scatter(Y,V,color='black')
      plt.show()
      '''
      return L,U,V,Y
      
###################################################################################
#########Bins and interpolates simulated lightcurves in the same scheme ########### 
########### V-input simulated lightcurve ##########
########### U-masking array              ##########
############       RETURNS:           #############
########### V-resampled lightcurve       ##########
###################################################

def resample(V,U):
      L=len(V)
      code2='''
      int i,k,l;
      double a,b,m;

      for(i=1;i<(L-1);i++) {
          a=V[i-1];
          if(U[i]==0) {
              k=1;
              while((i+k)<L) {
                   if(U[i+k]!=0) {
                       b=V[i+k];
                       break;
                      }
                   k=k+1;
                }
              m=(b-a)/(k+2.0);
              for(l=i;l<(i+k);l++)  V[l]=a+m*(l-i+1);
              i+=k;
            }
        }
      '''
      inline(code2,['V','U','L'])
      return V


#####################################################################
#########Generates R simulated light curves of given shape###########
#########using the Timmer-Kroenig algorithm               ###########
##### g-power law slope                                      ######## 
##### FG-frequency array                                     ######## 
##### U-masking array                                        ######## 
#####        RETURNS:                                        ########
##### V-R simulated light curves in the form of RxP matrix   ########
#####################################################################

rndraws=npr.normal(0,1,M)
def timmkron(g,FG,U):
        L=int(M/2)
        norm=L*np.sqrt(FG[0])
        Z=np.zeros(L+1,dtype=complex)
        V=np.zeros((R,Q))

        for i in range(1,L):
            Z[i]=norm * np.sqrt(ps_shape(FG[i-1],g)) * (rndraws[i]+1j*rndraws[M-i])
            
        Z[0]= 0+0j
        Z[L]= norm * np.sqrt(ps_shape(FG[L-1],g)) * (rndraws[0]+0j)

        W=nft.irfft(Z,n=M) 
        
        p0=0
        p1=Q+50
        '''
        plt.plot(range(M),W)
        plt.show()
        '''
        for k in range(R):
            V[k,:]=resample(W[p0:p0+Q],U)
            p0+=p1
        return V


#####################################################################
#########Bins observed PSD- calculates mean and rms      ############ 
##### data-observed PSD                                      ######## 
##### ln2lg-array to go from linear to log bins of frequency ######## 
##### B-number of bins                                       ######## 
#####        RETURNS:                                        ########
##### param-mean and rms of PSD in each frequency bin        ########
#####################################################################

def binmeanstd_1D(data,ln2lg,B):
  param=np.zeros((2,B))
    
  code1='''
  #include <math.h>
  #include <iostream>
  using namespace std;

  int i,j,k;
  double x,z,nu;
  double l[B],m[B];

  for(j=0;j<B;j++) {
      l[j]=1.0/(1 + ln2lg[2*j+1] - ln2lg[2*j]); 
      m[j]=1.0/(ln2lg[2*j+1] - ln2lg[2*j]);

      //cout<<ln2lg[2*j]<<","<<ln2lg[2*j+1]<<endl; 
   }

   i=0;
   for(j=0;j<B;j++) {
         x=0.0;
         
         while(i>=ln2lg[2*j] && i<=ln2lg[2*j+1]) { 
               x=x+log10(data[i]); 
               i=i+1;
               }
         param[j]=(x * l[j]) + 0.253;

       }
   i=0;
   for(j=0;j<B;j++) {
         x=0.0;
         while(i>=ln2lg[2*j] && i<=ln2lg[2*j+1]) { 
               z=log10(data[i])-param[j];
               x=x+z*z; 
               i=i+1;
               }
         //param[B+j]= m[j] * x;
         param[B+j]=sqrt(param[B+j]*l[j]);     
    }
  '''
  inline(code1,['data','param','ln2lg','B'])

  return param 
  
#####################################################################
#########Bins simulated PSDs- calculates mean and rms      ##########
##### data-R simulated PSD                                   ######## 
##### ln2lg-array to go from linear to log bins of frequency ########
##### P-number of frequency bins (linear)                    ########  
##### B-number of bins                                       ######## 
#####        RETURNS:                                        ########
##### param-mean and rms of PSD in each frequency bin        ########
#####################################################################  

def binmeanstd_2D(data,ln2lg,P,B):
  param=np.zeros((2,B))
  mean_bin=np.zeros((R,B))
    
  code1='''
  #include <math.h>

  int i,j,k;
  double x,y,nu,mu;
  double l[B];
  nu=1.0/R;
  mu=1.0/(R-1);

  for(j=0;j<B;j++) {
      l[j]=1.0/(1 + ln2lg[2*j+1] - ln2lg[2*j]);  
   }

  for(k=1;k<(R+1);k++) {
      i=0;
      for(j=0;j<B;j++) {
         x=0.0;
         while(i>=ln2lg[2*j] && i<=ln2lg[2*j+1]) { 
               x=x+log10(data[k*P+i]); 
               i=i+1;
               }
         mean_bin[(k-1)*B+j]=(x * l[j])+0.253;
       }     
     }
   
   for(j=0;j<B;j++) {
          for(k=0;k<R;k++) { 
               param[j]=param[j]+mean_bin[k*B+j]; 
           }
          param[j]=nu*param[j];
       }

    for(j=0;j<B;j++) {
          for(k=0;k<R;k++) {
               y=mean_bin[k*B+j]-param[j];
               param[B+j]=param[B+j]+(y*y); 
          }
          param[B+j]=sqrt(nu * param[B+j]);
       }

    '''
  inline(code1,['data','mean_bin','param','ln2lg','P','R','B'])

  return mean_bin,param      

fig,ax=plt.subplots()
#ax.tick_params(direction='in', length=10, width=2.5,labelsize=15)

##################################################
##### Iteration over the NoF lightcurves #########
##################################################

for n in range(NoF):
    filename=objct+'/'+paths[n]
    print "Filename:",paths[n]

    time0=np.genfromtxt(filename,usecols=0)
    time=(time0-time0[0])
    flux=np.genfromtxt(filename,usecols=1)
    error=np.genfromtxt(filename,usecols=2)
    N=len(time)

    P,del_t=least_div(time,N)

    total_t=time[-1] - time[0]    
    fmin=1.0/total_t
    fmax=P/(2*total_t)
    
    print "Data size:",N,"Time(in s)",total_t
    print "Smallest sampling interval(in s)",del_t,"for which the data size is:",P
    print "-------------------------------------------------------------------------------"
    print " " 
    
##################################################################################
##### Resamples observed lightcurve given new sampling interval del_t ############
##################################################################################

    del_t=120
    Q,mask,flux_new,time_new=bin_intp(flux,time,del_t)

    if Q%2==0:
       P=Q/2
    else:
       P=(Q-1)/2

    total_t=time_new[-1] - time_new[0]    
    fmin=1.0/total_t
    fmax=(1.0*P)/total_t

    print "New binned data size:",Q,",New sampling interval(in s)",del_t
    print "Min frequency(in Hz):",fmin,",Max frequency(in Hz):",fmax
    
    
    fgmin=1.0/(M*del_t)
    fgmax=1.0/(2*del_t)
    #fgmin=fmin
    #fgmax=(M/2)*fmin

    print "Min frequency(in Hz):",fgmin,",Max frequency(in Hz):",fgmax
    print "-------------------------------------------------------------------------------"
    print "                            "
    
    freq_gen=np.linspace(fgmin,fgmax,int(M/2))    
    freq=np.linspace(fmin,fmax,P)
    freq_log=np.arange(np.log10(fmin),np.log10(fmax),0.35)
    bigB=len(freq_log)
    B=bigB-1    
    psd=np.zeros((R+1,P))
    
    
###################################################################
######### Creating an array for linear to log binning #############
###################################################################

    ln2lg=[]
    f0=0
    f1=1
    c=1
    while c<bigB:
           while freq[f1+1]<(10**(freq_log[c])):    f1+=1
           ln2lg.append([f0,f1])
           c+=1
           f0=f1+1
           f1=f0+1
    ln2lg=np.array(ln2lg)
    #print ln2lg

###################################################################
################# Calculation of Poisson noise ####################
###################################################################
    
    mu_sq=(np.mean(flux))**2 
    poisson_noise=np.mean(np.square(error))/(fmax-fmin)
    poisson_noise_norm=poisson_noise/mu_sq 
    print "Normalised poisson noise:",poisson_noise_norm


###################################################################

    psd[0,:]=powspec_obs(flux_new,time_new,freq)
    stat0=binmeanstd_1D(psd,ln2lg,B) 
    
    
    #for i in range(bigB): plt.axvline(freq_log[i],color='black') 
    #plt.plot(np.log10(freq),np.log10(psd[0,:]),marker='o')
    #plt.errorbar(freq_log[:-1],stat0[0,:],yerr=stat0[1,:],linestyle='--',marker='o')
    #plt.show()
    
    freq_log+=0.5*(freq_log[1]-freq_log[0])
    
###################################################################
##### Finding the fit parameters for each power law PSD slope #####
###################################################################

    for g in range(K):
        lc_sim=timmkron([pl_ind[g]],freq_gen,mask)
     
        for k in range(1,R+1):  
            psd[k,:]=powspec(lc_sim[k-1,:],time_new,freq)
            #plt.scatter(np.log10(freq),np.log10(psd[k,:]),marker='.',color='black')
    
        psd_log,stat=binmeanstd_2D(psd,ln2lg,P,B)
        
        #for k in range(R): plt.plot(freq_log[:-1],psd_log[k,:],color='gray')        
        #plt.errorbar(freq_log[:-1],stat[0,:],yerr=stat[1,:],marker='o',linestyle='none')
        #plt.show()
        
        chisqtor=1e8
        chiguess=np.zeros(J)

        for j in range(J):        
          #chiguess[j]=np.sum(np.square((stat[0,:] + guess[j] - stat0[0,:])/stat[1,:]))
          chiguess[j]=np.sum(np.square((np.log10(np.power(10,(stat[0,:] + guess[j])) + poisson_noise_norm) - stat0[0,:] )/stat[1,:]))      
          chiguess[j]=chiguess[j]/B 
        
        chi[n,0]=np.amin(chiguess)
        normbest=guess[np.argmin(chiguess)]
        
        #plt.errorbar(freq_log[:-1],stat0[0,:],yerr=stat0[1,:],linestyle='--',marker='o')
        #plt.errorbar(freq_log[:-1],stat[0,:]+normbest,yerr=stat[1,:])
        #plt.show()
        
        #plt.plot(guess,chiguess)
        #plt.show()
        
               
        for k in range(R):              
           chi[n,k+1]=np.sum(np.square((psd_log[k,:] - stat[0,:])/stat[1,:]))
           chi[n,k+1]=chi[n,k+1]/B

        #plt.errorbar(freq_log[:-1],stat0[0,:],yerr=stat0[1,:],linestyle='--',marker='o')
        #plt.errorbar(freq_log[:-1],np.log10(np.power(10,(stat[0,:] + normbest)) + poisson_noise_norm),yerr=stat[1,:],linestyle='none',marker='o')
        #plt.show()  
    
        m=0
        for k in range(1,R+1):   
             if chi[n,0]<chi[n,k]: m+=1
        #print chi[n,0],m
        sucssf[n,g]=(1.0*m)/R
        print pl_ind[g],normbest,sucssf[n,g]
        if sucssf[n,g]==np.amax(sucssf[n,:]):
              max_ind[n]=pl_ind[g]
              max_mean=np.log10(np.power(10,(stat[0,:] + normbest)) + poisson_noise_norm)
              max_std=stat[1,:]
              max_chi=chiguess
              
###################################################################
########## Saving best fit parameters and plotting ################
###################################################################
     
    hdr[n,:]=fmin,fmax,max_ind[n],np.amax(sucssf[n,:]),chi[n,0],poisson_noise_norm
    toprint=np.column_stack((freq_log[:-1],stat0[0,:],max_mean,max_std))
    np.savetxt(objct+'/psd'+str(n),toprint)

    toprint=np.column_stack((guess,max_chi))
    np.savetxt(objct+'/chinorm'+str(n),toprint)

    ax.set_ylabel('log(PSD)',fontsize=16)
    ax.set_xlabel('log Frequency(Hz) ',fontsize=16)
    ax.errorbar(freq_log[:-1],max_mean,yerr=max_std,linestyle='none',marker='o',markersize=8,markeredgecolor=c1[n],markerfacecolor=c1[n],ecolor=c1[n])
    ax.plot(freq_log[:-1],stat0[0,:],label=lbl[n],linestyle='--',linewidth=2,color=c2[n])
    #ax.axhline(np.log10(poisson_noise_norm),linestyle='-.',color=c2[n])
plt.legend(loc=1,fontsize=16)
plt.show()

toprint=np.vstack((pl_ind,sucssf))
np.savetxt(objct+'/succf',np.transpose(toprint))

fig,ax=plt.subplots()
ax.tick_params(direction='in', length=10, width=2.5,labelsize=15)
ax.set_ylabel('Success fraction',fontsize=16)
ax.set_xlabel('Power law slope',fontsize=16)
for n in range(NoF):   
    plt.plot(pl_ind,sucssf[n,:],linewidth=3,label=lbl[n],color=c1[n])
    plt.axvline(max_ind[n],linewidth=2,linestyle='--',color=c2[n])
plt.legend(loc=1,fontsize=16)
plt.show()
#np.savetxt(objct+'details',np.transpose(hdr))

