import numpy as np
import matplotlib.pyplot as plt

def hist_plot(stats, mode = 'VT', savefig = False, name = ' '):
    all_truth, all_pred,all_bell, gmass, VT_diff = stats
    
    if mode == 'VT':
        gmass_diff = (all_pred-all_truth).numpy()
    if mode == 'galaxy':
        bad = np.argwhere(gmass[:,0]==0).ravel()
        gmass_diff = np.log10(gmass[:,0])-np.log10(gmass[:,1])
        gmass_diff[bad] = 0
    #gmass_diff[bad] = 0
    one_sig = np.percentile(gmass_diff,[16,50,84])
    two_sig = np.percentile(gmass_diff,[2.5,50,97.5])
    print(np.mean(gmass_diff),np.std(gmass_diff))
    print(one_sig)
    print(two_sig)
    plt.hist(gmass_diff,bins=40,density=True, range = [-1,1])
    #sns.kdeplot(gmass_diff,bw_adjust=0.4)
    height  = plt.gca().get_ylim()[1]
    two_sig[two_sig<-1] = -1
    plt.fill_betweenx([0, plt.gca().get_ylim()[1]], two_sig[0], two_sig[2], alpha=0.1, color='purple')
    plt.fill_betweenx([0, plt.gca().get_ylim()[1]], one_sig[0], one_sig[2], alpha=0.2, color='r')
    plt.axvline(one_sig[1], color='r', linestyle='--')
    
    #plt.ylim(0,1.2)
    
    if mode == 'VT':
        plt.xlabel('VT mass Difference')
        plt.title(name+' VT Hist')
    if mode == 'galaxy':
        plt.xlabel('Galaxy mass Difference')
        plt.title(name+' integrated mass Hist')
        
    plt.text(0.3,height,'$\sigma$ = %.3f'% ((one_sig[2]-one_sig[0])/2), fontsize = 12)
    plt.text(0.3,0.93*height,'$\mu$ = %.3f'% (np.mean(gmass_diff)), fontsize = 12)
    plt.xlim(-1,1)
    
    if savefig:
        if mode == 'VT':
            plt.savefig('./Analyses/'+name+'_VT.pdf',dpi=400)
        if mode == 'galaxy':
            plt.savefig('./Analyses/'+name+'_gmass.pdf',dpi=400)
    plt.show()
