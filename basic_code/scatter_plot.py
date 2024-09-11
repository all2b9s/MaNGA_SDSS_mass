import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
def scatter_plot(stats, name = ' ', savefig = False):
    all_truth, all_pred,all_bell, gmass, VT_diff = stats
    # Sample data
    plt.rcParams['axes.facecolor'] = 'white' #'#EEF7F2'

    # Create the scatter plot using seaborn
    plt.figure(figsize=(8, 6))  # Adjust the figure size as needed

    # Scatter plots for 'Galaxy mass' and 'VT cell mass'
    sns.scatterplot(x=all_truth,y=all_pred, color='#5698C3', s = 1,alpha=0.1)
    sns.scatterplot(x=np.log10(gmass[:,1]), y=np.log10(gmass[:,0]), color='red', s= 5,alpha=0.5)
    sns.scatterplot(x=[0],y=[0], color='#5698C3', label='VT cell mass', s = 50, alpha= 1)
    sns.scatterplot(x=[0], y=[0], color='red', label='galaxy',s= 50,alpha= 1 )
    #sns.scatterplot(x=np.log10(VT_diff[:, 0]), y=np.log10(VT_diff[:, 1]), color='red', label='Mean VT mass for each galaxy',s= 3,alpha=0.7)
    # Add the y=x line
    x = np.linspace(4,12,20)
    sns.lineplot(x=x, y=x, color='purple', linestyle='-')

    # Add labels and title
    plt.xlim(6,12)
    plt.ylim(6,12)
    plt.xlabel('Truth')
    plt.ylabel('Prediction')
    plt.legend(loc='upper left')
    plt.title(name+' model')

    if savefig:
        plt.savefig('./Analyses/'+name+'_scatter.png',dpi=400)
    # Show the plot
    plt.show()