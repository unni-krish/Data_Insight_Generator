import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestRegressor
import streamlit as st



def plot_advanced_graphs(df, X, y,plot_name,target_column):

    colors = sns.color_palette("Purples", n_colors=len(df.drop(target_column, axis=1).columns))

    plt.figure(figsize=(10, 6))
    
    if plot_name == 'Correlation Matrix Heatmap':
        sns.heatmap(df.corr(), annot=True, cmap='Purples')
        # plt.title('Correlation Matrix Heatmap')
    
    elif plot_name == 'Pair Plot':
        sns.pairplot(df, hue=target_column, palette='Purples')
        # plt.title('Pair Plot')
    
    elif plot_name == 'Box Plot':
        sns.boxplot(data=df, palette='Purples')
        # plt.title('Box Plot')
    
    elif plot_name == 'Violin Plot':
        sns.violinplot(data=df, palette='Purples')
        # plt.title('Violin Plot')
    
    elif plot_name == 'Histogram with KDE':
        for i, column in enumerate(df.drop(target_column, axis=1).columns):
            df[column].plot(kind='hist', bins=30, alpha=0.5, color=colors[i], edgecolor='black', density=True, label=column)
            df[column].plot(kind='kde', color=colors[i])
        # plt.title('Histogram with KDE')
    

    elif plot_name == 'Parallel Coordinates Plot':
        pd.plotting.parallel_coordinates(df, target_column, color=['purple', 'blue'])
        # plt.title('Parallel Coordinates Plot')
    
    elif plot_name == 'PCA Plot':
        pca = PCA(n_components=2)
        components = pca.fit_transform(X)
        plt.scatter(components[:, 0], components[:, 1], c=y, cmap='tab20b')
        # plt.title('PCA Plot')
    
    elif plot_name == 'T-SNE Plot':
        tsne = TSNE(n_components=2)
        tsne_results = tsne.fit_transform(X)
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=y, cmap='tab20b')
        # plt.title('T-SNE Plot')
    
    elif plot_name == 'Cluster Heatmap':
        sns.clustermap(df.corr(), cmap='Purples', annot=True)
        # plt.title('Cluster Heatmap')
    
    
    elif plot_name == 'Residual Plot':
        model = RandomForestRegressor()
        model.fit(X, y)
        preds = model.predict(X)
        residuals = y - preds
        sns.residplot(preds, residuals, lowess=True, color='purple')
        # plt.title('Residual Plot')
    
    
    elif plot_name == 'Cumulative Distribution Function (CDF) Plot':
        sns.ecdfplot(df['Feature1'], color='purple')
        # plt.title('Cumulative Distribution Function (CDF) Plot')

     # Save the plot as an image in the specified folder
    path = 'static/plots'
    plot_filename = os.path.join(path, f"{plot_name}.png")
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.close()
    
    st.toast(f"{plot_name} Successfully Saved",icon="✅")

def Generate_Plots(df, X, y,plot_name,task_type,target_column):

    # model = load_model(model, task_type)
    plot_advanced_graphs(df, X, y, plot_name,target_column)




