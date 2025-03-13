# visualize.py <datafile.csv>
#
# The visualize script creates a 2D PCA plot and 2D UMAP plot of the data in the given
# CSV data file. The resulting plots are stored in files <datafile>-pca.png and
# <datafile>-umap.png. The first field in each line of the CSV file is assumed to be a
# time stamp (yyyy-mm-dd hh:mm:ss.ffffff) and the last field is a string activity label.
# The remaining fields represent the feature vector that is visualized. The colors in the
# plots represent the value of the activity for the corresponding data point.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import umap.umap_ as umap
import sys
import os


def load_data(datafile):
    df = pd.read_csv(datafile)
    df = df.drop(columns=['stamp'])
    df = df.fillna(0.0)

    # Separate features and labels
    X = df.iloc[:, :-1]  # All columns except the last one
    y = df.iloc[:, -1]   # The last column (activity_label)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

# Apply PCA to reduce to 2 components
def generate_pca(X, y, basefile):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Convert PCA results into a DataFrame
    pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    pca_df['activity_label'] = y

    # Remove outliers using percentile filtering
    lower_bound = np.percentile(X_pca, 1, axis=0)
    upper_bound = np.percentile(X_pca, 99, axis=0)
    pca_df = pca_df[(pca_df['PC1'] >= lower_bound[0]) & (pca_df['PC1'] <= upper_bound[0]) &
                     (pca_df['PC2'] >= lower_bound[1]) & (pca_df['PC2'] <= upper_bound[1])]

    # Plot the PCA results
    plt.figure(figsize=(12, 7))
    sns.scatterplot(x='PC1', y='PC2', hue='activity_label', palette='tab10', data=pca_df, alpha=0.7)
    plt.title('PCA Visualization of Activity Data (Outliers Removed)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Activity Label', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(basefile + "-pca.png")
    plt.close

# Apply UMAP to reduce to 2 components
def generate_umap(X, y, basefile):
    umap_reducer = umap.UMAP(n_components=2, random_state=42)
    X_umap = umap_reducer.fit_transform(X)

    # Convert UMAP results into a DataFrame
    umap_df = pd.DataFrame(X_umap, columns=['UMAP1', 'UMAP2'])
    umap_df['activity_label'] = y

    # Plot the UMAP results and save to file
    plt.figure(figsize=(12, 7))
    sns.scatterplot(x='UMAP1', y='UMAP2', hue='activity_label', palette='tab10', data=umap_df, alpha=0.7)
    plt.title('UMAP Visualization of Activity Data')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.legend(title='Activity Label', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(basefile + '-umap.png', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    datafile = sys.argv[1]
    X, y = load_data(datafile)
    basefile = os.path.splitext(datafile)[0]
    generate_pca(X, y, basefile)
    generate_umap(X, y, basefile)
