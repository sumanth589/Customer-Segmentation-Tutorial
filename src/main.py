import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Step 1: Creating a folder for our results so things stay organized
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"I've created a folder for our results here: {output_dir}")

# Step 2: Loading the Credit Card behavior data
# Make sure CC GENERAL.csv is in your /data folder!
print("Loading the customer dataset...")
df = pd.read_csv('data/CC_GENERAL.csv') 

# Step 3: Cleaning the data (An important intermediate step)
# We remove the Customer ID because it's just a random number
df = df.drop('CUST_ID', axis=1)

# Fixing the error: We fill in any missing data points by copying the previous row
# This is called 'forward filling'
df = df.ffill() 

# Step 4: Scaling the data
# K-Means works best when all numbers are on the same scale (like 0 to 1)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Step 5: Finding the 'Elbow' (The experiment part of our tutorial)
# We want to find the perfect number of groups for our customers
print("Running the Elbow Method experiment... this takes a few seconds.")
distortions = []
K_range = range(1, 11)
for k in K_range:
    # We use n_init='auto' to keep the code quiet and modern
    km = KMeans(n_clusters=k, n_init='auto', random_state=42)
    km.fit(df_scaled)
    distortions.append(km.inertia_)

# Plotting the Elbow graph for our report
plt.figure(figsize=(10, 6))
plt.plot(K_range, distortions, 'bx-', color='#1f77b4')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (Error Score)')
plt.title('The Elbow Method: Finding the Mathematical "Sweet Spot"')
plt.grid(True, linestyle=':', alpha=0.6)

plt.savefig(os.path.join(output_dir, 'elbow_plot.png'), dpi=300)
print("Elbow plot saved to the /output folder.")
plt.show()

# Step 6: Simplifying the data with PCA
# Our data has 17 columns. PCA squashes them into 2 columns so we can plot them!
print("Compressing 17 features into 2D using PCA...")
pca = PCA(n_components=2)
principal_components = pca.fit_transform(df_scaled)
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Step 7: Final Clustering
# Looking at our Elbow plot, 4 seems like a good number of groups
print("Finalizing the 4 customer segments...")
kmeans = KMeans(n_clusters=4, n_init='auto', random_state=42)
kmeans.fit(df_scaled)
pca_df['Cluster'] = kmeans.labels_

# Step 8: Visualizing our Final Result (Creative Teaching Tool)
plt.figure(figsize=(10, 8))
colors = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e'] # Red, Blue, Green, Orange
for i in range(4):
    subset = pca_df[pca_df['Cluster'] == i]
    plt.scatter(subset['PC1'], subset['PC2'], c=colors[i], label=f'Group {i+1}', s=25, alpha=0.5)

plt.title('Final Results: Visualizing 4 Customer Segments in 2D Space', fontsize=14)
plt.xlabel('Principal Component 1 (Main Trend)')
plt.ylabel('Principal Component 2 (Secondary Trend)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.3)

plt.savefig(os.path.join(output_dir, 'final_clusters.png'), dpi=300)
print("Final cluster map saved! You are all done.")
plt.show()