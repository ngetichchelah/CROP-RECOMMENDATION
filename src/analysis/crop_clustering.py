"""
Crop Clustering Analysis - Discover crop groups and similarities
Unsupervised learning to complement classification system
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
#since K-Means uses Euclidean distance - features with large ranges 
#thus need for standard scaling...
from scipy.cluster.hierarchy import dendrogram, linkage
# dendrogram: Visual tree showing crop similarities
#linkage: Calculate hierarchical clustering relationships

from sklearn.decomposition import PCA
# PCA: Principal Component Analysis - reduce the 7 dimensions to 2 for plotting
#since we can't plot 7D data directly, PCA finds 2 most important dimensions for a 2D plot...

from sklearn.metrics import silhouette_score  
# silhouette_score: Measure clustering quality (0-1, higher = better)
# Used to validate our choice of K ()=no of clusters) 

import warnings
warnings.filterwarnings('ignore')

class CropClusterAnalyzer:
    """Analyze crop groupings using unsupervised learning"""
    
    def __init__(self, data_path='data/processed/crop_requirements_summary.csv'):
        """Initialize with crop requirements data"""
        self.data_path = data_path
        self.crop_data = None
        self.scaler = StandardScaler()
        self.X_scaled = None
        self.optimal_k = None
        
    def load_data(self):
        """Load crop requirements"""
        print()
        print("LOADING CROP REQUIREMENTS DATA")
        print()
        # Load data
        self.crop_data = pd.read_csv(self.data_path)
        
        # Use average values for clustering
        #Using only averages reduces dimensionality (7 features &not 28)
        features = ['N_avg', 'P_avg', 'K_avg', 'temp_avg', 
                    'humidity_avg', 'ph_avg', 'rainfall_avg']
        
        # Check if data has 'label' or 'crop' column
        crop_col = 'label' if 'label' in self.crop_data.columns else 'crop'
        
        self.crop_names = self.crop_data[crop_col].values
        # .values converts pandas Series to numpy array
        # We need crop names for labeling plots
        
        #Each row = one crop, each column = one feature
        X = self.crop_data[features].values
        
        # Standardize to have all features with equal weight
        self.X_scaled = self.scaler.fit_transform(X)
        
        print(f"Loaded {len(self.crop_names)} crops")
        print(f"Features: {features}")
        print()
        
        return self.X_scaled
    
    def find_optimal_clusters(self, max_k=10):
        """Use elbow method to find optimal number of clusters"""
        print()
        print("FINDING OPTIMAL NUMBER OF CLUSTERS")
        print()
        
        #inertia - Within-Cluster Sum of Squares
        #measures how close each data point in a cluster is to its cluster’s center
        #Lower inertia = tighter, more compact clusters
        inertias = []
        
        #Measures how well each sample fits within its cluster compared to others
        #Higher silhouette score = better-defined clusters
        silhouette_scores = []
        
        # Range of K values to test (2 to max_k)
        # K=1 is meaningless (all crops in one group)
        K_range = range(2, max_k + 1)
        
        from sklearn.metrics import silhouette_score
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            #Run algorithm 10 times with different initializations and keeps best 
            
            kmeans.fit(self.X_scaled)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(self.X_scaled, kmeans.labels_))
        
        # Plot elbow curve
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Elbow plot
        axes[0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
        axes[0].set_xlabel('Number of Clusters (K)', fontsize=12)
        axes[0].set_ylabel('Inertia (Within-cluster sum of squares)', fontsize=12)
        axes[0].set_title('Elbow Method for Optimal K', fontsize=14, fontweight='bold')
        axes[0].grid(alpha=0.3)
        
        # Silhouette plot
        axes[1].plot(K_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
        axes[1].set_xlabel('Number of Clusters (K)', fontsize=12)
        axes[1].set_ylabel('Silhouette Score', fontsize=12)
        axes[1].set_title('Silhouette Score by K', fontsize=14, fontweight='bold')
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/figures/optimal_clusters.png', dpi=300, bbox_inches='tight')
        print(" Elbow plot saved: results/figures/optimal_clusters.png")
        plt.show()
        
        # Recommend K (use elbow heuristic)
        # Calculate rate of change
        changes = np.diff(inertias)
        rate_changes = np.diff(changes)
        
        
        # Find elbow (maximum second derivative)
        elbow_idx = np.argmax(rate_changes) + 2  # +2 because of double diff
        #np.diff reduces array length by 1 and we did it twice
        self.optimal_k = K_range[min(elbow_idx, len(K_range)-1)]
        
        # hard code from domain knowledge (5 groups make agricultural sense)
        # 1. Nitrogen-fixers (legumes)
        # 2. Heavy feeders (cash crops)
        # 3. Water-intensive (tropical)
        # 4. Drought-tolerant (arid)
        # 5. Balanced (cereals/fruits)
        self.optimal_k = 5
        
        print(f"\n Recommended K: {self.optimal_k}")
        print(f"   (Based on agricultural domain knowledge)")
        print()
        
        return self.optimal_k
    
    def perform_kmeans(self, n_clusters=None):
        """Perform K-Means clustering
        K-MEANS ALGORITHM (Simple Explanation):
        1. Randomly place K cluster centers in feature space
        2. Assign each crop to nearest center (Euclidean distance)
        3. Move centers to average position of assigned crops
        4. Repeat steps 2-3 until centers stop moving (convergence)
        5. Result: K clusters with crops grouped by similarity
        """
        
        if n_clusters is None:
            n_clusters = self.optimal_k or 5
        
        print()
        print(f"PERFORMING K-MEANS CLUSTERING (K={n_clusters})")
        print()
        
        # Fit K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(self.X_scaled)
        
        # Add to dataframe
        crop_col = 'label' if 'label' in self.crop_data.columns else 'crop'
        # Add cluster assignments to DataFrame
        self.crop_data['cluster'] = clusters
        
        # Analyze clusters
        print("\nCluster Composition:")
        print()
        for i in range(n_clusters):
            cluster_crops = self.crop_data[self.crop_data['cluster'] == i][crop_col].tolist()
            print(f"\nCluster {i+1}: {len(cluster_crops)} crops")
            print(f"  Crops: {', '.join(cluster_crops)}")
            
            # Calculate cluster characteristics
            cluster_data = self.crop_data[self.crop_data['cluster'] == i]
            print(f"  Avg N: {cluster_data['N_avg'].mean():.1f} kg/ha")
            print(f"  Avg Rainfall: {cluster_data['rainfall_avg'].mean():.1f} mm")
            print(f"  Avg Temp: {cluster_data['temp_avg'].mean():.1f}°C")
        
        print()
        
        # Save clustered data
        self.crop_data.to_csv('results/crop_clusters.csv', index=False)
        print("Clustered data saved: results/crop_clusters.csv")
        
        return clusters, kmeans
    
    # after perform_kmeans() method

    def perform_dbscan(self):
        """
        DBSCAN clustering - finds outliers and arbitrary-shaped clusters
        MINIMAL VERSION - just core functionality
        """
        from sklearn.cluster import DBSCAN
        
        print()
        print("DBSCAN CLUSTERING - OUTLIER DETECTION")
        print()
        
        # DBSCAN with reasonable parameters
        dbscan = DBSCAN(eps=1.5, min_samples=2)
        clusters = dbscan.fit_predict(self.X_scaled)
        
        # Identify outliers (cluster = -1)
        outliers = self.crop_names[clusters == -1]
        n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        
        print(f"\nDBSCAN Results:")
        print(f"  Clusters found: {n_clusters}")
        print(f"  Outliers detected: {len(outliers)}")
        
        if len(outliers) > 0:
            print(f"\n🔍 OUTLIER CROPS (Unique requirements):")
            for crop in outliers:
                print(f"  - {crop}")
            # print("\n💡 These crops have unique requirements not shared by others")
            # print("   (e.g., coffee needs very specific conditions)")
        
        return clusters, outliers

    def plot_dbscan_visualization(self, clusters):
        """Visualize DBSCAN clusters using PCA projection"""
        print()
        print("CREATING DBSCAN CLUSTER VISUALIZATION")
        print()
        
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # PCA to 2D
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X_scaled)
        
        # Identify outliers
        outlier_mask = clusters == -1
        
        plt.figure(figsize=(10, 6))
        
        # Plot clusters (excluding outliers)
        sns.scatterplot(
            x=X_pca[~outlier_mask, 0], 
            y=X_pca[~outlier_mask, 1],
            hue=clusters[~outlier_mask],
            palette='viridis', 
            s=150, edgecolor='black', alpha=0.8
        )
        
        # Plot outliers (in red)
        sns.scatterplot(
            x=X_pca[outlier_mask, 0], 
            y=X_pca[outlier_mask, 1],
            color='red', s=200, marker='X', label='Outliers'
        )
        
        plt.title('DBSCAN Crop Clusters (PCA Projection)', fontsize=14, fontweight='bold')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('results/figures/dbscan_clusters.png', dpi=300, bbox_inches='tight')
        print("✅ DBSCAN visualization saved: results/figures/dbscan_clusters.png")
        plt.show()

    def plot_dendrogram(self):
        """
        Create hierarchical clustering dendrogram
        MINIMAL VERSION - just the tree diagram
        """
        from scipy.cluster.hierarchy import dendrogram, linkage
        import matplotlib.pyplot as plt
        
        print()
        print("CREATING HIERARCHICAL CLUSTERING DENDROGRAM")
        print()
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(self.X_scaled, method='ward')
        
        # Create dendrogram
        plt.figure(figsize=(16, 8))
        dendrogram(
            linkage_matrix,
            labels=self.crop_names,
            leaf_rotation=90,
            leaf_font_size=11,
            color_threshold=7
        )
        plt.title('Crop Similarity Dendrogram (Hierarchical Clustering)', 
                fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Crop', fontsize=12)
        plt.ylabel('Distance (Ward Linkage)', fontsize=12)
        plt.axhline(y=7, color='red', linestyle='--', linewidth=2, 
                label='Cut Height (5 clusters)')
        plt.legend()
        plt.tight_layout() 
        plt.savefig('results/figures/crop_dendrogram.png', dpi=300, bbox_inches='tight')
        print("✅ Dendrogram saved: results/figures/crop_dendrogram.png")
        plt.show()
        
        print("\n💡 Interpretation:")
        print("   - Crops close together = similar requirements")
        print("   - Height indicates dissimilarity")
        print("   - Cutting at red line produces 5 groups")


    def analyze_pca_loadings(self):
        """
        Show which original features contribute most to principal components
        MINIMAL VERSION - just the basics
        """
        from sklearn.decomposition import PCA
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        print("\n" + "="*60)
        print("PCA LOADINGS ANALYSIS")
        print("="*60)
        
        # Fit PCA
        pca = PCA(n_components=7)  # All components
        pca.fit(self.X_scaled)
        
        # Feature names
        features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        
        # Loadings (contribution of each feature to PCs)
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(7)],
            index=features
        )
        
        print("\nPCA Loadings (first 3 components):")
        print(loadings[['PC1', 'PC2', 'PC3']].round(3))
        
        # Variance explained
        print(f"\nVariance explained:")
        for i, var in enumerate(pca.explained_variance_ratio_[:3]):
            print(f"  PC{i+1}: {var*100:.1f}%")
        
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        print(f"  PC1-PC3: {cumvar[2]*100:.1f}% (cumulative)")
        
        # Visualize loadings
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(loadings[['PC1', 'PC2', 'PC3']], 
                    annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                    cbar_kws={'label': 'Loading'})
        ax.set_title('PCA Loadings - Top 3 Components', fontweight='bold', fontsize=14)
        ax.set_xlabel('Principal Component')
        ax.set_ylabel('Original Feature')
        plt.tight_layout()
        plt.savefig('results/figures/pca_loadings.png', dpi=300)
        print("\n✅ Loadings heatmap saved: results/figures/pca_loadings.png")
        plt.show()
        
        return loadings, pca


    def find_similar_crops(self, target_crop, top_n=5):
        """
        Find crops most similar to a target crop
        MINIMAL VERSION - just Euclidean distance
        """
        from sklearn.metrics.pairwise import euclidean_distances
        
        print()
        print(f"FINDING CROPS SIMILAR TO: {target_crop.upper()}")
        print()
        
        crop_col = 'label' if 'label' in self.crop_data.columns else 'crop'
        
        # Find index of target crop
        try:
            target_idx = np.where(self.crop_names == target_crop)[0][0]
        except:
            print(f"❌ Crop '{target_crop}' not found!")
            return None
        
        # Calculate distances
        distances = euclidean_distances(
            self.X_scaled[target_idx].reshape(1, -1), 
            self.X_scaled
        )[0]
        
        # Get top N similar (excluding itself)
        similar_indices = distances.argsort()[1:top_n+1]
        
        print(f"\nTop {top_n} crops similar to {target_crop}:")
        print()
        for rank, idx in enumerate(similar_indices, 1):
            similar_crop = self.crop_names[idx]
            distance = distances[idx]
            print(f"{rank}. {similar_crop:<15} (Distance: {distance:.2f})")
        
        print()
        return similar_indices
    
    #---------------------added for better cluster labels--------------------------
    def assign_cluster_names(self):
        """
    Give clusters meaningful agricultural names instead of numbers
    """
        crop_col = 'label' if 'label' in self.crop_data.columns else 'crop'
    
    # Define cluster characteristics
        cluster_names = {
            0: "Tropical Fruits (High Water)",
            1: "Nitrogen-Fixing Legumes",
            2: "Balanced Cereals & Fruits",
            3: "Heavy Feeders (High P/K)",
            4: "High-Nitrogen Cash Crops"
        }
        
        self.crop_data['cluster_name'] = self.crop_data['cluster'].map(cluster_names)
        
        print()
        print("CLUSTER NAMING")
        print()
        
        for cluster_id, name in cluster_names.items():
            crops = self.crop_data[self.crop_data['cluster'] == cluster_id][crop_col].tolist()
            print(f"\n{name}:")
            print(f"  {', '.join(crops)}")
        print()
    
    
    def plot_cluster_visualization(self, clusters):
        """Visualize clusters using PCA"""
        print()
        print("CREATING CLUSTER VISUALIZATIONS")
        print()
        
        # PCA for 2D visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X_scaled)
        
        # Create plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Scatter plot
        crop_col = 'label' if 'label' in self.crop_data.columns else 'crop'
        scatter = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], 
                                 c=clusters, cmap='viridis', 
                                 s=200, alpha=0.7, edgecolors='black')
        
        # Add crop labels
        for i, crop in enumerate(self.crop_names):
            axes[0].annotate(crop, (X_pca[i, 0], X_pca[i, 1]), 
                           fontsize=9, ha='center', va='bottom')
        
        axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', 
                          fontsize=12)
        axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', 
                          fontsize=12)
        axes[0].set_title('Crop Clusters (PCA Projection)', fontsize=14, fontweight='bold')
        axes[0].grid(alpha=0.3)
        plt.colorbar(scatter, ax=axes[0], label='Cluster')
        
        # Plot 2: Cluster characteristics heatmap
        cluster_features = self.crop_data.groupby('cluster')[
            ['N_avg', 'P_avg', 'K_avg', 'temp_avg', 'humidity_avg', 
             'ph_avg', 'rainfall_avg']
        ].mean()
        
        sns.heatmap(cluster_features.T, annot=True, fmt='.1f', 
                   cmap='YlOrRd', ax=axes[1], cbar_kws={'label': 'Average Value'})
        axes[1].set_xlabel('Cluster', fontsize=12)
        axes[1].set_ylabel('Feature', fontsize=12)
        axes[1].set_title('Cluster Feature Profiles', fontsize=14, fontweight='bold')
        axes[1].set_xticklabels([f'Cluster {i+1}' for i in range(len(cluster_features))])
        
        plt.tight_layout()
        plt.savefig('results/figures/crop_clusters_visualization.png', 
                   dpi=300, bbox_inches='tight')
        print(" Cluster visualization saved: results/figures/crop_clusters_visualization.png")
        plt.show()
    
    
    def plot_dendrogram(self):
        """Enhanced dendrogram with cluster colors"""
        print("CREATING HIERARCHICAL CLUSTERING DENDROGRAM")
    
        from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
        
        linkage_matrix = linkage(self.X_scaled, method='ward')
        
        # Create figure
        plt.figure(figsize=(18, 10))
        
        # Plot dendrogram with colors
        dendro = dendrogram(
            linkage_matrix,
            labels=self.crop_names,
            leaf_rotation=90,
            leaf_font_size=12,
            color_threshold=7,
            above_threshold_color='gray'
        )
        
        # Add cluster labels at bottom
        crop_col = 'label' if 'label' in self.crop_data.columns else 'crop'
        ax = plt.gca()
        xlbls = ax.get_xmajorticklabels()
        
        # Color labels by cluster
        cluster_colors = ['purple', 'blue', 'teal', 'orange', 'green']
        for lbl in xlbls:
            crop_name = lbl.get_text()
            if crop_name in self.crop_data[crop_col].values:
                cluster = self.crop_data[self.crop_data[crop_col] == crop_name]['cluster'].values[0]
                lbl.set_color(cluster_colors[cluster])
        
        plt.title('Crop Similarity Dendrogram (Ward Linkage)', 
                fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('Crop (colored by K-Means cluster)', fontsize=14)
        plt.ylabel('Distance', fontsize=14)
        plt.axhline(y=7, color='red', linestyle='--', linewidth=2, 
                label='Cut Height (5 clusters)', alpha=0.7)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig('results/figures/crop_dendrogram_enhanced.png', dpi=300, bbox_inches='tight')
        print(" dendrogram saved: results/figures/crop_dendrogram.png")
        plt.show()
    
    # def plot_dendrogram(self):
    #     """Create hierarchical clustering dendrogram
    #     - Tree diagram showing how crops are grouped hierarchically
    #     - Bottom: Individual crops (leaves)
    #     - Top: All crops merged into one group (root)
    #     - Height: Dissimilarity between clusters (higher = less similar)
        
    #     HOW IT WORKS:
    #     1. Start with each crop as its own cluster (22 clusters)
    #     2. Find two most similar clusters, merge them
    #     3. Repeat until all crops in one cluster
    #     4. Draw tree showing merge sequence
    #     """
        
    #     print()
    #     print("CREATING HIERARCHICAL CLUSTERING DENDROGRAM")
    #     print()
        
    #     # Perform hierarchical clustering
    #     linkage_matrix = linkage(self.X_scaled, method='ward')
    #     # ward - Minimize within-cluster variance at each merge
        
    #     # Create dendrogram
    #     plt.figure(figsize=(16, 8))
    #     dendrogram(
    #         linkage_matrix,
    #         labels=self.crop_names,
    #         leaf_rotation=90,
    #         leaf_font_size=11,
    #         color_threshold=7
    #     )
    #     plt.title('Crop Similarity Dendrogram (Hierarchical Clustering)', 
    #              fontsize=16, fontweight='bold', pad=20)
    #     plt.xlabel('Crop', fontsize=12)
    #     plt.ylabel('Distance (Ward Linkage)', fontsize=12)
    #     plt.axhline(y=7, color='red', linestyle='--', linewidth=2, 
    #                label='Cut Height (5 clusters)')
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.savefig('results/figures/crop_dendrogram.png', dpi=300, bbox_inches='tight')
    #     print("Dendrogram saved: results/figures/crop_dendrogram.png")
    #     plt.show()
        
    #     print("\n💡 Interpretation:")
    #     print("   - Crops close together have similar requirements")
    #     print("   - Height indicates dissimilarity")
    #     print("   - Cutting at red line produces 5 groups")
    #     print()
    
    def find_similar_crops(self, target_crop, top_n=5):
        """Find crops most similar to target crop"""
        print()
        print(f"FINDING CROPS SIMILAR TO: {target_crop.upper()}")
        print()
        
        crop_col = 'label' if 'label' in self.crop_data.columns else 'crop'
        
        # Get index of target crop
        try:
            target_idx = np.where(self.crop_names == target_crop)[0][0]
        except:
            print(f"Crop '{target_crop}' not found!")
            return None
        
        # Calculate Euclidean distances
        from sklearn.metrics.pairwise import euclidean_distances
        distances = euclidean_distances(self.X_scaled[target_idx].reshape(1, -1), 
                                       self.X_scaled)[0]
        
        # Get top N similar (excluding itself)
        similar_indices = distances.argsort()[1:top_n+1]
        
        print(f"\nTop {top_n} crops similar to {target_crop}:")
        print()
        for rank, idx in enumerate(similar_indices, 1):
            similar_crop = self.crop_names[idx]
            distance = distances[idx]
            
            # Get cluster
            cluster = self.crop_data.iloc[idx]['cluster']
            
            print(f"{rank}. {similar_crop:<15} (Distance: {distance:.2f}, Cluster: {int(cluster)+1})")
        
        print()
        
        return similar_indices
    
    def recommend_crop_rotation(self):
        """Recommend crop rotation sequences based on clusters"""
        print()
        print("CROP ROTATION RECOMMENDATIONS")
        print()
        
        crop_col = 'label' if 'label' in self.crop_data.columns else 'crop'
        
        # Identify nitrogen-fixing crops (legumes - low N requirement)
        legumes = self.crop_data[self.crop_data['N_avg'] < 50][crop_col].tolist()
        
        # Identify heavy feeders (high N requirement)
        heavy_feeders = self.crop_data[self.crop_data['N_avg'] > 80][crop_col].tolist()
        
        print("\n Recommended Rotation Strategy:")
        print()
        print("\n1. NITROGEN-FIXING CROPS (Plant First):")
        print(f"   {', '.join(legumes)}")
        print("   → These crops ADD nitrogen to soil (40-80 kg N/ha)")
        
        print("\n2. HEAVY FEEDERS (Plant Second):")
        print(f"   {', '.join(heavy_feeders)}")
        print("   → These crops USE the nitrogen added by legumes")
        
        print("\n3. MODERATE FEEDERS (Plant Third):")
        moderate = [c for c in self.crop_names if c not in legumes and c not in heavy_feeders]
        print(f"   {', '.join(moderate)}")
        print("   → Balance and prepare soil for next cycle")
        
        # print("\n Example 3-Year Rotation:")
        # print("   Year 1: Chickpea (adds 60 kg N/ha)")
        # print("   Year 2: Cotton (uses high N)")
        # print("   Year 3: Rice (moderate N)")
        # print("   Year 4: Repeat cycle")
        
        # print("\n Benefits:")
        # print("   • Reduces fertilizer costs by 30-50%")
        # print("   • Improves soil health")
        # print("   • Breaks pest/disease cycles")
        # print("   • Increases long-term yields")
        
        print()
    
    #----------------------------3D Interactive added------------------------------------------
    def plot_3d_clusters(self, clusters):
        """
        Create 3D interactive plot using Plotly
        """
        print()
        print("CREATING 3D INTERACTIVE VISUALIZATION")
        print()
        
        import plotly.graph_objects as go
        from sklearn.decomposition import PCA
        
        # PCA to 3D (instead of 2D)
        pca_3d = PCA(n_components=3)
        X_3d = pca_3d.fit_transform(self.X_scaled)
        
        crop_col = 'label' if 'label' in self.crop_data.columns else 'crop'
        
        # Create 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=X_3d[:, 0],
            y=X_3d[:, 1],
            z=X_3d[:, 2],
            mode='markers+text',
            marker=dict(
                size=12,
                color=clusters,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Cluster"),
                line=dict(color='black', width=1)
            ),
            text=self.crop_names,
            textposition="top center",
            textfont=dict(size=10),
            hovertemplate='<b>%{text}</b><br>Cluster: %{marker.color}<extra></extra>'
        )])
        
        fig.update_layout(
            title='3D Crop Clusters (Interactive)',
            scene=dict(
                xaxis_title=f'PC1 ({pca_3d.explained_variance_ratio_[0]*100:.1f}%)',
                yaxis_title=f'PC2 ({pca_3d.explained_variance_ratio_[1]*100:.1f}%)',
                zaxis_title=f'PC3 ({pca_3d.explained_variance_ratio_[2]*100:.1f}%)'
            ),
            width=1000,
            height=800
        )
        
        fig.write_html('results/figures/crop_clusters_3d.html')
        print(" 3D interactive plot saved: results/figures/crop_clusters_3d.html")
        print("   Open in browser to rotate and explore!")
        
        # Show total variance captured
        total_var = sum(pca_3d.explained_variance_ratio_) * 100
        print(f"\n Total variance captured in 3D: {total_var:.1f}%")
        print()
    
    ##---------------added to Cluster Validation Metrics-------------------
    def calculate_clustering_metrics(self, clusters):
        """
        Calculate clustering quality metrics
        """
        print()
        print("CLUSTERING QUALITY METRICS")
        print()
        
        from sklearn.metrics import (silhouette_score, calinski_harabasz_score, 
                                    davies_bouldin_score)
        
        # Silhouette Score (0 to 1, higher is better)
        # Measures how similar crops are to their own cluster vs other clusters
        silhouette = silhouette_score(self.X_scaled, clusters)
        
        # Calinski-Harabasz Index (higher is better)
        # Ratio of between-cluster to within-cluster variance
        calinski = calinski_harabasz_score(self.X_scaled, clusters)
        
        # Davies-Bouldin Index (lower is better)
        # Average similarity ratio of each cluster with its most similar cluster
        davies_bouldin = davies_bouldin_score(self.X_scaled, clusters)
        
        print(f"\n📊 Silhouette Score: {silhouette:.3f}")
        print("   Interpretation:")
        if silhouette > 0.5:
            print("   EXCELLENT - Clusters are well-separated")
        elif silhouette > 0.3:
            print("   GOOD - Reasonable cluster structure")
        elif silhouette > 0.2:
            print("   FAIR - Some overlap between clusters")
        else:
            print("   POOR - Clusters not well-defined")
        
        print(f"\n📊 Calinski-Harabasz Index: {calinski:.2f}")
        print("   Interpretation: Higher = better separation (no absolute threshold)")
        
        print(f"\n📊 Davies-Bouldin Index: {davies_bouldin:.3f}")
        print("   Interpretation:")
        if davies_bouldin < 1.0:
            print("   EXCELLENT - Low cluster overlap")
        elif davies_bouldin < 1.5:
            print("   GOOD - Acceptable separation")
        else:
            print("   FAIR - Some clusters may be too similar")
        
        print()
        
        return {
            'silhouette': silhouette,
            'calinski_harabasz': calinski,
            'davies_bouldin': davies_bouldin
        }
    
    def generate_cluster_insights(self):
        """Generate agricultural insights from clusters"""
        print()
        print("AGRICULTURAL INSIGHTS FROM CLUSTERING")
        print()
        
        crop_col = 'label' if 'label' in self.crop_data.columns else 'crop'
        
        for cluster_id in sorted(self.crop_data['cluster'].unique()):
            cluster_data = self.crop_data[self.crop_data['cluster'] == cluster_id]
            
            # Characterize cluster
            avg_n = cluster_data['N_avg'].mean()
            avg_rain = cluster_data['rainfall_avg'].mean()
            avg_temp = cluster_data['temp_avg'].mean()
            
            # Determine cluster type
            if avg_n < 50:
                cluster_type = "🌱 NITROGEN-FIXING / LOW-N CROPS"
                advice = "Ideal for soil improvement, rotate before heavy feeders"
            elif avg_rain > 200:
                cluster_type = "🌊 HIGH WATER / TROPICAL CROPS"
                advice = "Need irrigation or high rainfall regions, monsoon suitable"
            elif avg_rain < 80:
                cluster_type = "🌵 DROUGHT-TOLERANT CROPS"
                advice = "Perfect for arid regions, low water requirements"
            elif avg_n > 80:
                cluster_type = "🍃 HEAVY FEEDERS / CASH CROPS"
                advice = "High fertilizer needs, plant after legumes for best results"
            else:
                cluster_type = "🌾 BALANCED / CEREAL CROPS"
                advice = "Moderate requirements, versatile for most soils"
            
            print(f"\nCluster {cluster_id + 1}: {cluster_type}")
            print("-"*60)
            crops_in_cluster = cluster_data[crop_col].tolist()
            print(f"Crops: {', '.join(crops_in_cluster)}")
            print(f"Avg Nitrogen: {avg_n:.1f} kg/ha")
            print(f"Avg Rainfall: {avg_rain:.1f} mm")
            print(f"Avg Temperature: {avg_temp:.1f}°C")
            print(f"💡 Advice: {advice}")
        
        print()

    #---------------export cluster ------------------------------------------
    def export_cluster_summary(self):
        """
        Export detailed cluster summary for presentation
        """
        crop_col = 'label' if 'label' in self.crop_data.columns else 'crop'
        
        summary = []
        for cluster_id in sorted(self.crop_data['cluster'].unique()):
            cluster_crops = self.crop_data[self.crop_data['cluster'] == cluster_id]
            
            summary.append({
                'Cluster': cluster_id + 1,
                'Crop_Count': len(cluster_crops),
                'Crops': ', '.join(cluster_crops[crop_col].tolist()),
                'Avg_N': round(cluster_crops['N_avg'].mean(), 1),
                'Avg_P': round(cluster_crops['P_avg'].mean(), 1),
                'Avg_K': round(cluster_crops['K_avg'].mean(), 1),
                'Avg_Rainfall': round(cluster_crops['rainfall_avg'].mean(), 1),
                'Avg_Temp': round(cluster_crops['temp_avg'].mean(), 1),
                'Type': self._characterize_cluster(cluster_crops)
            })
        
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv('results/cluster_summary_table.csv', index=False)
        print("Cluster summary exported: results/cluster_summary_table.csv")
        
        return summary_df

    def _characterize_cluster(self, cluster_data):
        """Helper to characterize cluster type"""
        avg_n = cluster_data['N_avg'].mean()
        avg_rain = cluster_data['rainfall_avg'].mean()
        
        if avg_n < 50:
            return "Nitrogen-Fixing Legumes"
        elif avg_rain > 150:
            return "Water-Intensive Tropicals"
        elif avg_rain < 80:
            return "Drought-Tolerant"
        elif avg_n > 80:
            return "Heavy Feeders"
        else:
            return "Balanced/Versatile"
    
    
def main():
    """Main execution"""
    print()
    print("CROP CLUSTERING ANALYSIS")
    print()
    
    # Initialize
    analyzer = CropClusterAnalyzer()
    
    # Load data
    analyzer.load_data()
    
    # Find optimal K
    analyzer.find_optimal_clusters(max_k=10)
    
    # Perform clustering
    clusters, kmeans = analyzer.perform_kmeans(n_clusters=5)
    
    # After perform_kmeans
    analyzer.assign_cluster_names()
    
    # Visualizations
    analyzer.plot_cluster_visualization(clusters)
    
    # DBSCAN for outlier detection
    #dbscan_clusters, outliers = analyzer.perform_dbscan()
    
    clusters, outliers = analyzer.perform_dbscan()
    analyzer.plot_dbscan_visualization(clusters)

    analyzer.plot_dendrogram()
    
    # After plot_cluster_visualization
    analyzer.plot_3d_clusters(clusters)

    # After perform_kmeans
    metrics = analyzer.calculate_clustering_metrics(clusters)
    
    # Insights
    analyzer.generate_cluster_insights()
    
    # Similarity analysis
    print()
    analyzer.find_similar_crops('rice', top_n=5)
    analyzer.find_similar_crops('chickpea', top_n=5)
    
    # Rotation recommendations
    analyzer.recommend_crop_rotation()
    
    print()
    print("CLUSTERING ANALYSIS COMPLETE!")
    print()
    
    print("Generated Files:")
    print("   - results/crop_clusters.csv")
    print("   - results/figures/optimal_clusters.png")
    print("   - results/figures/crop_clusters_visualization.png")
    print("   - results/figures/crop_dendrogram.png")

if __name__ == "__main__":
    main()