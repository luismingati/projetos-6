import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

def train_kmeans_model(csv_path, output_dir="models"):
    """
    Treina um modelo KMeans com base no dataset fornecido e salva o modelo treinado.
    Args:
        csv_path (str): Caminho para o arquivo CSV com os dados de treinamento
        output_dir (str): Diretório onde o modelo será salvo
    Returns:
        KMeans: Modelo KMeans treinado
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Carregando dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    features = ['cores_vivas', 'versatilidade', 'conforto', 'formalidade', 'estampas']
    X = df[features].values

    k_opt = 5
    print(f"Treinando modelo KMeans com k={k_opt}...")
    kmeans = KMeans(n_clusters=k_opt, random_state=40, n_init=20)
    labels = kmeans.fit_predict(X)

    inertia = kmeans.inertia_
    silhouette = silhouette_score(X, labels)
    print(f"Inércia (k={k_opt}): {inertia:.2f}")
    print(f"Silhouette Score (k={k_opt}): {silhouette:.4f}")

    print(f"Salvando modelo em {output_dir}...")
    with open(os.path.join(output_dir, 'kmeans_model.pkl'), 'wb') as f:
        pickle.dump(kmeans, f)
    print("Modelo salvo com sucesso!")

    return kmeans

def generate_visualizations(X_scaled, labels, output_dir):
    """
    Gera visualizações dos clusters usando PCA e histogramas.
    
    Args:
        X_scaled (numpy.ndarray): Dados normalizados
        labels (numpy.ndarray): Rótulos dos clusters
        output_dir (str): Diretório para salvar as visualizações
    """
    # Diretório para visualizações
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Aplicar PCA para visualização em 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Visualizar clusters no espaço PCA
    plt.figure(figsize=(10, 8))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.8)
    plt.title('Clusters de Usuários (PCA)', fontsize=14)
    plt.xlabel('Componente Principal 1', fontsize=12)
    plt.ylabel('Componente Principal 2', fontsize=12)
    plt.colorbar(label='Cluster')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(viz_dir, 'pca_clusters.png'), dpi=300, bbox_inches='tight')
    
    # Distribuição de usuários por cluster
    plt.figure(figsize=(8, 6))
    unique_labels, counts = np.unique(labels, return_counts=True)
    plt.bar(unique_labels, counts, color=['blue', 'green', 'red'])
    plt.title('Distribuição de Usuários por Cluster', fontsize=14)
    plt.xlabel('Cluster', fontsize=12)
    plt.ylabel('Quantidade de Usuários', fontsize=12)
    plt.xticks(unique_labels)
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(viz_dir, 'cluster_distribution.png'), dpi=300, bbox_inches='tight')
    
    print(f"Visualizações salvas em {viz_dir}")

def predict_profile(features_array, models_dir="models"):
    """
    Classifica um perfil de usuário com base nas features fornecidas.
    
    Args:
        features_array (list or numpy.ndarray): Array com as 5 features do usuário
                                           [cores_vivas, versatilidade, conforto, formalidade, estampas]
        models_dir (str): Diretório contendo os modelos treinados
    
    Returns:
        tuple: (cluster_id, profile_name) O ID do cluster e o nome do perfil correspondente
    """
    # Mapear clusters para perfis
    cluster_map = {
        0: 'Profissional Moderno',
        1: 'Casual Despojado',
        2: 'Aventureiro Fashion',
        3: 'Esportivo Casual',
        4: 'Minimalista Chic',
    }
    
    # Carregar modelos treinados
    try:
        with open(os.path.join(models_dir, 'kmeans_model.pkl'), 'rb') as f:
            kmeans = pickle.load(f)
        
        with open(os.path.join(models_dir, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
    except FileNotFoundError:
        raise ValueError("Modelos não encontrados. Execute o treinamento primeiro.")
    
    # Preparar dados para previsão
    if isinstance(features_array, list):
        features_array = np.array([features_array])
    elif isinstance(features_array, np.ndarray) and features_array.ndim == 1:
        features_array = features_array.reshape(1, -1)
    
    # Normalizar dados
    scaled_features = scaler.transform(features_array)
    
    # Prever cluster
    cluster = kmeans.predict(scaled_features)[0]
    profile = cluster_map.get(cluster)
    
    return cluster, profile

def test_model_with_examples():
    """
    Testa o modelo com algumas personas de exemplo para verificar a consistência.
    """
    # Definir personas de teste
    personas = {
        'Persona X': [2, 4, 3, 5, 1],  # Profissional Moderno
        'Persona Y': [4, 3, 5, 2, 2],  # Casual Despojado
        'Persona Z': [5, 2, 3, 1, 5],  # Aventureiro Fashion
        'Persona W': [4, 5, 4, 1, 3],  # Esportivo Casual
        'Persona V': [1, 3, 2, 4, 1],  # Minimalista Chic
    }
    
    print("\nTestando modelo com personas de exemplo:")
    print("-" * 50)
    
    for name, features in personas.items():
        try:
            cluster, profile = predict_profile(features)
            print(f"{name}: {features} -> Cluster {cluster} ({profile})")
        except Exception as e:
            print(f"Erro ao testar {name}: {e}")
    
    print("-" * 50)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Uso: python train_model.py <caminho_para_csv>")
        sys.exit(1)
    csv_path = sys.argv[1]
    kmeans = train_kmeans_model(csv_path)
    test_model_with_examples()