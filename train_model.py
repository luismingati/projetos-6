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
        output_dir (str): Diretório onde os modelos serão salvos
    
    Returns:
        tuple: Modelo KMeans treinado e StandardScaler
    """
    # Criar diretório para os modelos se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Carregar o dataset
    print(f"Carregando dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Definir as features para o treinamento
    features = ['cores_vivas', 'versatilidade', 'conforto', 'formalidade', 'estampas']
    X = df[features].values
    
    # Normalizar os dados
    print("Normalizando dados...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Treinar o modelo KMeans com k=3 (conforme análise prévia)
    print("Treinando modelo KMeans com k=3...")
    k_opt = 3
    kmeans = KMeans(n_clusters=k_opt, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    # Avaliar o modelo
    inertia = kmeans.inertia_
    silhouette = silhouette_score(X_scaled, labels)
    print(f"Inércia (k=3): {inertia:.2f}")
    print(f"Silhouette Score (k=3): {silhouette:.4f}")
    
    # Salvar os modelos treinados
    print(f"Salvando modelos em {output_dir}...")
    with open(os.path.join(output_dir, 'kmeans_model.pkl'), 'wb') as f:
        pickle.dump(kmeans, f)
    
    with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    print("Modelos salvos com sucesso!")
    
    # Gerar visualizações (opcional)
    if os.environ.get('GENERATE_PLOTS', 'False').lower() == 'true':
        generate_visualizations(X_scaled, labels, output_dir)
    
    return kmeans, scaler

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
    cluster_map = {0: 'Profissional Moderno', 1: 'Casual Despojado', 2: 'Aventureiro Fashion'}
    
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
        'Persona X': [3, 4, 2, 5, 1],  # Alta formalidade (Profissional Moderno)
        'Persona Y': [1, 1, 5, 3, 2],  # Alto conforto (Casual Despojado)
        'Persona Z': [5, 2, 4, 1, 5]   # Cores vivas e estampas (Aventureiro Fashion)
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
    
    # Treinar modelo
    kmeans, scaler = train_kmeans_model(csv_path)
    
    # Testar o modelo com exemplos
    test_model_with_examples()