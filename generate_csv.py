import os
import numpy as np
import pandas as pd

# Simular dataset de usuários fictícios com resposta de 1 a 5 e label de cluster
np.random.seed(42)
means = {
    0: {'cores_vivas': 2, 'versatilidade': 4, 'conforto': 3, 'formalidade': 5, 'estampas': 1},  # Profissional Moderno
    1: {'cores_vivas': 4, 'versatilidade': 3, 'conforto': 5, 'formalidade': 2, 'estampas': 2},  # Casual Despojado
    2: {'cores_vivas': 5, 'versatilidade': 2, 'conforto': 3, 'formalidade': 1, 'estampas': 5},  # Aventureiro Fashion
    3: {'cores_vivas': 4, 'versatilidade': 5, 'conforto': 4, 'formalidade': 1, 'estampas': 3},  # Esportivo Casual
    4: {'cores_vivas': 1, 'versatilidade': 2, 'conforto': 2, 'formalidade': 3, 'estampas': 1},  # Minimalista Chic
}

rows = []
for cluster, m in means.items():
    for _ in range(12):
        sample = {k: int(np.clip(np.random.normal(loc=v, scale=0.5), 1.0, 5.0)) for k, v in m.items()}
        sample['cluster'] = cluster
        rows.append(sample)

df = pd.DataFrame(rows)
df.insert(0, 'user_id', [f'U{i+1:02d}' for i in range(len(df))])

os.makedirs('sample_data', exist_ok=True)

# Salvar CSV para download
csv_path = 'sample_data/user_style_dataset.csv'
df.to_csv(csv_path, index=False)

# Mostrar tabela ao usuário (usando display padrão do pandas em vez de ace_tools)
print("Dataset de Usuários Estilo:")
print(df)  # Esta função funciona em ambientes notebook como Jupyter ou Colab
# Alternativamente, se display não funcionar no seu ambiente:
# print(df)