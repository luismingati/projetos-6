from fastapi import FastAPI, HTTPException, Depends
import pandas as pd
import sqlite3
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import os
from openai import OpenAI
import logging
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from dotenv import load_dotenv

# Carrega variáveis de ambiente do .env
load_dotenv()

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Clothes API", description="API para recomendação de roupas e classificação de perfil de moda")

# Caminhos para modelos treinados
MODEL_PATH = "models/kmeans_model.pkl"
SCALER_PATH = "models/scaler.pkl"

# Mapeamento dos clusters para perfis de moda
CLUSTER_MAP = {0: 'Profissional Moderno', 1: 'Casual Despojado', 2: 'Aventureiro Fashion'}

# Carregar o dataset de roupas uma vez ao iniciar a aplicação para otimizar
try:
    df = pd.read_parquet(
        "hf://datasets/wbensvage/clothes_desc/data/my_clothes_desc.parquet"
    )
except Exception as e:
    logger.error(f"Erro ao carregar o dataset: {e}")
    df = None  # Define df como None se houver erro

# Função para carregar modelos
def load_models():
    try:
        # Carrega o modelo KMeans treinado
        with open(MODEL_PATH, 'rb') as f:
            kmeans = pickle.load(f)
        
        # Carrega o scaler usado para padronizar os dados
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        
        return kmeans, scaler
    except FileNotFoundError as e:
        logger.error(f"Arquivos de modelo não encontrados: {e}")
        raise HTTPException(
            status_code=500,
            detail="Modelos de classificação não encontrados. Execute o treinamento primeiro."
        )
    except Exception as e:
        logger.error(f"Erro ao carregar modelos: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao carregar modelos de classificação: {str(e)}"
        )

# --- Configuração do Banco de Dados SQLite e OpenAI ---
DATABASE_URL = "clothes_ids.db"

# Validação e inicialização obrigatória do OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("A variável de ambiente OPENAI_API_KEY não está definida.")
client = OpenAI()

# Inicializa o banco de dados na inicialização da aplicação
def init_db():
    conn = sqlite3.connect(DATABASE_URL)
    cursor = conn.cursor()
    cursor.execute(
        '''
        CREATE TABLE IF NOT EXISTS saved_ids (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id TEXT NOT NULL UNIQUE,
            text TEXT,
            color_grade INTEGER,
            versatile_grade INTEGER,
            comfort_grade INTEGER,
            formal_grade INTEGER,
            stamps_grade INTEGER,
            cluster TEXT
        )
        '''
    )
    conn.commit()
    conn.close()

# Função auxiliar para garantir a existência dos diretórios de modelo
def ensure_model_dir():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Inicializa o banco de dados e garante diretório de modelos na inicialização da aplicação
@app.on_event("startup")
def startup_event():
    init_db()
    ensure_model_dir()
    logger.info("Aplicação inicializada com sucesso")

# --- Modelos Pydantic ---
class ImageIDs(BaseModel):
    ids: List[str]

class ClothingFeatures(BaseModel):
    cores_vivas: int = Field(..., ge=1, le=5, description="Classificação de 1 a 5 para 'Gosto de cores vivas'")
    versatilidade: int = Field(..., ge=1, le=5, description="Classificação de 1 a 5 para 'Prefiro peças versáteis que combinam com tudo'")
    conforto: int = Field(..., ge=1, le=5, description="Classificação de 1 a 5 para 'Busco conforto acima de estilo'")
    formalidade: int = Field(..., ge=1, le=5, description="Classificação de 1 a 5 para 'Gosto de roupas mais formais'")
    estampas: int = Field(..., ge=1, le=5, description="Classificação de 1 a 5 para 'Me atraem estampas chamativas'")

class UserProfile(BaseModel):
    features: ClothingFeatures
    
class ProfileResponse(BaseModel):
    cluster: int
    profile: str
    description: str

# Gera JSON Schema e remove restrições incompatíveis para OpenAI
def get_features_schema():
    schema = ClothingFeatures.model_json_schema()
    for prop in schema.get("properties", {}).values():
        prop.pop("minimum", None)
        prop.pop("maximum", None)
    schema["additionalProperties"] = False
    return schema

# Descrever os perfis de usuário
PROFILE_DESCRIPTIONS = {
    'Profissional Moderno': "Você valoriza roupas formais com um toque contemporâneo. Prioriza peças versáteis que podem transitar entre ambientes profissionais e sociais mais sofisticados.",
    'Casual Despojado': "Seu foco está no conforto e praticidade. Prefere roupas que ofereçam liberdade de movimento e bem-estar, sem abrir mão de um visual casual e descontraído.",
    'Aventureiro Fashion': "Você gosta de se expressar através de cores vibrantes e estampas marcantes. Não teme ousar e experimentar looks diferentes, priorizando autenticidade e criatividade."
}

@app.get("/clothes")
def get_random_clothes():
    """Retorna 10 itens de roupa aleatórios do dataset."""
    if df is None:
        return {"error": "Dataset não pôde ser carregado."}
    if df.empty:
        return {"message": "O dataset está vazio."}

    random_sample = df.sample(min(10, len(df)))
    processed = random_sample.copy()
    for col in processed.select_dtypes(include=["object"]).columns:
        processed[col] = processed[col].apply(
            lambda x: x.decode('utf-8', errors='replace') if isinstance(x, bytes) else str(x)
        )

    return processed.set_index(random_sample.index)["text"].to_dict()

@app.post("/clothes")
def save_image_ids(image_data: ImageIDs):
    """Recebe uma lista de IDs (índices) e salva no banco de dados SQLite."""
    conn = sqlite3.connect(DATABASE_URL)
    cursor = conn.cursor()
    saved_count = 0
    errors = []

    # Carrega os modelos para classificação
    try:
        kmeans, scaler = load_models()
    except HTTPException:
        # Se os modelos não estiverem disponíveis, continue sem classificação
        kmeans, scaler = None, None
        logger.warning("Modelos não encontrados. Continuando sem classificação de cluster.")

    for raw_id in image_data.ids:
        try:
            idx = int(raw_id)
        except ValueError:
            errors.append(f"ID inválido (não numérico): {raw_id}")
            continue

        if df is None or 'text' not in df.columns:
            errors.append(f"Dataset não carregado ou coluna 'text' ausente para ID {idx}.")
            continue
        if idx not in df.index:
            errors.append(f"ID {idx} não encontrado no dataset.")
            continue

        text_value = df.loc[idx, 'text']

        # --- OpenAI Structured Output obrigatório ---
        try:
            resp = client.chat.completions.create(
                model="o4-mini-2025-04-16",
                messages=[
                    {"role": "system", "content": (
                        "Você é um assistente que classifica descrições de roupas em 5 categorias."
                    )},
                    {"role": "user", "content": text_value}
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "ClothingFeatures",
                        "strict": True,
                        "schema": get_features_schema()
                    }
                }
            )
            features = ClothingFeatures.parse_raw(resp.choices[0].message.content)
        except Exception as e:
            logger.error(f"Erro OpenAI para ID {idx}: {e}")
            raise HTTPException(
                status_code=500,
                detail={"message": f"Falha obrigatória na OpenAI para ID {idx}", "error": str(e)}
            )

        # Classificar o cluster se os modelos estiverem disponíveis
        cluster_name = None
        if kmeans is not None and scaler is not None:
            try:
                features_array = np.array([[
                    features.cores_vivas,
                    features.versatilidade,
                    features.conforto,
                    features.formalidade,
                    features.estampas
                ]])
                scaled_features = scaler.transform(features_array)
                cluster = kmeans.predict(scaled_features)[0]
                cluster_name = CLUSTER_MAP.get(cluster)
            except Exception as e:
                logger.error(f"Erro ao classificar ID {idx}: {e}")
                # Continua sem o cluster se houver erro

        # --- Inserção no SQLite ---
        try:
            cursor.execute(
                """
                INSERT OR IGNORE INTO saved_ids
                  (image_id, text, color_grade, versatile_grade, comfort_grade, formal_grade, stamps_grade, cluster)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(idx), text_value,
                    features.cores_vivas,
                    features.versatilidade,
                    features.conforto,
                    features.formalidade,
                    features.estampas,
                    cluster_name
                )
            )
            if cursor.rowcount > 0:
                saved_count += 1
        except sqlite3.Error as db_e:
            logger.error(f"DB error ID {idx}: {db_e}")
            errors.append(f"Erro no DB ao salvar ID {idx}: {db_e}")

    conn.commit()
    conn.close()

    if errors:
        raise HTTPException(
            status_code=500,
            detail={
                "message": f"{saved_count} IDs salvos com sucesso, mas ocorreram erros.",
                "errors": errors
            }
        )

    return {"message": f"{saved_count} de {len(image_data.ids)} IDs salvos com sucesso."}

@app.post("/classify_profile", response_model=ProfileResponse)
def classify_user_profile(user_data: UserProfile):
    """
    Classifica o perfil do usuário com base nas features fornecidas.
    
    Este endpoint recebe as 5 características de preferência do usuário e retorna
    a classificação do perfil de moda correspondente, utilizando o modelo K-means treinado.
    """
    # Carrega os modelos de classificação
    kmeans, scaler = load_models()
    
    # Prepara os dados para classificação
    features = [
        user_data.features.cores_vivas,
        user_data.features.versatilidade,
        user_data.features.conforto,
        user_data.features.formalidade,
        user_data.features.estampas
    ]
    
    # Transforma as características no formato esperado pelo modelo
    features_array = np.array([features])
    
    try:
        # Normaliza os dados usando o mesmo scaler usado no treinamento
        scaled_features = scaler.transform(features_array)
        
        # Classifica o perfil do usuário
        cluster = int(kmeans.predict(scaled_features)[0])  # Converte para int padrão do Python
        profile = CLUSTER_MAP.get(cluster)
        description = PROFILE_DESCRIPTIONS.get(profile)
        
        # Retorna a classificação e descrição do perfil
        return ProfileResponse(
            cluster=cluster,
            profile=profile,
            description=description
        )
    except Exception as e:
        logger.error(f"Erro ao classificar perfil: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao classificar perfil: {str(e)}"
        )

# Para rodar localmente: uvicorn main:app --reload