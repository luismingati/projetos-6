from datetime import datetime
import uuid
from fastapi import FastAPI, File, HTTPException, Depends, UploadFile
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
from supabase import Client, create_client

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
CLUSTER_MAP = {0: 'Profissional Moderno', 1: 'Casual Despojado', 2: 'Aventureiro Fashion', 3: 'Esportivo Casual', 4: 'Minimalista Chic'}

# --- Configuração do Supabase ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_BUCKET_NAME = os.getenv("SUPABASE_BUCKET_NAME", "clothes-images")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("As variáveis SUPABASE_URL e SUPABASE_ANON_KEY devem estar definidas no .env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

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

    cursor.execute(
        '''
        CREATE TABLE IF NOT EXISTS uploaded_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            supabase_path TEXT NOT NULL UNIQUE,
            public_url TEXT,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            file_size INTEGER,
            content_type TEXT
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

class UploadResponse(BaseModel):
    message: str
    uploaded_files: List[Dict[str, str]]
    failed_files: List[Dict[str, str]]

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
    'Aventureiro Fashion': "Você gosta de se expressar através de cores vibrantes e estampas marcantes. Não teme ousar e experimentar looks diferentes, priorizando autenticidade e criatividade.",
    'Esportivo Casual': "Buscando o equilíbrio entre desempenho e estilo, você opta por peças funcionais com tecidos tecnológicos, mas que mantêm um visual descontraído e moderno para o dia a dia ativo.",
    'Minimalista Chic': "Você valoriza a simplicidade e a elegância atemporal. Prefere cores neutras, cortes clean e peças que transmitam sofisticação discreta e versatilidade."
}

# Função auxiliar para gerar nome único para arquivo
def generate_unique_filename(original_filename: str) -> str:
    """Gera um nome único para o arquivo baseado em timestamp e UUID"""
    file_extension = os.path.splitext(original_filename)[1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"{timestamp}_{unique_id}{file_extension}"

# Função para validar tipo de arquivo de imagem
def validate_image_file(file: UploadFile) -> bool:
    """Valida se o arquivo é uma imagem válida"""
    allowed_types = [
        "image/jpeg", "image/jpg", "image/png", 
        "image/gif", "image/webp", "image/bmp"
    ]
    return file.content_type in allowed_types

@app.post("/upload-images", response_model=UploadResponse)
async def upload_images_to_supabase(files: List[UploadFile] = File(...)):
    """
    Endpoint para fazer upload de múltiplas imagens para o Supabase Storage.
    
    Args:
        files: Lista de arquivos de imagem para upload
        
    Returns:
        UploadResponse com detalhes dos arquivos enviados e falhas
    """
    uploaded_files = []
    failed_files = []
    
    # Conecta ao banco local para registrar uploads
    conn = sqlite3.connect(DATABASE_URL)
    cursor = conn.cursor()
    
    for file in files:
        try:
            # Valida se é um arquivo de imagem
            if not validate_image_file(file):
                failed_files.append({
                    "filename": file.filename,
                    "error": f"Tipo de arquivo não suportado: {file.content_type}"
                })
                continue
            
            # Lê o conteúdo do arquivo
            file_content = await file.read()
            file_size = len(file_content)
            
            # Gera nome único para o arquivo
            unique_filename = generate_unique_filename(file.filename)
            supabase_path = f"uploads/{unique_filename}"
            
            # Faz upload para o Supabase Storage
            try:
                upload_result = supabase.storage.from_(SUPABASE_BUCKET_NAME).upload(
                    path=supabase_path,
                    file=file_content,
                    file_options={
                        "content-type": file.content_type,
                        "upsert": False  # Não sobrescreve arquivos existentes
                    }
                )
                
                # Gera URL pública do arquivo
                public_url = supabase.storage.from_(SUPABASE_BUCKET_NAME).get_public_url(supabase_path)
                
                # Salva informações no banco local
                cursor.execute(
                    """
                    INSERT INTO uploaded_images 
                    (filename, supabase_path, public_url, file_size, content_type)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (file.filename, supabase_path, public_url, file_size, file.content_type)
                )
                
                uploaded_files.append({
                    "original_filename": file.filename,
                    "supabase_path": supabase_path,
                    "public_url": public_url,
                    "file_size": f"{file_size} bytes"
                })
                
                logger.info(f"Upload bem-sucedido: {file.filename} -> {supabase_path}")
                
            except Exception as supabase_error:
                failed_files.append({
                    "filename": file.filename,
                    "error": f"Erro no Supabase: {str(supabase_error)}"
                })
                logger.error(f"Erro no upload para Supabase - {file.filename}: {supabase_error}")
                
        except Exception as e:
            failed_files.append({
                "filename": file.filename if file.filename else "arquivo_sem_nome",
                "error": f"Erro geral: {str(e)}"
            })
            logger.error(f"Erro geral no upload - {file.filename}: {e}")
    
    # Confirma transações no banco local
    conn.commit()
    conn.close()
    
    # Prepara resposta
    total_uploaded = len(uploaded_files)
    total_failed = len(failed_files)
    
    message = f"Upload concluído: {total_uploaded} arquivo(s) enviado(s) com sucesso"
    if total_failed > 0:
        message += f", {total_failed} arquivo(s) falharam"
    
    return UploadResponse(
        message=message,
        uploaded_files=uploaded_files,
        failed_files=failed_files
    )

@app.get("/uploaded-images")
def get_uploaded_images():
    """
    Retorna lista de todas as imagens que foram enviadas para o Supabase.
    """
    conn = sqlite3.connect(DATABASE_URL)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, filename, supabase_path, public_url, upload_date, file_size, content_type
        FROM uploaded_images 
        ORDER BY upload_date DESC
    """)
    
    images = []
    for row in cursor.fetchall():
        images.append({
            "id": row[0],
            "filename": row[1],
            "supabase_path": row[2],
            "public_url": row[3],
            "upload_date": row[4],
            "file_size": row[5],
            "content_type": row[6]
        })
    
    conn.close()
    
    return {
        "total_images": len(images),
        "images": images
    }

@app.delete("/uploaded-images/{image_id}")
def delete_uploaded_image(image_id: int):
    """
    Remove uma imagem do Supabase Storage e do banco local.
    """
    conn = sqlite3.connect(DATABASE_URL)
    cursor = conn.cursor()
    
    # Busca informações da imagem
    cursor.execute(
        "SELECT supabase_path, filename FROM uploaded_images WHERE id = ?",
        (image_id,)
    )
    result = cursor.fetchone()
    
    if not result:
        conn.close()
        raise HTTPException(status_code=404, detail="Imagem não encontrada")
    
    supabase_path, filename = result
    
    try:
        # Remove do Supabase Storage
        supabase.storage.from_(SUPABASE_BUCKET_NAME).remove([supabase_path])
        
        # Remove do banco local
        cursor.execute("DELETE FROM uploaded_images WHERE id = ?", (image_id,))
        conn.commit()
        conn.close()
        
        return {"message": f"Imagem '{filename}' removida com sucesso"}
        
    except Exception as e:
        conn.close()
        logger.error(f"Erro ao remover imagem {image_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao remover imagem: {str(e)}"
        )

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
    
@app.get("/user_profile", response_model=ProfileResponse)
def get_user_profile():
    # Carrega modelos treinados
    kmeans, scaler = load_models()
    
    # Conecta ao banco e calcula médias das features
    conn = sqlite3.connect(DATABASE_URL)
    cursor = conn.cursor()
    
    # Verifica se há dados suficientes
    cursor.execute("SELECT COUNT(*) FROM saved_ids")
    count = cursor.fetchone()[0]
    
    if count == 0:
        raise HTTPException(status_code=404, detail="Nenhuma roupa salva no banco...")
    
    # Calcula médias de todas as features
    cursor.execute("""
        SELECT 
            AVG(color_grade) as avg_cores_vivas,
            AVG(versatile_grade) as avg_versatilidade,
            AVG(comfort_grade) as avg_conforto,
            AVG(formal_grade) as avg_formalidade,
            AVG(stamps_grade) as avg_estampas
        FROM saved_ids
    """)
    
    # Processa e classifica
    features_array = np.array([cursor.fetchone()])
    scaled_features = scaler.transform(features_array)
    cluster = kmeans.predict(scaled_features)[0]
    
    # Retorna perfil
    return ProfileResponse(
        cluster=cluster,
        profile=CLUSTER_MAP.get(cluster),
        description=PROFILE_DESCRIPTIONS.get(CLUSTER_MAP.get(cluster))
    )

# Para rodar localmente: uvicorn main:app --reload