from fastapi import FastAPI, HTTPException
import pandas as pd
import sqlite3
from pydantic import BaseModel, Field
from typing import List
import os
from openai import OpenAI
import logging
from dotenv import load_dotenv

# Carrega variáveis de ambiente do .env
load_dotenv()

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Carregar o dataset uma vez ao iniciar a aplicação para otimizar
try:
    df = pd.read_parquet(
        "hf://datasets/wbensvage/clothes_desc/data/my_clothes_desc.parquet"
    )
except Exception as e:
    logger.error(f"Erro ao carregar o dataset: {e}")
    df = None  # Define df como None se houver erro

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
            stamps_grade INTEGER
        )
        '''
    )
    conn.commit()
    conn.close()

init_db()

# --- Modelos Pydantic ---
class ImageIDs(BaseModel):
    ids: List[str]

class ClothingFeatures(BaseModel):
    cores_vivas: int = Field(..., ge=1, le=5, description="Classificação de 1 a 5 para 'Gosto de cores vivas'")
    versatilidade: int = Field(..., ge=1, le=5, description="Classificação de 1 a 5 para 'Prefiro peças versáteis que combinam com tudo'")
    conforto: int = Field(..., ge=1, le=5, description="Classificação de 1 a 5 para 'Busco conforto acima de estilo'")
    formalidade: int = Field(..., ge=1, le=5, description="Classificação de 1 a 5 para 'Gosto de roupas mais formais'")
    estampas: int = Field(..., ge=1, le=5, description="Classificação de 1 a 5 para 'Me atraem estampas chamativas'")

# Gera JSON Schema e remove restrições incompatíveis
schema = ClothingFeatures.model_json_schema()
for prop in schema.get("properties", {}).values():
    prop.pop("minimum", None)
    prop.pop("maximum", None)
schema["additionalProperties"] = False

@app.post("/clothes")
def save_image_ids(image_data: ImageIDs):
    """Recebe uma lista de IDs (índices) e salva no banco de dados SQLite."""
    conn = sqlite3.connect(DATABASE_URL)
    cursor = conn.cursor()
    saved_count = 0
    errors = []

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
                        "schema": schema
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

        # --- Inserção no SQLite ---
        try:
            cursor.execute(
                """
                INSERT OR IGNORE INTO saved_ids
                  (image_id, text, color_grade, versatile_grade, comfort_grade, formal_grade, stamps_grade)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(idx), text_value,
                    features.cores_vivas,
                    features.versatilidade,
                    features.conforto,
                    features.formalidade,
                    features.estampas
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

# Para rodar localmente: uvicorn main:app --reload
