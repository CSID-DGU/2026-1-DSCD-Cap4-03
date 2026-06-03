from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parents[2] / ".env")


@dataclass
class Settings:
    app_name: str = os.getenv("APP_NAME", "ROUPLE Backend")
    backend_host: str = os.getenv("BACKEND_HOST", "127.0.0.1")
    backend_port: int = int(os.getenv("BACKEND_PORT", "8000"))
    s3_bucket: str = os.getenv("S3_BUCKET", "rouple-images-dev")
    s3_region: str = os.getenv("S3_REGION", "ap-northeast-2")
    s3_prefix: str = os.getenv("S3_PREFIX", "users")
    presign_expire_seconds: int = int(os.getenv("S3_PRESIGN_EXPIRE_SECONDS", "900"))
    aws_access_key_id: str | None = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str | None = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_session_token: str | None = os.getenv("AWS_SESSION_TOKEN")
    s3_public_base_url: str | None = os.getenv("S3_PUBLIC_BASE_URL")
    mysql_host: str = os.getenv("MYSQL_HOST", "127.0.0.1")
    mysql_port: int = int(os.getenv("MYSQL_PORT", "3306"))
    mysql_user: str = os.getenv("MYSQL_USER", "root")
    mysql_password: str = os.getenv("MYSQL_PASSWORD", "")
    mysql_db: str = os.getenv("MYSQL_DB", "Rouple_db")
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "cap4cap4")
    database_url: str | None = os.getenv("DATABASE_URL")
    db_auto_create_tables: bool = os.getenv("DB_AUTO_CREATE_TABLES", "true").lower() == "true"
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", os.getenv("KG_OPENAI_MODEL", "gpt-5.4-mini"))
    dgu_llm_api_key: str | None = os.getenv("DGU_LLM_API_KEY")
    dgu_llm_base_url: str = os.getenv("DGU_LLM_BASE_URL", "https://factchat-cloud.mindlogic.ai/v1/gateway")
    dgu_llm_model: str = os.getenv("DGU_LLM_MODEL", "gpt-5.4-mini")
    skin_model_checkpoint: str | None = os.getenv("SKIN_MODEL_CHECKPOINT")
    skin_model_device: str = os.getenv("SKIN_MODEL_DEVICE", "cpu")
    skin_model_img_size: int = int(os.getenv("SKIN_MODEL_IMG_SIZE", "224"))
    frontend_origins_raw: str = os.getenv(
        "FRONTEND_ORIGINS",
        "http://127.0.0.1:8000,http://localhost:8000,http://127.0.0.1:5173,http://localhost:5173,http://127.0.0.1:3000,http://localhost:3000",
    )

    @property
    def frontend_origins(self) -> list[str]:
        return [origin.strip() for origin in self.frontend_origins_raw.split(",") if origin.strip()]

    @property
    def resolved_s3_public_base_url(self) -> str:
        if self.s3_public_base_url:
            return self.s3_public_base_url.rstrip("/")
        return f"https://{self.s3_bucket}.s3.{self.s3_region}.amazonaws.com"

    @property
    def sqlalchemy_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        if os.getenv("USE_LOCAL_SQLITE", "true").lower() == "true":
            return "sqlite:///./rouple_dev.db"
        return (
            f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_db}?charset=utf8mb4"
        )


settings = Settings()
