from pydantic import BaseModel, Field


class PresignRequest(BaseModel):
    file_name: str = Field(min_length=1, max_length=255)
    mime_type: str = Field(min_length=1, max_length=100)
    file_size: int | None = Field(default=None, ge=1)


class PresignResponse(BaseModel):
    upload_url: str
    public_url: str
    s3_key: str
    expires_in: int
