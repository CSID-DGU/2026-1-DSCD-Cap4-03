from pydantic import BaseModel, Field


class ImageCreateRequest(BaseModel):
    storage_url: str = Field(min_length=1, max_length=1000)
    s3_key: str = Field(min_length=1, max_length=500)
    original_file_name: str | None = Field(default=None, max_length=255)
    mime_type: str | None = Field(default=None, max_length=100)
    file_size: int | None = Field(default=None, ge=1)
    crop_data: dict | None = None
    upload_status: str = Field(default="UPLOADED", max_length=30)


class ImageCreateResponse(BaseModel):
    image_id: int
    user_id: int
    storage_url: str
    s3_key: str
    uploaded_at: str
