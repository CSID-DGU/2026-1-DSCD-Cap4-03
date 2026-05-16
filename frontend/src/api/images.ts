import { api } from './client';

export interface PresignResponse {
  upload_url: string;
  public_url: string;
  s3_key: string;
  expires_in: number;
}

export interface ImageResponse {
  image_id: number;
  user_id: number;
  storage_url: string;
  s3_key: string;
  uploaded_at: string;
}

export const imagesApi = {
  presign: (body: { file_name: string; mime_type: string; file_size: number }) =>
    api.post<PresignResponse>('/files/presign', body),

  createImage: (body: {
    storage_url: string;
    s3_key: string;
    original_file_name: string;
    mime_type: string;
    file_size: number;
    crop_data: { x: number; y: number; width: number; height: number };
    upload_status: string;
  }) => api.post<ImageResponse>('/images', body),

  /** S3 presigned URL로 blob 직접 PUT (백엔드 미경유) */
  uploadToS3: async (uploadUrl: string, blob: Blob): Promise<void> => {
    await fetch(uploadUrl, {
      method: 'PUT',
      body: blob,
      headers: { 'Content-Type': 'image/jpeg' },
    });
  },
};
