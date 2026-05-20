import { base64ToBlob } from '../utils/image';

export async function uploadImage(base64: string) {
  const blob = base64ToBlob(base64);

  const formData = new FormData();
  formData.append('file', blob, 'face.jpg');

  const res = await fetch('http://localhost:8080/api/upload', {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    throw new Error('업로드 실패');
  }

  return res.json(); // { url: "s3 url" }
}