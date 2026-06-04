import { useState, useCallback, useRef } from 'react';
import Cropper from 'react-easy-crop';
import type { Area } from 'react-easy-crop';
import { useNavigate } from 'react-router-dom';
import { imagesApi } from '../api/images';
import {
  Lightbulb, Eye, Minus, Ban, Glasses, Scissors,
  Camera, ZoomIn, ZoomOut, Check, Ruler, Microscope,
  type LucideIcon,
} from 'lucide-react';
import './DiagnosisPage.css';


const TARGET_WIDTH = 480;
const TARGET_HEIGHT = 640;

const SKIP_S3 = false;

const GUIDES: { Icon: LucideIcon; text: string }[] = [
  { Icon: Lightbulb, text: '밝은 조명에서 촬영해주세요' },
  { Icon: Eye,       text: '정면을 바라보고 찍어주세요' },
  { Icon: Minus,     text: '무표정으로 촬영해주세요' },
  { Icon: Ban,       text: '화장 없이 생얼로 찍어주세요' },
  { Icon: Glasses,   text: '안경 및 악세사리를 제거해주세요' },
  { Icon: Scissors,  text: '앞머리가 있다면 넘기고 찍어주세요' },
];

type Step = 'upload' | 'crop' | 'preview';

export default function DiagnosisPage() {
  const navigate = useNavigate();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [step, setStep] = useState<Step>('upload');
  const [image, setImage] = useState<string | null>(null);
  const [crop, setCrop] = useState({ x: 0, y: 0 });
  const [zoom, setZoom] = useState(1);
  const [croppedAreaPixels, setCroppedAreaPixels] = useState<Area | null>(null);
  const [resultImage, setResultImage] = useState<string | null>(null);
  const [resultBlob, setResultBlob] = useState<Blob | null>(null);

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) {
      const url = URL.createObjectURL(e.target.files[0]);
      setImage(url);
      setStep('crop');
    }
  };

  const onCropComplete = useCallback((_: Area, pixels: Area) => {
    setCroppedAreaPixels(pixels);
  }, []);

  const createCroppedImage = async () => {
    if (!image || !croppedAreaPixels) return;

    const img = new Image();
    img.src = image;
    await new Promise((resolve) => (img.onload = resolve));

    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = TARGET_WIDTH;
    canvas.height = TARGET_HEIGHT;

    ctx.drawImage(
      img,
      croppedAreaPixels.x, croppedAreaPixels.y,
      croppedAreaPixels.width, croppedAreaPixels.height,
      0, 0, TARGET_WIDTH, TARGET_HEIGHT
    );

    // base64 (미리보기용)
    const base64 = canvas.toDataURL('image/jpeg', 0.9);
    setResultImage(base64);

    // Blob (S3 업로드용)
    canvas.toBlob((blob) => {
      if (blob) setResultBlob(blob);
    }, 'image/jpeg', 0.9);

    setStep('preview');
  };

  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const handleAnalyze = async () => {
    if (!resultImage || !resultBlob) return;
    setIsUploading(true);
    setUploadError(null);

    try {
      let imageId: number;
      let imageUrl: string;

      if (SKIP_S3) {
        const res = await imagesApi.localUpload(resultBlob);
        imageId  = res.image_id;
        imageUrl = resultImage;
      } else {
        // 1. presigned URL 발급
        const presign = await imagesApi.presign({
          file_name: 'skin.jpg',
          mime_type: 'image/jpeg',
          file_size: resultBlob.size,
        });

        // 2. S3 직접 PUT 업로드
        await imagesApi.uploadToS3(presign.upload_url, resultBlob);

        // 3. 이미지 메타데이터 저장
        const img = await imagesApi.createImage({
          storage_url:        presign.public_url,
          s3_key:             presign.s3_key,
          original_file_name: 'skin.jpg',
          mime_type:          'image/jpeg',
          file_size:          resultBlob.size,
          crop_data: croppedAreaPixels
            ? { x: croppedAreaPixels.x, y: croppedAreaPixels.y, width: TARGET_WIDTH, height: TARGET_HEIGHT }
            : { x: 0, y: 0, width: TARGET_WIDTH, height: TARGET_HEIGHT },
          upload_status: 'UPLOADED',
        });
        imageId  = img.image_id;
        imageUrl = presign.public_url || resultImage;
      }

      setIsUploading(false);
      navigate('/loading', { state: { type: 'analysis', image_id: imageId, imageUrl } });
    } catch (err) {
      setIsUploading(false);
      const msg = err instanceof Error ? err.message : '';
      setUploadError(
        msg.includes('Failed to fetch')
          ? 'S3 CORS 오류: S3 버킷에 http://localhost:5173 이 허용되지 않았어요. 백엔드 팀에 S3 CORS 설정을 요청하세요.'
          : msg || '업로드 중 오류가 발생했어요. 다시 시도해주세요.'
      );
    }
  };

  const handleReset = () => {
    setImage(null);
    setResultImage(null);
    setResultBlob(null);
    setStep('upload');
    setCrop({ x: 0, y: 0 });
    setZoom(1);
  };

  return (
    <div className="dp-page">

      {/* ── 배너 ── */}
      <div className="dp-banner">
        <div className="dp-banner-inner">
          <div className="dp-badge">SKIN ANALYSIS</div>
          <h1 className="dp-title">내 피부, 제대로 알고 계세요?</h1>
          <p className="dp-sub">얼굴 사진 한 장으로 피부 타입, 수분, 유분, 트러블까지 AI가 분석해드려요</p>
        </div>
      </div>

      <div className="dp-body">
        <div className="dp-layout">

          {/* ── 좌측: 가이드 박스 ── */}
          <div className="dp-guide-col">
            <div className="dp-guide-box">
              <div className="dp-guide-box-title">
                <Camera size={14} />
                촬영 가이드
              </div>
              {GUIDES.map((g) => (
                <div className="dp-guide-item" key={g.text}>
                  <g.Icon size={16} color="#7c3aed" />
                  <span>{g.text}</span>
                </div>
              ))}
            </div>
          </div>

          {/* ── 가운데: 스텝 + 카드 ── */}
          <div className="dp-main-col">

            {/* 스텝 인디케이터 */}
            <div className="dp-steps">
              {(['upload', 'crop', 'preview'] as Step[]).map((s, idx) => {
                const labels = ['사진 선택', '얼굴 맞추기', '확인 & 분석'];
                const isActive = step === s;
                const isDone = ['upload', 'crop', 'preview'].indexOf(step) > idx;
                return (
                  <div className="dp-step-item" key={s}>
                    <div className={`dp-step-circle ${isActive ? 'active' : ''} ${isDone ? 'done' : ''}`}>
                      {isDone ? '✓' : idx + 1}
                    </div>
                    <span className={`dp-step-label ${isActive ? 'active' : ''}`}>{labels[idx]}</span>
                    {idx < 2 && <div className={`dp-step-line ${isDone ? 'done' : ''}`} />}
                  </div>
                );
              })}
            </div>

            {/* ── STEP 1: 사진 선택 ── */}
            {step === 'upload' && (
              <div className="dp-card dp-upload-card">
                <input
                  type="file"
                  accept="image/*"
                  ref={fileInputRef}
                  onChange={onFileChange}
                  style={{ display: 'none' }}
                />
                <div className="dp-upload-area" onClick={() => fileInputRef.current?.click()}>
                  <div className="dp-upload-icon"><Camera size={36} color="#7c3aed" /></div>
                  <div className="dp-upload-title">사진을 업로드하세요</div>
                  <div className="dp-upload-sub">클릭하여 파일을 업로드 해주세요</div>
                  <div className="dp-upload-hint">JPG, PNG · 최대 10MB</div>
                  <button className="dp-btn-upload">사진 선택하기</button>
                </div>
              </div>
            )}

            {/* ── STEP 2: 크롭 ── */}
            {step === 'crop' && image && (
              <div className="dp-card dp-crop-card">
                <div className="dp-crop-layout">

                  {/* 크롭 영역 */}
                  <div className="dp-crop-col">
                    <div className="dp-crop-label">얼굴을 원형 가이드에 맞춰주세요</div>
                    <div className="dp-crop-wrapper">
                      <Cropper
                        image={image}
                        crop={crop}
                        zoom={zoom}
                        aspect={3 / 4}
                        onCropChange={setCrop}
                        onZoomChange={setZoom}
                        onCropComplete={onCropComplete}
                      />
                      <div className="dp-face-guide" />
                      <span className="dp-face-eye dp-face-eye--left" />
                      <span className="dp-face-eye dp-face-eye--right" />
                      <span className="dp-face-mouth" />
                    </div>

                    {/* 줌 슬라이더 */}
                    <div className="dp-zoom-row">
                      <ZoomOut size={16} color="#7c3aed" className="dp-zoom-icon" />
                      <input
                        type="range"
                        className="dp-zoom-slider"
                        min={1} max={3} step={0.05}
                        value={zoom}
                        onChange={(e) => setZoom(Number(e.target.value))}
                      />
                      <ZoomIn size={16} color="#7c3aed" className="dp-zoom-icon" />
                    </div>
                  </div>

                  {/* 안내 */}
                  <div className="dp-crop-guide-col">
                    <div className="dp-crop-guide-title">
                      <Check size={14} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 4 }} />이렇게 맞춰주세요
                    </div>
                    <ul className="dp-crop-guide-list">
                      <li>눈, 코, 입이 가이드 안에 들어오게</li>
                      <li>얼굴이 화면 중앙에 오도록</li>
                      <li>최대한 가이드라인에 맞게</li>
                      <li>슬라이더로 크기를 조절하세요</li>
                    </ul>
                    <div className="dp-crop-size-info">
                      <Ruler size={13} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 4 }} />저장 크기
                      <strong style={{ marginLeft: 6 }}>{TARGET_WIDTH} × {TARGET_HEIGHT}px</strong>
                    </div>
                    <div className="dp-crop-btns">
                      <button className="dp-btn-secondary" onClick={handleReset}>다시 선택</button>
                      <button className="dp-btn-primary" onClick={createCroppedImage}>얼굴 맞추기 완료</button>
                    </div>
                  </div>

                </div>
              </div>
            )}

            {/* ── STEP 3: 미리보기 & 분석 ── */}
            {step === 'preview' && resultImage && (
              <div className="dp-card dp-preview-card">
                <div className="dp-preview-layout">

                  <div className="dp-preview-img-col">
                    <div className="dp-preview-label">최종 이미지</div>
                    <div className="dp-preview-img-wrap">
                      <img src={resultImage} alt="최종 이미지" className="dp-preview-img" />
                      <div className="dp-preview-size-badge">{TARGET_WIDTH}×{TARGET_HEIGHT}</div>
                    </div>
                  </div>

                  <div className="dp-preview-info-col">
                    <div className="dp-preview-title">분석 준비 완료!</div>
                    <p className="dp-preview-sub">
                      사진이 잘 나왔나요?<br />
                      얼굴이 선명하게 보이면 분석을 시작해보세요.
                    </p>

                    <div className="dp-preview-checklist">
                      <div className="dp-check-item"><Check size={13} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 4 }} />480 × 640px 고정 크기</div>
                      <div className="dp-check-item"><Check size={13} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 4 }} />JPEG 압축률 90%</div>
                      <div className="dp-check-item"><Check size={13} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 4 }} />서버 전송 준비 완료</div>
                    </div>

                    <div className="dp-preview-btns">
                      <button className="dp-btn-secondary" onClick={() => { setStep('crop'); setUploadError(null); }}>
                        다시 자르기
                      </button>
                      <button
                        className={`dp-btn-primary ${isUploading ? 'loading' : ''}`}
                        onClick={handleAnalyze}
                        disabled={isUploading}
                      >
                        {isUploading
                          ? '업로드 중...'
                          : <><Microscope size={15} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 6 }} />분석 시작하기</>}
                      </button>
                    </div>
                    {uploadError && (
                      <div style={{ marginTop: 12, padding: '10px 14px', background: '#FEF2F2', border: '1.5px solid #FECACA', borderRadius: 8, fontSize: 13, color: '#dc2626', fontWeight: 600, lineHeight: 1.5 }}>
                        {uploadError}
                      </div>
                    )}
                  </div>

                </div>
              </div>
            )}

          </div>

        </div>
      </div>
    </div>
  );
}
