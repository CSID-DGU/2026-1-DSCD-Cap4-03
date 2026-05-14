import { useState, useCallback, useRef } from 'react';
import Cropper from 'react-easy-crop';
import type { Area } from 'react-easy-crop';
import { useNavigate } from 'react-router-dom';
import './DiagnosisPage.css';

const TARGET_WIDTH = 480;
const TARGET_HEIGHT = 640;

const GUIDES = [
  { icon: '💡', text: '밝은 조명에서 촬영해주세요' },
  { icon: '👁️', text: '정면을 바라보고 찍어주세요' },
  { icon: '😐', text: '무표정으로 촬영해주세요' },
  { icon: '🚫', text: '화장품 없이 생얼로 찍어주세요' },
  { icon: '👓', text: '안경 및 악세사리를 제거해주세요' },
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
  const [isUploading, setIsUploading] = useState(false);

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

  const handleAnalyze = async () => {
    if (!resultImage) return;
    setIsUploading(true);

    try {
      // ── S3 업로드 (API 연결 시 아래 주석 해제)
      // const formData = new FormData();
      // formData.append('image', resultBlob!, 'skin.jpg');
      // const data = await uploadImage(formData);
      // navigate('/analysis', { state: { imageUrl: data.url } });

      // 임시: base64 그대로 전달
      navigate('/analysis', { state: { imageUrl: resultImage } });
    } catch (err) {
      console.error('업로드 실패:', err);
    } finally {
      setIsUploading(false);
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
          <div className="dp-badge">AI 피부 진단</div>
          <h1 className="dp-title">피부 사진을 올려주세요</h1>
          <p className="dp-sub">정확한 분석을 위해 아래 가이드를 따라 촬영해주세요</p>
        </div>
      </div>

      <div className="dp-body">

        {/* ── 가이드 ── */}
        <div className="dp-guide-row">
          {GUIDES.map((g) => (
            <div className="dp-guide-item" key={g.text}>
              <span className="dp-guide-icon">{g.icon}</span>
              <span className="dp-guide-text">{g.text}</span>
            </div>
          ))}
        </div>

        {/* ── 스텝 인디케이터 ── */}
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
              <div className="dp-upload-icon">📷</div>
              <div className="dp-upload-title">사진을 업로드하세요</div>
              <div className="dp-upload-sub">클릭하거나 파일을 끌어다 놓으세요</div>
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
                  {/* 얼굴 가이드 오버레이 */}
                  <div className="dp-face-guide" />
                  <span className="dp-face-eye dp-face-eye--left" />
                  <span className="dp-face-eye dp-face-eye--right" />
                  <span className="dp-face-mouth" />
                </div>

                {/* 줌 슬라이더 */}
                <div className="dp-zoom-row">
                  <span className="dp-zoom-icon">🔍−</span>
                  <input
                    type="range"
                    className="dp-zoom-slider"
                    min={1} max={3} step={0.05}
                    value={zoom}
                    onChange={(e) => setZoom(Number(e.target.value))}
                  />
                  <span className="dp-zoom-icon">🔍+</span>
                </div>
              </div>

              {/* 안내 */}
              <div className="dp-crop-guide-col">
                <div className="dp-crop-guide-title">✅ 이렇게 맞춰주세요</div>
                <ul className="dp-crop-guide-list">
                  <li>눈, 코, 입이 가이드 안에 들어오게</li>
                  <li>얼굴이 화면 중앙에 오도록</li>
                  <li>턱선이 아래 원 안에 포함되게</li>
                  <li>슬라이더로 크기를 조절하세요</li>
                </ul>
                <div className="dp-crop-size-info">
                  <span>📐 저장 크기</span>
                  <strong>{TARGET_WIDTH} × {TARGET_HEIGHT}px</strong>
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
                  <div className="dp-check-item">✅ 480 × 640px 고정 크기</div>
                  <div className="dp-check-item">✅ JPEG 압축률 90%</div>
                  <div className="dp-check-item">✅ 서버 전송 준비 완료</div>
                </div>

                <div className="dp-preview-btns">
                  <button className="dp-btn-secondary" onClick={() => setStep('crop')}>
                    다시 자르기
                  </button>
                  <button
                    className={`dp-btn-primary ${isUploading ? 'loading' : ''}`}
                    onClick={handleAnalyze}
                    disabled={isUploading}
                  >
                    {isUploading ? '업로드 중...' : '🔬 분석 시작하기'}
                  </button>
                </div>
              </div>

            </div>
          </div>
        )}

      </div>
    </div>
  );
}
