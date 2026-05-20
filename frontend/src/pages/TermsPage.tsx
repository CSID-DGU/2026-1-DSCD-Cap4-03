import './PolicyPage.css';

export default function TermsPage() {
  return (
    <div className="policy-page">
      <div className="policy-inner">

        <div className="policy-header">
          <div className="policy-badge">Terms of Service</div>
          <h1 className="policy-title">이용약관</h1>
          <p className="policy-updated">최종 업데이트: 2026년 5월 19일</p>
        </div>

        <div className="policy-body">

          <p className="policy-intro">
            본 약관은 ROUPLE(이하 "서비스")이 제공하는 AI 피부 분석 및 스킨케어 루틴 추천 서비스의 이용에 관한 조건을 규정합니다.
            서비스를 이용함으로써 본 약관에 동의한 것으로 간주합니다.
          </p>

          <section className="policy-section">
            <h2>제1조 (목적)</h2>
            <p>
              본 약관은 ROUPLE이 제공하는 AI 피부 분석 및 맞춤형 스킨케어 루틴 추천 서비스의 이용 조건과
              절차, 이용자와 서비스 간의 권리·의무 및 책임 사항을 규정함을 목적으로 합니다.
            </p>
          </section>

          <section className="policy-section">
            <h2>제2조 (정의)</h2>
            <ul>
              <li><strong>"서비스"</strong>란 ROUPLE이 제공하는 AI 피부 분석, 루틴 추천, 제품 정보 등 일체의 서비스를 말합니다.</li>
              <li><strong>"이용자"</strong>란 본 약관에 동의하고 서비스를 이용하는 자를 말합니다.</li>
              <li><strong>"AI 분석 결과"</strong>란 이용자가 업로드한 이미지를 AI 모델이 분석하여 생성한 피부 상태 점수 및 코멘트를 말합니다.</li>
            </ul>
          </section>

          <section className="policy-section">
            <h2>제3조 (회원 가입)</h2>
            <ul>
              <li>이용자는 서비스가 정한 양식에 따라 정확한 정보를 입력하여 회원 가입을 신청합니다.</li>
              <li>타인의 정보를 도용하거나 허위 정보를 입력한 경우 서비스 이용이 제한될 수 있습니다.</li>
              <li>만 14세 미만의 아동은 법정 대리인의 동의 없이 회원 가입이 불가합니다.</li>
            </ul>
          </section>

          <section className="policy-section">
            <h2>제4조 (서비스 제공)</h2>
            <p>서비스는 다음을 제공합니다.</p>
            <ul>
              <li>AI 기반 피부 상태 분석 (트러블, 건조, 모공, 색소침착, 주름, 처짐 6개 지표)</li>
              <li>분석 결과에 기반한 맞춤형 스킨케어 루틴 추천</li>
              <li>화장품 성분 정보 및 알레르기 필터링</li>
              <li>분석 기록 및 루틴 저장</li>
            </ul>
            <p>
              서비스는 시스템 점검, 기술적 문제 등으로 일시 중단될 수 있으며, 이 경우 사전 공지를 원칙으로 합니다.
            </p>
          </section>

          <section className="policy-section">
            <h2>제5조 (AI 분석 결과의 한계)</h2>
            <p>
              AI 피부 분석 결과는 참고 목적으로 제공되며, 의학적 진단이나 치료를 대체하지 않습니다.
              피부 질환이 의심되거나 심각한 피부 문제가 있는 경우 반드시 전문 의료인과 상담하시기 바랍니다.
              서비스는 AI 분석 결과의 정확성을 보증하지 않으며, 이를 근거로 발생한 손해에 대해 책임을 지지 않습니다.
            </p>
          </section>

          <section className="policy-section">
            <h2>제6조 (이용자의 의무)</h2>
            <ul>
              <li>이용자는 본인 얼굴 사진만 업로드해야 하며, 타인의 이미지를 무단으로 사용해서는 안 됩니다.</li>
              <li>서비스의 정상적인 운영을 방해하는 행위를 해서는 안 됩니다.</li>
              <li>서비스를 통해 취득한 정보를 서비스의 사전 동의 없이 복제·배포해서는 안 됩니다.</li>
              <li>계정 및 비밀번호는 이용자 본인이 관리하며, 타인에게 양도할 수 없습니다.</li>
            </ul>
          </section>

          <section className="policy-section">
            <h2>제7조 (지식재산권)</h2>
            <p>
              서비스에 포함된 AI 모델, 추천 알고리즘, 디자인 등에 관한 지식재산권은 서비스 운영자에게 귀속됩니다.
              이용자가 서비스를 통해 작성한 리뷰, 댓글 등의 콘텐츠에 대한 권리는 이용자에게 귀속되나,
              서비스는 서비스 개선 목적으로 이를 활용할 수 있습니다.
            </p>
          </section>

          <section className="policy-section">
            <h2>제8조 (서비스 이용 제한)</h2>
            <p>다음 행위를 한 이용자는 서비스 이용이 제한되거나 계정이 삭제될 수 있습니다.</p>
            <ul>
              <li>타인의 개인정보 도용 또는 허위 정보 등록</li>
              <li>서비스 시스템에 대한 무단 접근 또는 해킹 시도</li>
              <li>음란물, 폭력적 콘텐츠 등 불법 이미지 업로드</li>
              <li>상업적 목적의 무단 크롤링 또는 자동화 접속</li>
            </ul>
          </section>

          <section className="policy-section">
            <h2>제9조 (면책 조항)</h2>
            <p>
              서비스는 천재지변, 불가항력적 사유로 인한 서비스 장애에 대해 책임을 지지 않습니다.
              이용자 본인의 귀책 사유로 인한 서비스 이용 장애에 대해서도 책임을 지지 않습니다.
              추천된 제품 구매 후 발생한 피부 트러블 등 부작용에 대해 서비스는 책임을 지지 않습니다.
            </p>
          </section>

          <section className="policy-section">
            <h2>제10조 (약관 변경)</h2>
            <p>
              서비스는 필요한 경우 약관을 변경할 수 있으며, 변경된 약관은 서비스 내 공지 또는 이메일을 통해
              변경 7일 전부터 안내합니다. 변경 이후 계속 서비스를 이용하면 변경된 약관에 동의한 것으로 간주합니다.
            </p>
          </section>

          <section className="policy-section">
            <h2>제11조 (분쟁 해결)</h2>
            <p>
              본 약관과 관련된 분쟁은 대한민국 법률에 따라 해결하며, 분쟁 발생 시 서울중앙지방법원을 관할 법원으로 합니다.
            </p>
          </section>

        </div>
      </div>
    </div>
  );
}
