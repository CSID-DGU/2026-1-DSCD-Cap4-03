import './PolicyPage.css';

export default function PrivacyPage() {
  return (
    <div className="policy-page">
      <div className="policy-inner">

        <div className="policy-header">
          <div className="policy-badge">Privacy Policy</div>
          <h1 className="policy-title">개인정보처리방침</h1>
          <p className="policy-updated">최종 업데이트: 2026년 5월 19일</p>
        </div>

        <div className="policy-body">

          <p className="policy-intro">
            ROUPLE(이하 "서비스")은 이용자의 개인정보를 중요하게 생각하며, 개인정보 보호법 및 관련 법령을 준수합니다.
            본 방침은 서비스 이용 과정에서 수집되는 개인정보의 항목, 수집 목적, 보유 기간 및 이용자의 권리에 대해 안내합니다.
          </p>

          <section className="policy-section">
            <h2>1. 수집하는 개인정보 항목</h2>
            <p>서비스는 다음과 같은 개인정보를 수집합니다.</p>
            <div className="policy-table-wrap">
              <table className="policy-table">
                <thead>
                  <tr>
                    <th>구분</th>
                    <th>항목</th>
                    <th>수집 시점</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>필수</td>
                    <td>이름, 닉네임, 이메일, 비밀번호(암호화 저장)</td>
                    <td>회원가입 시</td>
                  </tr>
                  <tr>
                    <td>선택</td>
                    <td>성별, 생년월일, 피부 타입, 피부 고민, 알레르기 성분</td>
                    <td>프로필 설정 시</td>
                  </tr>
                  <tr>
                    <td>자동 수집</td>
                    <td>얼굴 이미지(분석 후 삭제 가능), 접속 IP, 서비스 이용 기록</td>
                    <td>서비스 이용 중</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </section>

          <section className="policy-section">
            <h2>2. 개인정보 수집 및 이용 목적</h2>
            <ul>
              <li>회원 가입 및 본인 확인</li>
              <li>AI 피부 분석 서비스 제공 및 분석 결과 저장</li>
              <li>맞춤형 스킨케어 루틴 추천</li>
              <li>서비스 개선을 위한 통계 분석 (비식별 처리 후 활용)</li>
              <li>공지사항 전달 및 고객 문의 응대</li>
            </ul>
          </section>

          <section className="policy-section">
            <h2>3. 개인정보 보유 및 이용 기간</h2>
            <p>
              수집한 개인정보는 회원 탈퇴 시 즉시 파기합니다. 단, 관련 법령에 따라 일정 기간 보관이 필요한 경우 해당 기간 동안 보관됩니다.
            </p>
            <ul>
              <li>전자상거래 소비자 보호에 관한 법률: 계약 또는 청약 철회 기록 5년</li>
              <li>통신비밀보호법: 로그인 기록 3개월</li>
            </ul>
          </section>

          <section className="policy-section">
            <h2>4. 개인정보 제3자 제공</h2>
            <p>
              서비스는 이용자의 개인정보를 원칙적으로 제3자에게 제공하지 않습니다.
              다만, 이용자가 사전에 동의한 경우 또는 법령에 근거한 경우에는 예외로 합니다.
            </p>
          </section>

          <section className="policy-section">
            <h2>5. 개인정보 처리 위탁</h2>
            <p>서비스는 원활한 서비스 제공을 위해 다음과 같이 개인정보 처리 업무를 위탁합니다.</p>
            <ul>
              <li><strong>Amazon Web Services (AWS S3)</strong>: 이미지 파일 저장 및 관리</li>
            </ul>
          </section>

          <section className="policy-section">
            <h2>6. 얼굴 이미지 처리</h2>
            <p>
              업로드한 얼굴 이미지는 AI 피부 분석에만 사용되며, 분석 완료 후 서버에 안전하게 보관됩니다.
              분석 이미지는 서비스 탈퇴 시 또는 이용자 요청 시 즉시 삭제됩니다.
              이미지는 마케팅 또는 제3자 제공 목적으로 절대 사용되지 않습니다.
            </p>
          </section>

          <section className="policy-section">
            <h2>7. 이용자의 권리</h2>
            <p>이용자는 언제든지 다음 권리를 행사할 수 있습니다.</p>
            <ul>
              <li>개인정보 열람 요청</li>
              <li>개인정보 정정·삭제 요청</li>
              <li>개인정보 처리 정지 요청</li>
              <li>회원 탈퇴 (마이페이지에서 직접 처리 가능)</li>
            </ul>
          </section>

          <section className="policy-section">
            <h2>8. 개인정보 보호책임자</h2>
            <div className="policy-contact">
              <p><strong>담당 팀:</strong> Cap4 (2026-1 데이터사이언스 캡스톤 디자인 3조)</p>
              <p><strong>이메일:</strong> cap4.rouple@example.com</p>
            </div>
          </section>

          <section className="policy-section">
            <h2>9. 개인정보처리방침 변경</h2>
            <p>
              본 방침은 법령 또는 서비스 변경에 따라 수정될 수 있습니다.
              변경 사항은 서비스 내 공지 또는 이메일을 통해 사전 안내합니다.
            </p>
          </section>

        </div>
      </div>
    </div>
  );
}
