첨부된 레퍼런스 이미지를 분석하여, 리액트 기반으로 전체 웹사이트를 제작할 수 있도록 풀-와이드 Hero Section을 디자인하라. 이 Hero Section은 이후 제작될 모든 섹션과 페이지의 **Design System 기준점**이 되어야 한다. 웹사이트 타입은 **Personalized skin analysis-skincare recommendation Platform**이다.

1. Design System Definition:
- **background Color:** #ffffff (차분하고 따뜻한 라이트 그레이 배경 컬러)
- **primary Color:**  #A78BFA (약간 살짝 배경 퍼플)
- **Accent Color:** #7c3aed (브랜드를 대표하는 퍼플, CTA 및 핵심 강조 요소에만 제한적으로 사용)
- **Text Color:** #1A1A1A (헤드라인), #5F5F5F (바디 텍스트)
- **Typography:** Modern sans-serif 기반. Bold한 대문자 또는 세미볼드 헤드라인, 가독성 높은 Regular 바디 텍스트. 텍스트에는 그라데이션을 절대 사용하지 않는다.
- **Border Radius:** 24px 이상, 부드럽고 친근한 라운드 코너
- **Shadow:** 낮은 대비의 소프트 드롭 섀도우 (예: 0px 12px 32px rgba(0,0,0,0.08))
- **Icon Style:** 배경 없는 단색 아이콘만 사용하거나, 모든 아이콘에 동일한 단일 배경색 적용

2. Layout & Structure (Hero Section):
- 데스크톱 기준 1440px 이상 풀-와이드 프레임
- 좌측: 텍스트 중심의 콘텐츠 영역 (Eyebrow Text, Headline, Description, Primary / Secondary CTA)
- 우측: 실제 생활감이 느껴지는 인물 중심의 고해상도 이미지
- 모든 요소는 **Auto Layout** 기반으로 구성하여 반응형 확장과 유지보수가 용이해야 한다
- Hero 영역 하단은 부드러운 페이드 또는 화이트 영역으로 자연스럽게 다음 섹션과 연결

3. UI Elements & Animation:
- 버튼은 **Component**로 정의하며, Primary 버튼은 Accent Color 단색 배경 + 다크 텍스트
- Secondary 버튼은 아웃라인 스타일 또는 뉴트럴 톤
- 카드, 뱃지, 네비게이션 요소 모두 동일한 Border Radius와 Shadow 규칙을 공유
- **Framer Motion 로직**을 고려한 애니메이션 적용:
  - Hero 진입 시 텍스트는 `staggered reveal on load`
  - 버튼과 인터랙티브 요소는 `smooth hover transition (opacity / y-axis 4~8px)`
  - 과도한 모션이나 화려한 효과는 배제하고, 안정적이고 신뢰감 있는 움직임만 허용

4. Consistency Mandate:
- 이후 제작되는 모든 섹션(Features, How it works, Testimonials, Pricing, Footer)과 모든 서브 페이지는
  반드시 위에서 정의한 **Design System Definition**을 엄격하게 준수해야 한다.
- 새로운 색상, 다른 Radius, 다른 아이콘 스타일의 임의 추가를 절대 허용하지 않는다.
- AI 생성물처럼 보이는 시각적 패턴을 명확히 배제한다:
  그라데이션 사용 금지, 파스텔 톤 아이콘 배경 금지, 아이콘별 상이한 색상 조합 금지.
  아이콘은 반드시 배경 없는 단색 아이콘이거나,
  전체 시스템에서 동일하게 정의된 단일 배경 규칙만을 사용해야 한다.
- 이 Hero Section을 기준으로 전체 웹사이트의 톤앤매너, 레이아웃, UI 규칙을
  일관되고 의도적으로 확장하라.
