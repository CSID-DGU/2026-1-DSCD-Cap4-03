import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { routineApi, type SavedRoutineItem } from '../api/routine';
import { analysisApi, type SkinHistoryItem } from '../api/analysis';
import { vanityApi } from '../api/vanity';
import { ChevronRight, Sparkles } from 'lucide-react';
import './RoutineHistoryPage.css';
import LoadingSpinner from '../components/common/LoadingSpinner';

function formatDate(iso: string) {
  const d = new Date(iso);
  const mm = String(d.getMonth() + 1).padStart(2, '0');
  const dd = String(d.getDate()).padStart(2, '0');
  const hh = String(d.getHours()).padStart(2, '0');
  const min = String(d.getMinutes()).padStart(2, '0');
  return `${d.getFullYear()}-${mm}-${dd} ${hh}:${min}`;
}

type BudgetInfo = {
  total_budget_min:    number | null;
  total_budget_max:    number | null;
  toner_budget_min:    number | null;
  toner_budget_max:    number | null;
  emulsion_budget_min: number | null;
  emulsion_budget_max: number | null;
  ampoule_budget_min:  number | null;
  ampoule_budget_max:  number | null;
  cream_budget_min:    number | null;
  cream_budget_max:    number | null;
};

type SessionGroup = { session_id: number; routines: SavedRoutineItem[] };

function formatRange(min: number | null | undefined, max: number | null | undefined, label: string): string | null {
  if ((min == null) && (max == null)) return null;
  if (max == null) return `${label} ${(min ?? 0).toLocaleString()}원+`;
  if (min == null || min === 0)      return `${label} ~${max.toLocaleString()}원`;
  return `${label} ${min.toLocaleString()}~${max.toLocaleString()}원`;
}

function getBudgetLabel(b: BudgetInfo): string {
  const parts = [
    formatRange(b.total_budget_min,    b.total_budget_max,    '전체'),
    formatRange(b.toner_budget_min,    b.toner_budget_max,    '토너'),
    formatRange(b.emulsion_budget_min, b.emulsion_budget_max, '에멀젼'),
    formatRange(b.ampoule_budget_min,  b.ampoule_budget_max,  '앰플'),
    formatRange(b.cream_budget_min,    b.cream_budget_max,    '크림'),
  ].filter(Boolean) as string[];
  return parts.length > 0 ? parts.join(' · ') : '예산 미설정';
}


export default function RoutineHistoryPage() {
  const navigate = useNavigate();

  const [analysisHistory, setAnalysisHistory]   = useState<SkinHistoryItem[]>([]);
  const [loading, setLoading]                   = useState(true);
  const [expandedResultId, setExpandedResultId] = useState<number | null>(null);
  const [sessionsByResult, setSessionsByResult] = useState<Record<number, SessionGroup[]>>({});
  const [budgetBySession, setBudgetBySession]   = useState<Record<number, BudgetInfo>>({});
  const [loadingBudgets, setLoadingBudgets]     = useState<Set<number>>(new Set());
  const [vanitySessionIds, setVanitySessionIds]         = useState<Set<number>>(new Set());
  const [vanityFixedBySession, setVanityFixedBySession] = useState<Record<number, string[]>>({});

  useEffect(() => {
    vanityApi.getRoutineHistory()
      .then(res => {
        setVanitySessionIds(new Set(res.routines.map(r => r.recommendation_session_id)));
      })
      .catch(() => {});

    Promise.all([
      routineApi.getHistory(),
      analysisApi.getHistory(),
    ]).then(async ([routineRes, analysisRes]) => {
      setAnalysisHistory(analysisRes.items);

      const items = routineRes.items;

      const missingIds = [...new Set(
        items.filter((r) => r.result_id == null).map((r) => r.session_id)
      )];
      const resultIdBySession: Record<number, number> = {};
      await Promise.all(
        missingIds.map(async (sid) => {
          try {
            const rec = await routineApi.getRecommendation(sid);
            resultIdBySession[sid] = rec.result_id;
            setBudgetBySession((prev) => ({
              ...prev,
              [sid]: {
                total_budget_min:    rec.total_budget_min    ?? null,
                total_budget_max:    rec.total_budget_max    ?? null,
                toner_budget_min:    rec.toner_budget_min    ?? null,
                toner_budget_max:    rec.toner_budget_max    ?? null,
                emulsion_budget_min: rec.emulsion_budget_min ?? null,
                emulsion_budget_max: rec.emulsion_budget_max ?? null,
                ampoule_budget_min:  rec.ampoule_budget_min  ?? null,
                ampoule_budget_max:  rec.ampoule_budget_max  ?? null,
                cream_budget_min:    rec.cream_budget_min    ?? null,
                cream_budget_max:    rec.cream_budget_max    ?? null,
              },
            }));
          } catch { /* 무시 */ }
        })
      );

      const byResult: Record<number, SessionGroup[]> = {};
      items.forEach((routine) => {
        const rid = routine.result_id ?? resultIdBySession[routine.session_id];
        if (rid == null) return;
        if (!byResult[rid]) byResult[rid] = [];
        const existing = byResult[rid].find((s) => s.session_id === routine.session_id);
        if (existing) existing.routines.push(routine);
        else byResult[rid].push({ session_id: routine.session_id, routines: [routine] });
      });
      setSessionsByResult(byResult);

    }).catch(() => {}).finally(() => setLoading(false));
  }, []);

  const handleExpand = async (resultId: number, sessions: SessionGroup[]) => {
    if (expandedResultId === resultId) { setExpandedResultId(null); return; }

    const toFetch = sessions.filter((s) => budgetBySession[s.session_id] === undefined);
    setExpandedResultId(resultId);
    if (toFetch.length === 0) return;

    /* 바니티 세션이면 고정 제품 목록도 로드 */
    toFetch.forEach(session => {
      if (vanitySessionIds.has(session.session_id) && !vanityFixedBySession[session.session_id]) {
        vanityApi.getRoutineDetail(session.session_id)
          .then(detail => {
            const names = (detail.routine_recommendation_results?.fixed_products ?? [])
              .map(p => p.product_name);
            setVanityFixedBySession(prev => ({ ...prev, [session.session_id]: names }));
          })
          .catch(() => {});
      }
    });

    setLoadingBudgets((prev) => new Set([...prev, ...toFetch.map((s) => s.session_id)]));
    await Promise.all(
      toFetch.map(async (session) => {
        const fallback: BudgetInfo = {
          total_budget_min: null,    total_budget_max: null,
          toner_budget_min: null,    toner_budget_max: null,
          emulsion_budget_min: null, emulsion_budget_max: null,
          ampoule_budget_min: null,  ampoule_budget_max: null,
          cream_budget_min: null,    cream_budget_max: null,
        };
        try {
          const rec = await routineApi.getRecommendation(session.session_id);
          setBudgetBySession((prev) => ({
            ...prev,
            [session.session_id]: {
              total_budget_min:    rec.total_budget_min    ?? null,
              total_budget_max:    rec.total_budget_max    ?? null,
              toner_budget_min:    rec.toner_budget_min    ?? null,
              toner_budget_max:    rec.toner_budget_max    ?? null,
              emulsion_budget_min: rec.emulsion_budget_min ?? null,
              emulsion_budget_max: rec.emulsion_budget_max ?? null,
              ampoule_budget_min:  rec.ampoule_budget_min  ?? null,
              ampoule_budget_max:  rec.ampoule_budget_max  ?? null,
              cream_budget_min:    rec.cream_budget_min    ?? null,
              cream_budget_max:    rec.cream_budget_max    ?? null,
            },
          }));
        } catch {
          setBudgetBySession((prev) => ({ ...prev, [session.session_id]: fallback }));
        } finally {
          setLoadingBudgets((prev) => { const n = new Set(prev); n.delete(session.session_id); return n; });
        }
      })
    );
  };

  const sorted = [...analysisHistory].sort(
    (a, b) => new Date(b.analyzed_at).getTime() - new Date(a.analyzed_at).getTime()
  );
  const latest = sorted[0];

  return (
    <div className="rh-page">

      <div className="rh-banner">
        <div className="rh-banner-inner">
          <div className="rh-badge">ROUTINE REPORT</div>
          <h1 className="rh-title">오늘 내 피부엔 어떤 루틴이 맞을까요?</h1>
          <p className="rh-sub">내 피부 분석 결과를 바탕으로 설계된 맞춤 루틴을 확인해보세요</p>
        </div>
      </div>

      <div className="rh-body">

        <div
          className="rh-new-card"
          onClick={() =>
            latest
              ? navigate('/routine/budget', { state: { resultId: latest.result_id, imageId: latest.image_id } })
              : navigate('/diagnosis')
          }
        >
          <div className="rh-new-icon"><Sparkles size={28} color="#7c3aed" /></div>
          <div>
            <div className="rh-new-title">
              {latest ? '최근 분석 결과로 루틴 추천받기' : '먼저 피부 진단을 받아보세요'}
            </div>
            <div className="rh-new-sub">
              {latest ? `${formatDate(latest.analyzed_at)} 분석 기반` : '진단 후 맞춤 루틴을 추천받을 수 있어요'}
            </div>
          </div>
          <span className="rh-new-arrow">→</span>
        </div>

        <div className="rh-section-title">분석 결과별 루틴</div>

        {loading && <LoadingSpinner text="루틴 기록을 불러오는 중이에요" />}

        {!loading && analysisHistory.length === 0 && (
          <div className="rh-empty">
            <p>아직 분석 기록이 없어요.</p>
            <button className="rh-btn-primary" onClick={() => navigate('/diagnosis')}>첫 피부 진단 받기</button>
          </div>
        )}

        {!loading && analysisHistory.length > 0 && (
          <div className="rh-list">
            {sorted.map((r, idx) => {
              const sessions   = sessionsByResult[r.result_id] ?? [];
              const isExpanded = expandedResultId === r.result_id;

              return (
                <div key={r.result_id} className={`rh-card ${isExpanded ? 'rh-card--expanded' : ''}`}>

                  <div className="rh-card-main">
                    <div className="rh-card-left">
                      {r.image_url && <img src={r.image_url} alt="썸네일" className="rh-thumb" />}
                      {idx === 0 && <span className="rh-latest-badge">최신</span>}
                    </div>

                    <div className="rh-card-info">
                      <div className="rh-card-date-row">
                        <div className="rh-card-date">{formatDate(r.analyzed_at)}</div>
                        {r.skin_type && <span className="rh-card-skin-tag">{r.skin_type}</span>}
                      </div>
                      {r.display_scores && Object.keys(r.display_scores).length > 0 && (
                        <div className="rh-card-scores">
                          {([
                            { key: 'acne',         label: '진정' },
                            { key: 'dryness',      label: '수분' },
                            { key: 'sagging',      label: '탄력' },
                            { key: 'pore',         label: '모공' },
                            { key: 'pigmentation', label: '색소침착' },
                            { key: 'wrinkle',      label: '주름' },
                          ] as const).map(({ key, label }, i, arr) => (
                            <span key={key} className="rh-score-chip">
                              {label} <b>{Math.round((1 - (r.display_scores[key] ?? 0)) * 100)}</b>
                              {i < arr.length - 1 && <span className="rh-score-dot">·</span>}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>

                    <div className="rh-card-actions">
                      <button
                        className={`rh-view-btn ${isExpanded ? 'active' : ''} ${sessions.length === 0 ? 'disabled' : ''}`}
                        onClick={() => sessions.length > 0 && handleExpand(r.result_id, sessions)}
                        disabled={sessions.length === 0}
                      >
                        {sessions.length === 0
                          ? '추천 기록 없음'
                          : isExpanded
                            ? '닫기 ↑'
                            : `이전 루틴 보기 (${sessions.length}회)`}
                      </button>
                      <button
                        className="rh-select-btn"
                        onClick={() => navigate('/routine/budget', { state: { resultId: r.result_id, imageId: r.image_id } })}
                      >
                        새로운 루틴 추천받기 →
                      </button>
                    </div>
                  </div>

                  {isExpanded && sessions.length > 0 && (
                    <div className="rh-expanded">
                      {sessions.map((session) => {
                        const budget     = budgetBySession[session.session_id];
                        const isLoadingB = loadingBudgets.has(session.session_id);
                        return (
                          <div
                            key={session.session_id}
                            className="rh-session-box"
                            onClick={async () => {
                              if (vanitySessionIds.has(session.session_id)) {
                                try {
                                  const result = await vanityApi.getRoutineDetail(session.session_id);
                                  navigate('/vanity/routine', { state: { result } });
                                } catch {
                                  navigate('/routine/result', { state: { session_id: session.session_id } });
                                }
                              } else {
                                navigate('/routine/result', { state: { session_id: session.session_id } });
                              }
                            }}
                          >
                            <div className="rh-session-date">
                              추천일 {formatDate(session.routines[0]?.saved_at)}
                              {vanitySessionIds.has(session.session_id) && (
                                <span style={{ marginLeft: 8, fontSize: 10, fontWeight: 700, background: '#EDE9FE', color: '#7c3aed', padding: '2px 8px', borderRadius: 999, verticalAlign: 'middle' }}>
                                  내 화장대 루틴
                                </span>
                              )}
                            </div>
                            <div className="rh-session-body">
                              <div className="rh-session-left">
                                {vanitySessionIds.has(session.session_id) && (
                                  <div className="rh-session-budget" style={{ marginBottom: 6 }}>
                                    <span className="rh-session-budget-label">고정 제품</span>
                                    <span className="rh-session-budget-value" style={{ color: '#7c3aed' }}>
                                      {vanityFixedBySession[session.session_id]?.join(', ') || '—'}
                                    </span>
                                  </div>
                                )}
                                <div className="rh-session-budget">
                                  <span className="rh-session-budget-label">예산 조건</span>
                                  <span className="rh-session-budget-value">
                                    {isLoadingB ? '불러오는 중...' : budget ? getBudgetLabel(budget) : '—'}
                                  </span>
                                </div>
                              </div>
                              <ChevronRight size={20} color="#7c3aed" />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}

                </div>
              );
            })}
          </div>
        )}

      </div>
    </div>
  );
}
