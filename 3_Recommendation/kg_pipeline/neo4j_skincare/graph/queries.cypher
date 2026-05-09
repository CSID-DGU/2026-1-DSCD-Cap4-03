// ================================================================
// 1. 그래프 탐색 / 디버깅
// ================================================================

// 특정 제품의 전성분 + 성분별 irritation/comedogenicity 확인
MATCH (p:Product {product_key: $product_key})-[:CONTAINS]->(i:Ingredient)
RETURN i.name          AS ingredient,
       i.irritation    AS irritation,
       i.comedogenicity AS comedogenicity,
       i.function_raw  AS functions
ORDER BY i.irritation DESC;

// 특정 성분이 어떤 Concern에 연결되는지 확인
MATCH (i:Ingredient {name: $ingredient_name})-[h:HELPS]->(c:Concern)
RETURN i.name AS ingredient, c.name AS concern, h.weight AS weight;

// 특정 제품의 CONFLICTS 엣지 전체 조회
MATCH (p:Product {product_key: $product_key})-[:CONTAINS]->(a:Ingredient)
      -[cf:CONFLICTS]->(b:Ingredient)
RETURN a.name AS ing1, b.name AS ing2,
       cf.risk AS risk, cf.source AS source
ORDER BY cf.risk DESC;

// UserSession 상태 확인 (concern + skin_type + allergy)
MATCH (u:UserSession {session_id: $session_id})
OPTIONAL MATCH (u)-[hc:HAS_CONCERN]->(c:Concern)
OPTIONAL MATCH (u)-[:HAS_SKIN_TYPE]->(st:SkinType)
OPTIONAL MATCH (u)-[:HAS_ALLERGY]->(al:Ingredient)
RETURN u.session_id                          AS session,
       u.gender                              AS gender,
       collect(DISTINCT {concern: c.name,
               importance: hc.importance})   AS concerns,
       st.name                               AS skin_type,
       collect(DISTINCT al.name)             AS allergies;


// ================================================================
// 2. Hard Filter (hard_filter.py에서 호출)
// ================================================================

// HF1: 알러지 성분 포함 여부
MATCH (u:UserSession {session_id: $session_id})-[:HAS_ALLERGY]->(i:Ingredient)
      <-[:CONTAINS]-(p:Product {product_key: $product_key})
RETURN count(i) AS hit;

// HF2: 피부타입 부적합 (IRRITATES 있고 SUITS 없는 경우 fit=0)
MATCH (u:UserSession {session_id: $session_id})-[:HAS_SKIN_TYPE]->(st:SkinType)
MATCH (p:Product {product_key: $product_key})-[:CONTAINS]->(i:Ingredient)
WITH st,
     count(CASE WHEN (i)-[:IRRITATES]->(st) THEN 1 END) AS irr_cnt,
     count(CASE WHEN (i)-[:SUITS]->(st)     THEN 1 END) AS suit_cnt
RETURN
  CASE WHEN irr_cnt > 0 AND suit_cnt = 0 THEN 0 ELSE 1 END AS fit_score;

// HF3: 그래프에 Product 노드 존재 여부
MATCH (p:Product {product_key: $product_key})
RETURN count(p) AS exists;


// ================================================================
// 3. Soft Re-rank (soft_score.py에서 호출)
// ================================================================

// concern_match_score:
// UserSession → HAS_CONCERN → Concern ← HELPS ← Ingredient ← CONTAINS ← Product
MATCH (u:UserSession {session_id: $session_id})-[hc:HAS_CONCERN]->(c:Concern)
      <-[h:HELPS]-(i:Ingredient)<-[:CONTAINS]-(p:Product {product_key: $product_key})
RETURN sum(hc.importance * h.weight) AS concern_score;

// skin_type_bonus:
// SUITS 비율 - IRRITATES 비율
MATCH (u:UserSession {session_id: $session_id})-[:HAS_SKIN_TYPE]->(st:SkinType)
MATCH (p:Product {product_key: $product_key})-[:CONTAINS]->(i:Ingredient)
WITH st,
     count(CASE WHEN (i)-[:SUITS]->(st)     THEN 1 END) AS suit_cnt,
     count(CASE WHEN (i)-[:IRRITATES]->(st) THEN 1 END) AS irr_cnt,
     count(i) AS total
RETURN
  CASE WHEN total = 0 THEN 0.0
       ELSE toFloat(suit_cnt - irr_cnt) / total
  END AS skin_bonus;

// irritation_penalty:
// 유저 피부타입 기준 IRRITATES 엣지 score 합산
MATCH (u:UserSession {session_id: $session_id})-[:HAS_SKIN_TYPE]->(st:SkinType)
MATCH (p:Product {product_key: $product_key})-[:CONTAINS]->(i:Ingredient)
      -[r:IRRITATES]->(st)
RETURN sum(r.score) AS irr_sum;

// 3개 점수 한 번에 조회 (soft_score 단일 쿼리 최적화 버전)
MATCH (u:UserSession {session_id: $session_id})-[:HAS_SKIN_TYPE]->(st:SkinType)
MATCH (p:Product {product_key: $product_key})-[:CONTAINS]->(i:Ingredient)
WITH u, p, st, collect(i) AS ings
// concern_score
OPTIONAL MATCH (u)-[hc:HAS_CONCERN]->(c:Concern)<-[h:HELPS]-(i2:Ingredient)
WHERE i2 IN ings
WITH u, p, st, ings,
     sum(hc.importance * h.weight) AS concern_score
// skin_bonus
WITH p, st, ings, concern_score,
     size([i IN ings WHERE (i)-[:SUITS]->(st)])     AS suit_cnt,
     size([i IN ings WHERE (i)-[:IRRITATES]->(st)]) AS irr_cnt,
     size(ings) AS total
// irritation_penalty
WITH concern_score,
     CASE WHEN total = 0 THEN 0.0
          ELSE toFloat(suit_cnt - irr_cnt) / total END AS skin_bonus,
     toFloat(irr_cnt) * 0.05                           AS irr_penalty
RETURN concern_score,
       skin_bonus,
       CASE WHEN irr_penalty > 1.0 THEN 1.0
            ELSE irr_penalty END AS irr_penalty;


// ================================================================
// 4. 루틴 조합 (routine_builder.py / conflict_checker.py에서 호출)
// ================================================================

// 슬롯별 후보 제품 조회 (카테고리 + product_key 필터)
MATCH (p:Product)-[:IN_CATEGORY]->(cat:Category)
WHERE cat.name IN $categories
  AND p.product_key IN $candidate_keys
RETURN p.product_key AS product_key,
       p.name        AS name,
       p.brand       AS brand,
       p.category    AS category,
       p.price       AS price;

// 루틴 내 CONFLICTS 엣지 조회 (충돌 패널티 계산)
UNWIND $product_keys AS pk
MATCH (p:Product {product_key: pk})-[:CONTAINS]->(a:Ingredient)
      -[cf:CONFLICTS]->(b:Ingredient)<-[:CONTAINS]-(p2:Product)
WHERE p2.product_key IN $product_keys
  AND p.product_key <> p2.product_key
RETURN a.name    AS ing1,
       b.name    AS ing2,
       cf.risk   AS risk,
       cf.source AS source;

// am/pm 금지 성분 포함 여부 확인
UNWIND $product_keys AS pk
MATCH (p:Product {product_key: pk})-[:CONTAINS]->(i:Ingredient)
WHERE i.name IN $avoid_list
RETURN p.product_key AS product_key,
       i.name        AS ingredient;

// 루틴 내 GOOD 궁합 성분 쌍 보너스 점수 (선택 적용)
UNWIND $product_keys AS pk
MATCH (p:Product {product_key: pk})-[:CONTAINS]->(a:Ingredient)
      -[:GOOD_WITH]->(b:Ingredient)<-[:CONTAINS]-(p2:Product)
WHERE p2.product_key IN $product_keys
  AND p.product_key <> p2.product_key
RETURN a.name AS ing1, b.name AS ing2,
       count(*) AS pair_count;