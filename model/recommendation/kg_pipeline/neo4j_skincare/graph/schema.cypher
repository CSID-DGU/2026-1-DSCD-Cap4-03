// ── Constraints ──────────────────────────────────────────
CREATE CONSTRAINT prod_key  IF NOT EXISTS
  FOR (p:Product)     REQUIRE p.product_key IS UNIQUE;

CREATE CONSTRAINT ing_name  IF NOT EXISTS
  FOR (i:Ingredient)  REQUIRE i.name IS UNIQUE;

CREATE CONSTRAINT concern_name IF NOT EXISTS
  FOR (c:Concern)     REQUIRE c.name IS UNIQUE;

CREATE CONSTRAINT skintype_name IF NOT EXISTS
  FOR (s:SkinType)    REQUIRE s.name IS UNIQUE;

CREATE CONSTRAINT session_id IF NOT EXISTS
  FOR (u:UserSession) REQUIRE u.session_id IS UNIQUE;

CREATE CONSTRAINT rule_id IF NOT EXISTS
  FOR (r:Rule)        REQUIRE r.rule_id IS UNIQUE;

// ── Indexes ───────────────────────────────────────────────
CREATE INDEX prod_category IF NOT EXISTS
  FOR (p:Product) ON (p.category);

CREATE INDEX ing_irritation IF NOT EXISTS
  FOR (i:Ingredient) ON (i.irritation);

CREATE INDEX ing_comedogenicity IF NOT EXISTS
  FOR (i:Ingredient) ON (i.comedogenicity);