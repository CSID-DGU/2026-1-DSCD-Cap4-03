-- MySQL 8.0+
-- Core entities DDL (0529 modified flow)

CREATE DATABASE IF NOT EXISTS Rouple_db
    DEFAULT CHARACTER SET utf8mb4
    DEFAULT COLLATE utf8mb4_0900_ai_ci;
    
USE Rouple_db;

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

DROP TABLE IF EXISTS PRODUCT;
DROP TABLE IF EXISTS PRODUCT_REVIEW;
DROP TABLE IF EXISTS INGREDIENT;
DROP TABLE IF EXISTS PRODUCT_INGREDIENT;
DROP TABLE IF EXISTS INGREDIENT_CONFLICT;

DROP TABLE IF EXISTS USER;
DROP TABLE IF EXISTS USER_PROFILE;
DROP TABLE IF EXISTS USER_ALLERGY;
DROP TABLE IF EXISTS USER_WISHLIST;
DROP TABLE IF EXISTS USER_VANITY;

DROP TABLE IF EXISTS USER_IMAGE;
DROP TABLE IF EXISTS SKIN_ANALYSIS_RESULT;
DROP TABLE IF EXISTS VANITY_MATCH_ITEM;
DROP TABLE IF EXISTS VANITY_MATCH_SESSION;

DROP TABLE IF EXISTS RECOMMENDATION_CANDIDATE;
DROP TABLE IF EXISTS RECOMMENDATION_RERANKED;
DROP TABLE IF EXISTS RECOMMENDATION_SESSION;
DROP TABLE IF EXISTS RECOMMENDATION_ROUTINE;
DROP TABLE IF EXISTS RECOMMENDATION_ITEM;

SET FOREIGN_KEY_CHECKS = 1;

-- COSMETICS: PRODUCT & INGREDIENT
CREATE TABLE PRODUCT (
    product_id BIGINT PRIMARY KEY,
    brand_name VARCHAR(50) NULL,
    brand_name_kor VARCHAR(50) NULL,
    product_name VARCHAR(100) NULL,
    product_name_kor VARCHAR(100) NULL,
    category VARCHAR(30) NULL,
    `function` VARCHAR(20) NULL,
    ranking INT NULL,
    price INT NULL,
    sim_1 BIGINT NULL,
    sim_2 BIGINT NULL,
    sim_3 BIGINT NULL,
    sim_4 BIGINT NULL,
    KEY idx_product_category (category),
    KEY idx_product_ranking (ranking),
    KEY idx_product_sim_1 (sim_1),
    KEY idx_product_sim_2 (sim_2),
    KEY idx_product_sim_3 (sim_3),
    KEY idx_product_sim_4 (sim_4),
    CONSTRAINT fk_product_sim_1
        FOREIGN KEY (sim_1) REFERENCES PRODUCT(product_id)
        ON DELETE SET NULL ON UPDATE CASCADE,
    CONSTRAINT fk_product_sim_2
        FOREIGN KEY (sim_2) REFERENCES PRODUCT(product_id)
        ON DELETE SET NULL ON UPDATE CASCADE,
    CONSTRAINT fk_product_sim_3
        FOREIGN KEY (sim_3) REFERENCES PRODUCT(product_id)
        ON DELETE SET NULL ON UPDATE CASCADE,
    CONSTRAINT fk_product_sim_4
        FOREIGN KEY (sim_4) REFERENCES PRODUCT(product_id)
        ON DELETE SET NULL ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE PRODUCT_REVIEW (
    review_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    product_id BIGINT NOT NULL,
    pro1 VARCHAR(50) NULL,
    pro2 VARCHAR(50) NULL,
    pro3 VARCHAR(50) NULL,
    pro4 VARCHAR(50) NULL,
    pro5 VARCHAR(50) NULL,
    pro6 VARCHAR(50) NULL,
    pro7 VARCHAR(50) NULL,
    con1 VARCHAR(50) NULL,
    con2 VARCHAR(50) NULL,
    con3 VARCHAR(50) NULL,
    con4 VARCHAR(50) NULL,
    con5 VARCHAR(50) NULL,
    con6 VARCHAR(50) NULL,
    con7 VARCHAR(50) NULL,
    pros_text TEXT NULL,
    cons_text TEXT NULL,
    UNIQUE KEY uq_product_review_product_id (product_id),
    CONSTRAINT fk_product_review_product
        FOREIGN KEY (product_id) REFERENCES PRODUCT(product_id)
        ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE INGREDIENT (
    ingredient_id BIGINT PRIMARY KEY,
    ingredient_name VARCHAR(200) NOT NULL,
    `function` TEXT NULL,
    rating VARCHAR(10) NULL,
    irritation VARCHAR(10) NULL,
    comedogenicity VARCHAR(10) NULL,
    cas_no VARCHAR(100) NULL,
    ec_no VARCHAR(100) NULL,
    allergy_category VARCHAR(30) NULL,
    UNIQUE KEY uq_ingredient_name (ingredient_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE PRODUCT_INGREDIENT (
    product_ingredient_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    product_id BIGINT NOT NULL,
    ingredient_id BIGINT NOT NULL,
    inci_brand VARCHAR(100) NULL,
    inci_product_name VARCHAR(100) NULL,
    `function` VARCHAR(200) NULL,
    image_url TEXT NULL,
    KEY idx_pi_product_id (product_id),
    KEY idx_pi_ingredient_id (ingredient_id),
    UNIQUE KEY uq_pi_product_ingredient (product_id, ingredient_id),
    CONSTRAINT fk_pi_product
        FOREIGN KEY (product_id) REFERENCES PRODUCT(product_id)
        ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT fk_pi_ingredient
        FOREIGN KEY (ingredient_id) REFERENCES INGREDIENT(ingredient_id)
        ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE INGREDIENT_CONFLICT (
    conflict_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    ingredient1_id BIGINT NOT NULL,
    ingredient2_id BIGINT NOT NULL,
    ingredient1_name VARCHAR(100) NULL,
    ingredient2_name VARCHAR(100) NULL,    KEY idx_conflict_ing1 (ingredient1_id),
    KEY idx_conflict_ing2 (ingredient2_id),
    UNIQUE KEY uq_conflict_pair (ingredient1_id, ingredient2_id),
    CONSTRAINT fk_conflict_ing1
        FOREIGN KEY (ingredient1_id) REFERENCES INGREDIENT(ingredient_id)
        ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT fk_conflict_ing2
        FOREIGN KEY (ingredient2_id) REFERENCES INGREDIENT(ingredient_id)
        ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- USER & PROFILE
CREATE TABLE USER (
    user_id INT AUTO_INCREMENT PRIMARY KEY,
    user_name VARCHAR(10) NULL,
    login_type VARCHAR(10) NOT NULL,
    email VARCHAR(50) NOT NULL,
    password VARCHAR(255) NULL,
    nickname VARCHAR(20) NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    UNIQUE KEY uq_user_email (email)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE USER_PROFILE (
    user_id INT PRIMARY KEY,
    gender VARCHAR(20) NULL,
    birth DATE NULL,
    skin_type VARCHAR(30) NULL,
    skin_concern VARCHAR(100) NULL,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    CONSTRAINT fk_user_profile_user
        FOREIGN KEY (user_id) REFERENCES USER(user_id)
        ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE USER_ALLERGY (
    allergy_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    allergy_category VARCHAR(30) NULL,
    allergy_ingredient VARCHAR(200) NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    KEY idx_user_allergy_user_id (user_id),
    KEY idx_user_allergy_category (allergy_category),
    UNIQUE KEY uq_user_allergy_pair (user_id, allergy_category, allergy_ingredient),
    CONSTRAINT fk_user_allergy_user
        FOREIGN KEY (user_id) REFERENCES USER(user_id)
        ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE USER_WISHLIST (
    wishlist_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    product_id BIGINT NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uq_wishlist_user_product (user_id, product_id),
    KEY idx_wishlist_user (user_id),
    KEY idx_wishlist_product (product_id),
    CONSTRAINT fk_wishlist_user
        FOREIGN KEY (user_id) REFERENCES USER(user_id)
        ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT fk_wishlist_product
        FOREIGN KEY (product_id) REFERENCES PRODUCT(product_id)
        ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE USER_VANITY (
    vanity_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    product_id BIGINT NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uq_vanity_user_product (user_id, product_id),
    KEY idx_vanity_user (user_id),
    KEY idx_vanity_product (product_id),
    CONSTRAINT fk_vanity_user
        FOREIGN KEY (user_id) REFERENCES USER(user_id)
        ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT fk_vanity_product
        FOREIGN KEY (product_id) REFERENCES PRODUCT(product_id)
        ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- SKIN IMAGE ANALYSIS
CREATE TABLE USER_IMAGE (
    image_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    storage_url VARCHAR(1000) NOT NULL,
    s3_key VARCHAR(500) NULL,
    original_file_name VARCHAR(255) NULL,
    mime_type VARCHAR(100) NULL,
    file_size INT NULL,
    crop_data TEXT NULL,
    upload_status VARCHAR(30) NULL,
    uploaded_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    KEY idx_user_image_user_id (user_id),
    CONSTRAINT fk_user_image_user
        FOREIGN KEY (user_id) REFERENCES USER(user_id)
        ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE SKIN_ANALYSIS_RESULT (
    result_id INT AUTO_INCREMENT PRIMARY KEY,
    image_id INT NOT NULL,
    user_id INT NOT NULL,
    acne_score DECIMAL(5,2) NULL,
    dryness_score DECIMAL(5,2) NULL,
    sagging_score DECIMAL(5,2) NULL,
    pore_score DECIMAL(5,2) NULL,
    pigmentation_score DECIMAL(5,2) NULL,
    wrinkle_score DECIMAL(5,2) NULL,
    analyzed_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    KEY idx_skin_result_image_id (image_id),
    KEY idx_skin_result_user_id (user_id),
    CONSTRAINT fk_skin_result_image
        FOREIGN KEY (image_id) REFERENCES USER_IMAGE(image_id)
        ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT fk_skin_result_user
        FOREIGN KEY (user_id) REFERENCES USER(user_id)
        ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- MY VANITY: SKIN MATCH
CREATE TABLE VANITY_MATCH_SESSION (
    match_session_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    result_id INT NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    KEY idx_vanity_match_user (user_id),
    KEY idx_vanity_match_result (result_id),
    CONSTRAINT fk_vanity_match_user
        FOREIGN KEY (user_id) REFERENCES USER(user_id)
        ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT fk_vanity_match_result
        FOREIGN KEY (result_id) REFERENCES SKIN_ANALYSIS_RESULT(result_id)
        ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE VANITY_MATCH_ITEM (
    match_item_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    match_session_id BIGINT NOT NULL,
    product_id BIGINT NOT NULL,
    vanity_fit_score DECIMAL(8,4) NULL,
    concern_match_score DECIMAL(8,4) NULL,
    skin_type_bonus DECIMAL(8,4) NULL,
    review_score DECIMAL(8,4) NULL,
    irritation_penalty DECIMAL(8,4) NULL,
    fit_label VARCHAR(20) NULL,
    recommend_action VARCHAR(20) NULL,
    reason_tags JSON NULL,
    caution_tags JSON NULL,
    KEY idx_vanity_match_item_session (match_session_id),
    KEY idx_vanity_match_item_product (product_id),
    CONSTRAINT fk_vanity_match_item_session
        FOREIGN KEY (match_session_id) REFERENCES VANITY_MATCH_SESSION(match_session_id)
        ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT fk_vanity_match_item_product
        FOREIGN KEY (product_id) REFERENCES PRODUCT(product_id)
        ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- RECOMMENDTATION
CREATE TABLE RECOMMENDATION_CANDIDATE (
    candidate_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    image_id INT NOT NULL,
    rank_in_category INT NOT NULL,
    product_id BIGINT NOT NULL,
    query_category VARCHAR(30) NOT NULL,
    score DECIMAL(12,8) NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    KEY idx_recommendation_candidate_image (image_id),
    KEY idx_recommendation_candidate_product (product_id),
    KEY idx_recommendation_candidate_category (query_category),
    KEY idx_recommendation_candidate_rank (rank_in_category),
    CONSTRAINT fk_recommendation_candidate_product
        FOREIGN KEY (product_id) REFERENCES PRODUCT(product_id)
        ON DELETE RESTRICT ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


CREATE TABLE RECOMMENDATION_SESSION (
    session_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    image_id INT NULL,
    result_id INT NULL,
    recommendation_type VARCHAR(30) NOT NULL DEFAULT 'basic',
    strict_budget BOOLEAN NOT NULL DEFAULT FALSE,
    total_budget_min INT NULL,
    total_budget_max INT NULL,
    slot_budget_min_json TEXT NULL,
    slot_budget_max_json TEXT NULL,
    budget_check_passed BOOLEAN NOT NULL DEFAULT TRUE,
    session_status VARCHAR(30) NOT NULL DEFAULT 'SUCCESS',
    failure_reason VARCHAR(100) NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    KEY idx_rec_session_user_id (user_id),
    KEY idx_rec_session_image_id (image_id),
    KEY idx_rec_session_result_id (result_id),
    KEY idx_rec_session_type (recommendation_type),
    CONSTRAINT fk_rec_session_user
        FOREIGN KEY (user_id) REFERENCES USER(user_id)
        ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT fk_rec_session_image
        FOREIGN KEY (image_id) REFERENCES USER_IMAGE(image_id)
        ON DELETE SET NULL ON UPDATE CASCADE,
    CONSTRAINT fk_rec_session_result
        FOREIGN KEY (result_id) REFERENCES SKIN_ANALYSIS_RESULT(result_id)
        ON DELETE SET NULL ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE RECOMMENDATION_RERANKED (
    reranked_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    session_id BIGINT NULL,
    user_id INT NOT NULL,
    image_id INT NULL,
    result_id INT NULL,
    product_id BIGINT NOT NULL,
    product_key VARCHAR(255) NULL,
    category VARCHAR(30) NOT NULL,
    brand_name VARCHAR(100) NULL,
    product_name VARCHAR(255) NULL,
    price INT NULL,
    embedding_rank INT NULL,
    embedding_score DECIMAL(12,8) NULL,
    rerank_rank_global INT NULL,
    rerank_rank_in_category INT NULL,
    rerank_score DECIMAL(8,4) NOT NULL,
    raw_rerank_score DECIMAL(8,4) NULL,
    vector_score DECIMAL(8,4) NULL,
    concern_score DECIMAL(8,4) NULL,
    skin_bonus DECIMAL(8,4) NULL,
    wishlist_bonus DECIMAL(8,4) NULL,
    review_score DECIMAL(8,4) NULL,
    irritation_penalty DECIMAL(8,4) NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    KEY idx_reranked_lookup (user_id, result_id, image_id),
    KEY idx_reranked_session (session_id),
    KEY idx_reranked_product (product_id),
    KEY idx_reranked_category_rank (category, rerank_rank_in_category),
    CONSTRAINT fk_reranked_session
        FOREIGN KEY (session_id) REFERENCES RECOMMENDATION_SESSION(session_id)
        ON DELETE SET NULL ON UPDATE CASCADE,
    CONSTRAINT fk_reranked_user
        FOREIGN KEY (user_id) REFERENCES USER(user_id)
        ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT fk_reranked_image
        FOREIGN KEY (image_id) REFERENCES USER_IMAGE(image_id)
        ON DELETE SET NULL ON UPDATE CASCADE,
    CONSTRAINT fk_reranked_result
        FOREIGN KEY (result_id) REFERENCES SKIN_ANALYSIS_RESULT(result_id)
        ON DELETE SET NULL ON UPDATE CASCADE,
    CONSTRAINT fk_reranked_product
        FOREIGN KEY (product_id) REFERENCES PRODUCT(product_id)
        ON DELETE RESTRICT ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE RECOMMENDATION_ROUTINE (
    routine_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    session_id BIGINT NOT NULL,
    routine_rank INT NOT NULL,
    routine_label VARCHAR(20) NULL,
    ampm_mode VARCHAR(10) NULL,
    routine_score DECIMAL(8,4) NULL,
    has_conflict BOOLEAN NOT NULL DEFAULT FALSE,
    conflict_pairs TEXT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    KEY idx_routine_session_id (session_id),
    UNIQUE KEY uq_routine_rank_per_session (session_id, routine_rank),
    CONSTRAINT fk_recommendation_routine_session
        FOREIGN KEY (session_id) REFERENCES RECOMMENDATION_SESSION(session_id)
        ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE RECOMMENDATION_ITEM (
    routine_item_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    routine_id BIGINT NOT NULL,
    slot_order INT NOT NULL,
    category VARCHAR(30) NULL,
    product_id BIGINT NULL,
    product_score DECIMAL(8,4) NULL,
    time_tag VARCHAR(10) NULL,
    source VARCHAR(20) NOT NULL DEFAULT 'recommendation',
    item_snapshot_json TEXT NULL,
    KEY idx_item_routine_id (routine_id),
    KEY idx_item_product_id (product_id),
    KEY idx_item_source (source),
    UNIQUE KEY uq_item_slot_per_routine (routine_id, slot_order),
    CONSTRAINT fk_recommendation_item_routine
        FOREIGN KEY (routine_id) REFERENCES RECOMMENDATION_ROUTINE(routine_id)
        ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT fk_recommendation_item_product
        FOREIGN KEY (product_id) REFERENCES PRODUCT(product_id)
        ON DELETE SET NULL ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
