
-- ============================================================
-- MailFlow AI - Database Initialization
-- PostgreSQL + pgvector
-- ============================================================

-- Required extensions
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================
-- 1. ATS TABLE
-- Stores job requirements and their embeddings
-- ============================================================

CREATE TABLE IF NOT EXISTS ats (
    job_id TEXT PRIMARY KEY DEFAULT 'DS-001',

    whole_embedding VECTOR,
    skills_embedding VECTOR,
    project_embedding VECTOR,
    experience_embedding VECTOR,

    fresher BOOLEAN NOT NULL DEFAULT FALSE,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    minimum_months INTEGER NOT NULL DEFAULT 0,
    preferred_months INTEGER NOT NULL DEFAULT 0
);


-- ============================================================
-- 2. CANDIDATE TABLE
-- Stores candidate information
-- ============================================================

CREATE TABLE IF NOT EXISTS candidate (
    candidate_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    email_id TEXT NOT NULL UNIQUE,
    phone_number TEXT,
    parsed_profile JSONB,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


-- ============================================================
-- 3. EMAIL PROCESSING TABLE
-- Tracks email workflow execution
-- ============================================================

CREATE TABLE IF NOT EXISTS email_processing (
    message_id TEXT PRIMARY KEY,

    thread_id TEXT,
    received_at TIMESTAMPTZ,

    category TEXT,
    confidence DOUBLE PRECISION,

    processing_status TEXT NOT NULL DEFAULT 'PENDING',
    current_step TEXT,
    error_message TEXT,

    retry_count INTEGER NOT NULL DEFAULT 0,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);


-- ============================================================
-- 4. HR TABLE
-- Stores candidate-job matching results and embeddings
-- ============================================================

CREATE TABLE IF NOT EXISTS hr (
    application_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    candidate_id UUID NOT NULL,
    job_id TEXT NOT NULL DEFAULT 'DS-001',

    resume_path TEXT,
    parsed_profile JSONB,

    whole_embedding VECTOR,
    skills_embedding VECTOR,
    project_embedding VECTOR,
    experience_embedding VECTOR,

    whole_score DOUBLE PRECISION,
    skills_score DOUBLE PRECISION,
    project_score DOUBLE PRECISION,
    experience_score DOUBLE PRECISION,

    suggestion TEXT,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_hr_candidate
        FOREIGN KEY (candidate_id)
        REFERENCES candidate(candidate_id),

    CONSTRAINT unique_candidate_job
        UNIQUE (candidate_id, job_id)
);


-- ============================================================
-- 5. USEFUL INDEXES
-- ============================================================

CREATE INDEX IF NOT EXISTS idx_email_processing_status
    ON email_processing(processing_status);

CREATE INDEX IF NOT EXISTS idx_email_processing_created_at
    ON email_processing(created_at);

CREATE INDEX IF NOT EXISTS idx_hr_job_id
    ON hr(job_id);

CREATE INDEX IF NOT EXISTS idx_hr_candidate_id
    ON hr(candidate_id);

CREATE INDEX IF NOT EXISTS idx_hr_created_at
    ON hr(created_at);


-- ============================================================
-- INITIALIZATION COMPLETE
-- ============================================================

