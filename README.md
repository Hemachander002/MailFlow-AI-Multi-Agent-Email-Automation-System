# MailFlow AI — Multi-Agent Email Automation System

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Agent Orchestration | LangGraph |
| LLM / Classification | Google Gemini API |
| Local AI Models | Ollama |
| OCR | Tesseract OCR |
| Embeddings | Ollama Embedding Models |
| Email Integration | Gmail API |
| File Storage | Google Drive API |
| Database | PostgreSQL |
| Workflow Persistence | LangGraph PostgreSQL Checkpointer |
| Monitoring | Prometheus |
| Visualization | Grafana |
| Distributed Tracing | OpenTelemetry, Tempo |
| Containerization | Docker, Docker Compose |

An AI-powered email automation system that uses multiple specialized agents to classify incoming emails, process invoice and job application attachments, and automate downstream workflows.

To Run :
1. clone this repo
2. create a V_environment
3. setup ur .env file based on ur configuration
4. change the config values in docker-compose.yaml file based on ur config
5. run docker compose
6. run this docker exec -i ai-email-automation-postgres-1 psql -U POSTGRES_USER_NAME -d DB_NAME < init.sql
7. verify the tables in DBeaver or any other similar softwares
8. you have to train open source language model ( i chose Qwen/Qwen2.5-3B-Instruct) on https://www.kaggle.com/datasets/osamahosamabdellatif/high-quality-invoice-images-for-ocr this dataset
   1. loaded Qwen2.5-3B-Instruct using 4-bit NF4 quantization.
   2. Attach LoRA adapters
   3. convert the conversations into token IDs using Qwen's own chat template
   4. set MAX_LENGTH = 2048
   5. My Training Configuration ( for reference )
        Epochs	3
        Batch size	1
        Gradient accumulation	8
        Effective batch size	8
        Learning rate	0.0002
        Weight decay	0.01
        Warmup ratio	0.03
        Scheduler	Cosine
        Optimizer	Paged AdamW 8-bit
        Evaluation	Every epoch
    6. It took me 5 hours for fine-tuning [GPU : T4]
    7. save the PEFT adapter and tokenizer
    8. Set up llama.cpp repo
    9. The important tools for our workflow are:
      - convert_hf_to_gguf.py — converts a Hugging Face model into GGUF format.
      - llama-quantize — quantizes the converted model.
      - llama-cli — runs a model locally for testing.
    10. merge the local model with the adapters
    11. convert the merged model to gguf format
    12. quantize the gguf model
    13. test it with llama-cli before integrating with ollama
    14. run the model in OLLAMA
        
  9. keep the models in the paths instructed in the scripts

---
## Overview
Managing emails containing invoices and job applications can involve repetitive manual work, including identifying email types, downloading attachments, extracting information, and storing results.
**MailFlow AI** automates this process using an agent-based architecture.
The system monitors eligible Gmail inbox messages, classifies them using Google Gemini, and routes them to the appropriate processing pipeline.
Depending on the classification:
- **INVOICE:** Invoice attachments are downloaded, stored in Google Drive, processed using OCR and an Ollama-powered extraction model, and saved to PostgreSQL.
- **APPLICATION:** Resume PDFs are downloaded, stored in Google Drive, and passed to a specialized Resume Agent for information extraction, embedding generation, candidate-job matching, and scoring.
- **SPAM / OTHER:** Messages are skipped without invoking specialist agents.
- **REVIEW:** Messages are flagged for manual review.

## Key Features

### 1. Intelligent Email Classification 
- Integrates Google Gemini for email classification.
- Categorizes emails into INVOICE, APPLICATION, SPAM, OTHER, and REVIEW.
- Uses email metadata and relevant message content for classification.
- Routes messages automatically based on classification results.

### 2. Automated Invoice Processing
- Identifies invoice image attachments.
- Downloads and stages attachments locally.
- Uploads invoice files to Google Drive.
- Uses Tesseract OCR to extract text from invoice images.
- Uses an Ollama-powered local model to extract structured invoice information.
- Stores invoice information and original images in PostgreSQL.
- Supports duplicate detection using invoice numbers and SHA-256 hashes.

### 3. Automated Resume Processing
- Identifies PDF attachments from job application emails.
- Extracts resume content using PDF extraction and OCR utilities.
- Uses Gemini for candidate profile parsing.
- Generates embeddings using Ollama.
- Compares candidate information with job requirements using embedding similarity.
- Calculates skills, projects, experience, and overall matching scores.
- Stores candidate and application information in PostgreSQL.

### 4. Multi-Agent Orchestration
- Uses LangGraph to orchestrate the complete workflow.
- Routes emails to specialized Invoice and Resume Agents.
- Maintains workflow state across execution steps.
- Uses PostgreSQL-backed checkpointing.
- Processes emails individually to support controlled execution and recovery.

### 5. Reliability and Data Management
- Maintains an email-processing ledger in PostgreSQL.
- Implements duplicate detection and idempotent attachment uploads.
- Supports staged attachment downloads and checks for missing attachments.
- Uses Google Drive for attachment storage.

---
## Workflow
1. The system retrieves eligible messages from the Gmail Inbox.
2. The email-processing ledger checks whether a message has already been processed or claimed.
3. LangGraph initializes the workflow for an eligible email.
4. Gemini classifies the email.
5. The workflow routes the message to the appropriate specialist agent.
6. Relevant attachments are downloaded and uploaded to Google Drive.
7. The Invoice Agent or Resume Agent processes the attachments.
8. Extracted information and processing results are stored in PostgreSQL.
9. Workflow state and execution information are maintained through PostgreSQL checkpointing.
---




