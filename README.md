# 🤖 DocChatBot – Dynamic Document-Based AI Chatbot (Strict RAG)

## 🧠 Overview
**DocChatBot** is a dynamic, document-based AI chatbot that allows users to upload documents and ask questions that are answered **strictly from the content of the currently uploaded document**.

The chatbot is built using **Retrieval-Augmented Generation (RAG)** and is designed to:
- Prevent hallucinations
- Avoid cross-document data leakage
- Handle semantic questions (meaning-based, not keyword-based)
- Work entirely with **free and open-source tools**

---

## 🎯 Problem Statement
Users often need quick answers from large documents such as PDFs, Word files, or CSVs. Traditional chatbots may hallucinate answers or rely on general knowledge, which leads to incorrect or unsafe responses.

**Objective:**  
Build a chatbot that:
- Answers questions only from uploaded documents
- Rejects invalid or unrelated questions
- Dynamically adapts to new document uploads
- Supports multiple document formats
- Allows users to control answer length

---

## 👥 Target Users
- Students
- Interns
- Researchers
- Employees
- Anyone working with large documents

---

## ✨ Key Features
- 📂 Dynamic document upload (no hardcoded files)
- 🔄 New embeddings created for every upload
- 🚫 No answers from previously uploaded documents
- 🧠 Semantic search (meaning-based retrieval)
- ❌ Strict refusal for invalid or unrelated questions
- 🎚️ Brief and Long answer modes
- 📚 Source-grounded responses
- 🆓 Built using 100% free tools

---

## 📂 Supported Document Types
DocChatBot supports multiple document formats, including:

- 📄 PDF (`.pdf`)
- 📝 Word (`.docx`)
- 📊 CSV (`.csv`)
- 📃 Text files (`.txt`)
- 📋 Tables and structured data
- 📁 Multiple files per session

All documents are converted into text and embeddings before retrieval.

---

## 🏗️ System Architecture (RAG Flow)
1. User uploads document(s)
2. Documents are converted to text
3. Text is split into chunks
4. Chunks are converted into embeddings
5. A **new FAISS vector index** is created (old index discarded)
6. User asks a question
7. Relevant chunks are retrieved using semantic similarity
8. LLM generates an answer strictly from retrieved context

Each document upload starts a **new isolated session**.

---

## 🔁 Session Isolation (Important Design Choice)
- Every document upload creates a **new vector index**
- Previous embeddings are discarded
- The chatbot cannot answer questions from old documents
- Prevents cross-document contamination

This ensures correctness, safety, and real-world reliability.

---

## 🧩 Edge Case Handling

### 1️⃣ Semantic Match Without Keywords
Even if exact keywords are missing, the chatbot answers based on meaning.

**Example:**
- Question: *“How long is the internship?”*
- Document: *“The program runs for six months.”*

✅ Correct answer is generated.

---

### 2️⃣ Question Not Relevant to Document
If a question is unrelated to the uploaded document:

