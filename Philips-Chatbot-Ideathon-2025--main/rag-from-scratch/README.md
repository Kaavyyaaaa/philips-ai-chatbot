# RAG From Scratch

LLMs are trained on a large but fixed corpus of data, limiting their ability to reason about private or recent information. Fine-tuning is one way to mitigate this, but is often [not well-suited for facutal recall](https://www.anyscale.com/blog/fine-tuning-is-for-form-not-facts) and [can be costly](https://www.glean.com/blog/how-to-build-an-ai-assistant-for-the-enterprise).
Retrieval augmented generation (RAG) has emerged as a popular and powerful mechanism to expand an LLM's knowledge base, using documents retrieved from an external data source to ground the LLM generation via in-context learning. 
These notebooks accompany a [video playlist](https://youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x&feature=shared) that builds up an understanding of RAG from scratch, starting with the basics of indexing, retrieval, and generation. 
![rag_detail_v2](https://github.com/langchain-ai/rag-from-scratch/assets/122662504/54a2d76c-b07e-49e7-b4ce-fc45667360a1)
 
---

# Philips AI Chatbot — Project Implementation

## Overview

The **Philips AI Chatbot** is an AI-powered FAQ assistant developed as part of the Philips Chatbot Ideathon 2025. It uses Retrieval-Augmented Generation (RAG) to answer questions using relevant information from a local FAQ knowledge base and, when configured, Philips-related web search results.

The application is designed to handle questions across three main domains:

- **Healthcare:** Medical devices, healthcare solutions, and clinical topics.
- **Consumer Products:** Electronics, personal care, home appliances, and lifestyle products.
- **Careers and Recruitment:** Job applications, recruitment processes, interviews, and career-related queries.

The goal is to provide contextual, topic-relevant answers rather than relying entirely on the language model's pretrained knowledge.

> **Disclaimer:** This is an independent educational project created for an ideathon. It is not an official Philips product or an endorsed Philips service.

## Key Features

- **Retrieval-Augmented Generation (RAG):** Retrieves relevant FAQ information before generating answers.
- **Semantic Search:** Uses sentence embeddings and FAISS to identify relevant text passages.
- **LLM Integration:** Uses Google Gemini through LangChain to generate natural-language responses.
- **Topic Detection:** Categorizes questions into healthcare, consumer products, recruitment, or general topics.
- **Topic-Aware Retrieval:** Filters retrieved documents and web results according to their relevance to the question.
- **Web Search Integration:** Uses Google Custom Search API to retrieve relevant results from Philips domains when credentials are configured.
- **Confidence Indicator:** Calculates a heuristic confidence score based on the overlap between query terms and retrieved information.
- **Follow-Up Questions:** Structures responses to suggest relevant next questions.
- **Interactive Web Interface:** Provides a browser-based chatbot using HTML, CSS, and JavaScript.
- **Audio Transcription:** Includes an endpoint that uses AssemblyAI to transcribe uploaded audio.

## Technology Stack

| Component | Technology |
|---|---|
| Backend | Python, FastAPI |
| Language Model | Google Gemini (`gemini-1.5-flash`) |
| LLM Orchestration | LangChain |
| Text Embeddings | Hugging Face Sentence Transformers (`all-MiniLM-L6-v2`) |
| Vector Database | FAISS |
| Knowledge Base | Local FAQ text file |
| Web Search | Google Custom Search API |
| Frontend | HTML, CSS, JavaScript |
| Templates | Jinja2 |
| Audio Transcription | AssemblyAI API |
| Configuration | Python dotenv |
| API Server | Uvicorn |

## Architecture and Workflow

The application follows a retrieval-and-generation workflow.

1. **User Query:** The user submits a question through the web interface.
2. **Topic Classification:** Keyword-based rules identify the likely topic of the query.
3. **Semantic Retrieval:** The query is compared with embedded FAQ passages stored in FAISS.
4. **Document Filtering:** Retrieved passages are filtered according to topic relevance.
5. **Web Search:** When configured, Google Custom Search retrieves additional results restricted to Philips domains.
6. **Context Preparation:** Relevant information is assembled into context for the language model.
7. **Answer Generation:** Gemini generates a response using the supplied context and instructions.
8. **Response Formatting:** The application formats the answer and returns available confidence information, sources, and follow-up questions to the frontend.

### Architecture Diagram

```text
                  User
                   |
                   v
            Chat Web Interface
                   |
                   v
              FastAPI API
                   |
                   v
             Topic Detection
                   |
                   v
          Semantic FAQ Retrieval
                   |
                   v
             FAISS Index
                   |
                   v
          Topic-Relevant Context
                   |
          +--------+--------+
          |                 |
          v                 v
      FAQ Results      Optional Web Search
          |                 |
          +--------+--------+
                   |
                   v
             Google Gemini
                   |
                   v
       Answer + Confidence + Sources
                   |
                   v
            Chat Web Interface
```

## Retrieval-Augmented Generation

RAG combines information retrieval with language generation. Instead of asking the language model to answer using only its pretrained knowledge, the application first searches for relevant information and supplies it as context.

This implementation uses the following components:

- **Document chunking:** `RecursiveCharacterTextSplitter` divides FAQ content into smaller passages using a configured chunk size of 300 characters and an overlap of 30 characters.
- **Embedding generation:** `all-MiniLM-L6-v2` converts text into numerical vector representations.
- **Vector search:** FAISS retrieves the three most similar FAQ documents for a query.
- **Relevance filtering:** Topic-specific keywords help filter retrieved documents.
- **Context-grounded generation:** Gemini receives the question and retrieved context, with instructions to acknowledge when the available information is insufficient.

This approach helps make answers more relevant to the supplied knowledge base. However, retrieval and confidence scoring do not guarantee that every answer is correct.

## Topic Detection

The application uses weighted keywords and topic-specific phrases to estimate the category of a query.

| Topic | Example subject areas |
|---|---|
| Healthcare | Medical equipment, patients, diagnostics, clinical solutions |
| Consumer | Electronics, home appliances, personal care, lifestyle |
| Recruitment | Job openings, interviews, applications, hiring |
| General | Questions that do not clearly match a specific topic |

The detected topic guides document filtering and web-result relevance scoring. This helps reduce unrelated context being passed to the language model.

## Project Structure

```text
rag-from-scratch/
├── backend.py
├── requirements.txt
├── faqs.txt
├── faiss_index/
│   ├── index.faiss
│   └── index.pkl
├── templates/
│   └── index.html
├── static/
│   ├── philips-logo.png
│   ├── script.js
│   └── style.css
├── test_gemini.py
├── rag_from_scratch_1_to_4.ipynb
├── rag_from_scratch_5_to_9.ipynb
├── rag_from_scratch_10_and_11.ipynb
├── rag_from_scratch_12_to_14.ipynb
└── rag_from_scratch_15_to_18.ipynb
```

| File or directory | Purpose |
|---|---|
| `backend.py` | FastAPI application, retrieval logic, topic detection, web search, and response generation |
| `faqs.txt` | Local FAQ knowledge base |
| `faiss_index/` | Persisted vector index and associated serialized metadata |
| `templates/index.html` | Chat interface template |
| `static/script.js` | Frontend interactions and API communication |
| `static/style.css` | Interface styling |
| `test_gemini.py` | Gemini connectivity test |
| `requirements.txt` | Python package dependencies |
| Jupyter notebooks | RAG concepts, experimentation, and learning exercises |

## Installation and Setup

### Prerequisites

- Python 3.10 or a compatible Python version
- pip
- Internet access for API calls and initial model downloads
- A Google Gemini API key
- Google Custom Search credentials for web search
- An AssemblyAI API key for audio transcription

### 1. Clone the repository

```bash
git clone https://github.com/Kaavyaaaaa/philips-ai-chatbot.git
cd philips-ai-chatbot/Philips-Chatbot-Ideathon-2025--main/rag-from-scratch
```

If you have organized the files differently, navigate to the directory containing `backend.py`, `faqs.txt`, `requirements.txt`, `templates/`, and `static/`.

### 2. Create a virtual environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install langchain-google-genai
```

The backend imports `langchain_google_genai`, so `langchain-google-genai` must also be installed. Add it to `requirements.txt` if it is not already listed.

### 4. Configure environment variables

Create a `.env` file in the project directory:

```dotenv
GOOGLE_API_KEY=your_gemini_api_key
SERP_API_KEY=your_google_custom_search_api_key
SEARCH_ENGINE_ID=your_search_engine_id
ASSEMBLYAI_API_KEY=your_assemblyai_api_key
```

Replace the example values with your own credentials. Never commit real API keys or the `.env` file to GitHub.

### 5. Run the application

From the directory containing `backend.py`, run:

```bash
uvicorn backend:app --reload
```

Open the chatbot in your browser:

**http://127.0.0.1:8000**

FastAPI's interactive API documentation is available at:

**http://127.0.0.1:8000/docs**

## API Endpoints

### `GET /`

Serves the chatbot interface using the Jinja2 template in `templates/index.html`.

### `POST /chat`

Accepts a question and returns a generated response with associated metadata.

Example request:

```json
{
  "query": "How can I find information about Philips careers?"
}
```

Example request using curl:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"How can I find information about Philips careers?\"}"
```

The response structure depends on the backend implementation and may include the answer, confidence information, retrieved sources, web sources, and follow-up questions.

### `POST /transcribe`

Accepts an uploaded audio file and uses the AssemblyAI API to obtain a transcription. An `ASSEMBLYAI_API_KEY` is required.

## Knowledge Base and Index Management

The FAQ data is stored in `faqs.txt`. At application startup, the backend reads the FAQ content, splits it into chunks, and initializes the embedding model.

If a saved FAISS index is available, the application loads it. Otherwise, it creates and saves a new index.

When the FAQ content changes, ensure the vector index is rebuilt so it reflects the updated information. The existing implementation loads a saved index when present, so changing `faqs.txt` alone may not update the indexed content.

**Security note:** The application loads the FAISS index with pickle deserialization enabled. Only load index files from trusted sources.

## Limitations

- Answer quality depends on the completeness and accuracy of the FAQ data and retrieved web results.
- Topic detection relies on keyword-based rules and may misclassify ambiguous questions.
- The confidence score is a heuristic based on word overlap; it is not a calibrated probability of correctness.
- Web search and audio transcription require valid external API credentials.
- The current development CORS configuration allows all origins and should be restricted before production deployment.
- The application requires additional testing and security hardening before public production use.

## Future Improvements

- Improve topic classification using a trained classifier or LLM-based routing.
- Add automated tests for API behavior, retrieval relevance, and answer quality.
- Improve source citations and direct links in generated answers.
- Rebuild the vector index automatically when FAQ data changes.
- Add evaluation metrics for retrieval accuracy, answer relevance, and groundedness.
- Improve validation and error handling for missing API credentials.
- Restrict CORS, add rate limiting, and review frontend HTML rendering.
- Pin compatible dependency versions and add deployment documentation.

## Acknowledgements

This project builds on concepts explored in the RAG-from-scratch notebooks and uses open-source tools and APIs, including FastAPI, LangChain, Hugging Face Sentence Transformers, and FAISS, alongside Google Gemini, Google Custom Search, and AssemblyAI integrations.

---

**Project:** Philips Chatbot Ideathon 2025  
**Repository:** [philips-ai-chatbot](https://github.com/Kaavyaaaaa/philips-ai-chatbot)
