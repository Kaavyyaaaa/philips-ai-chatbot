from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
from pathlib import Path
import re
from typing import List, Dict, Tuple
import numpy as np
import aiohttp
import json
import aiofiles
import asyncio
import httpx
import tempfile
from datetime import datetime

load_dotenv()
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configure templates
templates = Jinja2Templates(directory="templates")

# Custom prompt template
prompt_template = """You are a helpful Philips FAQ assistant. Use the following pieces of context to answer the question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Instructions:
1. First, verify that the provided context is relevant to the question. Only use information that directly relates to the question topic.
2. Format your response exactly like this:

ANSWER: [Provide your complete answer here. If mentioning phases or steps, you MUST list and explain ALL of them before moving to other sections.]
- When describing processes or phases, list each one with a clear explanation
- Use bullet points for lists
- Highlight *important terms* with asterisks
- Include any relevant links or references
- Make sure to complete your thoughts and sentences
- NEVER stop in the middle of listing items or phases

[!WARNING] prefix for any important warnings or notices.

FOLLOW_UP: [3 relevant follow-up questions, separated by |]

3. Keep the tone professional and helpful
4. If information is incomplete, acknowledge what's missing
5. Format warnings or important notices with [!WARNING] prefix
6. DO NOT use sources that are not directly related to the question topic
7. IMPORTANT: Always complete your answer FULLY before showing sources or other sections
8. When describing a process with multiple phases:
   - List all phases first
   - Then provide details for each phase
   - Complete the entire description before moving to other sections

Example format for processes:
ANSWER: The process has [X] phases:

1. First Phase: [Complete description]
2. Second Phase: [Complete description]
3. Third Phase: [Complete description]

[Any additional details or notes]

Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# Web search configuration
SERP_API_KEY = os.getenv("SERP_API_KEY")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")

# Enhanced topic configuration with weighted keywords
TOPIC_CONFIGS = {
    'healthcare': {
        'search_terms': 'healthcare medical devices solutions products philips',
        'relevant_keywords': [
            ('healthcare', 2.0), ('medical', 2.0), ('patient', 1.5),
            ('hospital', 1.5), ('diagnostic', 1.5), ('clinical', 1.5),
            ('treatment', 1.0), ('device', 1.0), ('solution', 0.5)
        ]
    },
    'consumer': {
        'search_terms': 'consumer products electronics personal care philips',
        'relevant_keywords': [
            ('consumer', 2.0), ('electronics', 1.5), ('personal', 1.0),
            ('home', 1.0), ('lifestyle', 1.5), ('appliance', 1.5)
        ]
    },
    'recruitment': {
        'search_terms': 'careers jobs recruitment hiring philips',
        'relevant_keywords': [
            ('career', 2.0), ('job', 2.0), ('recruitment', 2.0),
            ('hiring', 1.5), ('position', 1.0), ('vacancy', 1.0),
            ('employment', 1.5), ('work', 0.5), ('recruiter', 2.0),
            ('application', 1.0), ('apply', 1.0), ('interview', 1.5),
            ('candidate', 1.0), ('resume', 1.0), ('cv', 1.0), ('talent', 1.0)
        ]
    }
}

# Topic phrases for stronger matching
TOPIC_PHRASES = {
    'recruitment': [
        'recruitment process', 'job application', 'career', 'recruiter', 
        'hiring', 'job opening', 'employment', 'interview process',
        'suspicious communication', 'verify recruiter', 'legitimate recruiter'
    ],
    'healthcare': [
        'healthcare device', 'medical equipment', 'patient monitor',
        'clinical', 'hospital', 'diagnostic'
    ],
    'consumer': [
        'consumer product', 'home appliance', 'personal care',
        'electronics', 'lifestyle'
    ]
}

def determine_topic(query: str) -> Tuple[str, float]:
    """
    Determine the topic of a query based on phrases and weighted keywords.
    Returns a tuple of (topic, confidence_score).
    """
    query_lower = query.lower()
    
    # First, check for specific phrases that strongly indicate a topic
    for topic, phrases in TOPIC_PHRASES.items():
        if any(phrase in query_lower for phrase in phrases):
            return topic, 1.0
    
    # Calculate weighted scores for each topic
    topic_scores = {}
    query_words = set(query_lower.split())
    
    for topic, config in TOPIC_CONFIGS.items():
        score = 0.0
        total_weight = 0.0
        
        for keyword, weight in config['relevant_keywords']:
            if keyword in query_lower:
                score += weight
            total_weight += weight
        
        # Normalize score
        topic_scores[topic] = score / total_weight if total_weight > 0 else 0
    
    # Get the topic with highest score
    best_topic = max(topic_scores.items(), key=lambda x: x[1])
    
    # Return general if confidence is too low
    if best_topic[1] < 0.2:
        return 'general', 0.0
    
    return best_topic

def filter_search_results(results: List[dict], query_topic: str, confidence: float) -> List[dict]:
    """
    Filter and rank search results based on topic relevance and other factors.
    """
    if not results:
        return []

    filtered_results = []
    
    # Define topic exclusion rules
    topic_exclusions = {
        'recruitment': ['healthcare', 'medical', 'patient', 'clinical'],
        'healthcare': ['job', 'career', 'vacancy', 'apply'],
        'consumer': ['job', 'career', 'patient', 'clinical']
    }
    
    # Get exclusion keywords for the current topic
    exclusion_keywords = topic_exclusions.get(query_topic, [])
    
    for result in results:
        title = result.get('title', '').lower()
        snippet = result.get('snippet', '').lower()
        content = f"{title} {snippet}"
        
        # For general topic, skip exclusion check and use basic relevance scoring
        if query_topic == 'general':
            relevance_score = 1.0
            # Boost score for specific indicators of relevance
            if 'location' in content or 'office' in content or 'headquarters' in content:
                relevance_score *= 1.5
            if 'philips.com' in result.get('link', '').lower():
                relevance_score *= 1.2
            
            result['relevance_score'] = relevance_score
            filtered_results.append(result)
            continue
        
        # For specific topics, apply exclusion rules
        if any(kw in content for kw in exclusion_keywords):
            continue
            
        # Calculate relevance score
        relevance_score = 0.0
        
        # Check for topic-specific keywords
        if query_topic in TOPIC_CONFIGS:
            for keyword, weight in TOPIC_CONFIGS[query_topic]['relevant_keywords']:
                if keyword in content:
                    relevance_score += weight
        
        # Boost score for Philips domain
        if 'philips.com' in result.get('link', '').lower():
            relevance_score *= 1.5
            
        # Only include results with minimum relevance
        if relevance_score > 0.5 or query_topic == 'general':
            result['relevance_score'] = relevance_score
            filtered_results.append(result)
    
    # Sort by relevance score
    filtered_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
    
    print(f"Filtered {len(results)} results to {len(filtered_results)} relevant results")
    if filtered_results:
        print("Top result preview:", {
            'title': filtered_results[0].get('title', '')[:100],
            'score': filtered_results[0].get('relevance_score', 0)
        })
    
    return filtered_results

async def web_search(params: dict) -> List[dict]:
    """Perform a web search using Google Custom Search API."""
    try:
        if not SERP_API_KEY or not SEARCH_ENGINE_ID:
            print("Error: Missing SERP_API_KEY or SEARCH_ENGINE_ID in .env file")
            return []

        search_term = params["search_term"]
        query_topic = params.get("topic", "general")
        topic_confidence = params.get("confidence", 0.0)
        
        print(f"\nPerforming web search with:")
        print(f"- Query: {search_term}")
        print(f"- Topic: {query_topic}")
        print(f"- API Key: {SERP_API_KEY[:4]}...{SERP_API_KEY[-4:]}")
        print(f"- Search Engine ID: {SEARCH_ENGINE_ID}")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            search_params = {
                "key": SERP_API_KEY,
                "cx": SEARCH_ENGINE_ID,
                "q": search_term,
                "num": 10,
                "siteSearch": "philips.com",  # Restrict to Philips domains
                "siteSearchFilter": "i"  # Include only Philips domains
            }
            
            response = await client.get(
                "https://www.googleapis.com/customsearch/v1",
                params=search_params
            )
            
            print(f"Search API Response Status: {response.status_code}")
            
            if response.status_code != 200:
                print(f"API Error Response: {response.text}")
                return []
            
            data = response.json()
            
            if "error" in data:
                print(f"API Error: {json.dumps(data['error'], indent=2)}")
                return []
            
            if "items" not in data:
                print("No search results found")
                return []
            
            results = data["items"]
            print(f"Found {len(results)} raw results")
            
            # Apply enhanced filtering and ranking
            filtered_results = filter_search_results(results, query_topic, topic_confidence)
            
            print(f"Returning {len(filtered_results)} filtered results")
            return filtered_results[:5]  # Return top 5 most relevant results

    except httpx.TimeoutException:
        print("Search API request timed out")
        return []
    except httpx.RequestError as e:
        print(f"Search API request failed: {str(e)}")
        return []
    except Exception as e:
        print(f"Unexpected error in web search: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return []

def calculate_confidence_score(documents, query, is_greeting=False):
    """
    Calculate confidence score based on document relevance to query.
    Returns a dict with score, level, and reason.
    """
    # Special case for greeting
    if is_greeting:
        return {
            "score": 100,
            "level": "High",
            "reason": "Standard greeting response"
        }
    
    if not documents:
        return {
            "score": 0,
            "level": "Low",
            "reason": "No relevant information found in the knowledge base"
        }
    
    # Normalize text for comparison
    query = query.lower()
    docs_text = " ".join(documents).lower()
    
    # Split into words (excluding common words)
    common_words = {"what", "is", "are", "the", "in", "on", "at", "to", "for", "of", "with", "by"}
    query_words = set(w for w in query.split() if w not in common_words)
    doc_words = set(w for w in docs_text.split() if w not in common_words)
    
    if not query_words:
        return {
            "score": 0,
            "level": "Low",
            "reason": "Query too generic"
        }
    
    # Calculate word overlap ratio
    overlap = len(query_words.intersection(doc_words))
    coverage_ratio = overlap / len(query_words) if query_words else 0
    
    # Look for exact phrases (3+ words) in the documents
    query_phrases = [p for p in query.split(".") if len(p.split()) >= 3]
    exact_matches = 0
    for phrase in query_phrases:
        if phrase.strip() in docs_text:
            exact_matches += 1
    
    # Calculate final score
    base_score = coverage_ratio * 100
    if exact_matches > 0:
        base_score = min(100, base_score * 1.5)  # Boost score for exact matches
    
    score = min(100, base_score)
    
    # Determine confidence level and reason
    if score > 70:
        level = "High"
        reason = "Strong match found in Philips documentation"
    elif score > 40:
        level = "Medium"
        reason = "Partial information found in documentation"
    else:
        level = "Low"
        reason = "Limited or no relevant information found"
    
    return {
        "score": round(score),
        "level": level,
        "reason": reason
    }

def format_answer(text: str) -> str:
    """Format the answer with enhanced styling."""
    # Convert plain hyphens to bullet points
    text = re.sub(r'(?m)^- ', '• ', text)
    
    # Format warnings
    text = re.sub(r'\[!WARNING\](.*?)(?=\n|$)', r'<div class="warning-box">\1</div>', text)
    
    # Highlight important terms
    text = re.sub(r'\*(.*?)\*', r'<strong>\1</strong>', text)
    
    # Convert URLs to links
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
    text = re.sub(url_pattern, lambda m: f'<a href="{m.group(0)}" target="_blank">{m.group(0)}</a>', text)
    
    return text

# Load and process FAQ data
try:
    print("Loading FAQ data...")
    with open("faqs.txt", "r", encoding="utf-8") as f:
        faq_data = f.read()
    print(f"Loaded {len(faq_data)} characters of FAQ data")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,  # Reduced from 500 to get more focused chunks
        chunk_overlap=30,  # Reduced overlap to minimize context mixing
        separators=["\n\nQ:", "\nQ:", "\n\n", "\n", " ", ""]  # Added Q: as a separator to better split questions
    )
    texts = text_splitter.split_text(faq_data)
    print(f"Split into {len(texts)} text chunks")
    print("First chunk sample:", texts[0][:100])

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    faiss_index_path = "faiss_index"

    if os.path.exists(faiss_index_path):
        print("Loading existing FAISS index...")
        vector_store = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
        print("Successfully loaded FAISS index")
    else:
        print("Creating new FAISS index...")
        vector_store = FAISS.from_texts(texts, embeddings)
        vector_store.save_local(faiss_index_path)
        print("Successfully created and saved new FAISS index")

    # Test vector store retrieval
    test_query = "recruitment process"
    print(f"\nTesting vector store with query: {test_query}")
    test_docs = vector_store.similarity_search(test_query, k=1)
    if test_docs:
        print("Vector store test successful. Sample result:", test_docs[0].page_content[:100])
    else:
        print("Vector store test returned no results!")

except Exception as e:
    print(f"Error loading FAQ data: {str(e)}")
    import traceback
    print(f"Traceback: {traceback.format_exc()}")
    raise

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

def filter_relevant_documents(docs, query):
    """Filter documents to only include those relevant to the query topic."""
    print(f"\nFiltering documents for query: {query}")
    print(f"Input documents: {len(docs)}")
    
    # Determine query topic
    query_topic, confidence = determine_topic(query)
    print(f"Detected topic: {query_topic} with confidence: {confidence}")
    
    # Filter documents based on topic relevance
    filtered_docs = []
    for doc in docs:
        doc_lower = doc.page_content.lower()
        # Check if document matches query topic
        is_relevant = False
        if query_topic == 'general':
            is_relevant = True
        else:
            # Extract just the keywords from the weighted tuples
            topic_keywords = [keyword for keyword, _ in TOPIC_CONFIGS[query_topic]['relevant_keywords']]
            if any(keyword in doc_lower for keyword in topic_keywords):
                is_relevant = True
        
        if is_relevant:
            filtered_docs.append(doc)
            print(f"Found relevant document: {doc.page_content[:100]}...")
    
    result = filtered_docs or docs[:1]  # Return at least one doc if all filtered out
    print(f"Returning {len(result)} filtered documents")
    return result

class CustomRetrievalQA(RetrievalQA):
    async def _call(self, inputs):
        try:
            print(f"\nProcessing query: {inputs['query']}")
            
            # Use direct similarity search instead of ainvoke
            print("Performing similarity search...")
            docs = self.retriever.vectorstore.similarity_search(
                inputs["query"],
                k=3
            )
            print(f"Retrieved {len(docs)} documents from similarity search")
            
            # Filter documents for relevance
            filtered_docs = filter_relevant_documents(docs, inputs["query"])
            print(f"Filtered to {len(filtered_docs)} relevant documents")
            
            # Update the context with filtered documents
            context = "\n\n".join(doc.page_content for doc in filtered_docs)
            
            # Calculate initial confidence based on local FAQs
            initial_confidence = calculate_confidence_score([doc.page_content for doc in filtered_docs], inputs["query"])
            
            # If confidence is low, augment with web search
            if initial_confidence["score"] < 25:  # Low confidence threshold
                print("Low confidence in local FAQs, performing web search...")
                try:
                    web_results = await web_search({"search_term": f"Philips {inputs['query']}", "explanation": "Augmenting FAQ knowledge with web search"})
                    if web_results and isinstance(web_results, list):
                        # Add relevant web results to context
                        web_context = "\n\n".join(result.get("snippet", "") for result in web_results if "snippet" in result)
                        if web_context:
                            context = f"{context}\n\nAdditional information from Philips website:\n{web_context}"
                            print("Successfully augmented context with web search results")
                except Exception as e:
                    print(f"Web search failed: {str(e)}")
            
            inputs["context"] = context
            print(f"Final context length: {len(context)} characters")
            
            # Use ainvoke for the chain
            chain_response = await self.chain.ainvoke(inputs)
            print(f"Chain response: {chain_response}")
            
            # Add source attribution for web results if used
            if "web_results" in locals() and web_results:
                chain_response["web_sources"] = [
                    {"title": result.get("title", ""), "url": result.get("link", "")}
                    for result in web_results if "title" in result and "link" in result
                ]
            
            return chain_response
        except Exception as e:
            print(f"Error in CustomRetrievalQA: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            raise

# Update the chain creation with more specific parameters
qa_chain = CustomRetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 3,
        }
    ),
    return_source_documents=True,
    chain_type_kwargs={
        "prompt": PROMPT,
        "verbose": True,
        "document_separator": "\n\n",  # Add clear separation between documents
    }
)

class ChatRequest(BaseModel):
    query: str

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

async def process_web_results(search_results: List[dict]) -> str:
    """
    Process web search results into a format suitable for RAG.
    """
    if not search_results:
        return ""
    
    processed_text = []
    for result in search_results:
        processed_text.append(f"Title: {result['title']}\nURL: {result['link']}\nSummary: {result['snippet']}\n")
    
    return "\n".join(processed_text)

@app.post("/chat")
async def chat(request: ChatRequest):
    query = request.query.strip()
    
    # Check if it's a greeting
    greetings = {"hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"}
    is_greeting = query.lower() in greetings
    
    if is_greeting:
        return {
            "answer": "Hello! I'm your Philips FAQ assistant. How can I help you today?",
            "confidence": calculate_confidence_score([], query, is_greeting=True),
            "sources": []
        }
    
    try:
        # Get relevant documents from FAISS
        docs = vector_store.similarity_search(query, k=3)
        relevant_docs = filter_relevant_documents(docs, query)
        
        # Determine the topic and get topic-specific search terms
        current_topic, topic_confidence = determine_topic(query)
        print(f"\nDetected topic: {current_topic} with confidence: {topic_confidence}")
        
        # Construct search query
        search_query = f"site:philips.com {query}"
        print(f"Web search query: {search_query}")
        
        # Perform web search with enhanced parameters
        web_results = await web_search({
            "search_term": search_query,
            "topic": current_topic,
            "confidence": topic_confidence
        })
        
        # Process web results
        web_sources = []
        if web_results:
            print(f"Processing {len(web_results)} web results")
            for result in web_results:
                web_sources.append({
                    "title": result.get("title", ""),
                    "url": result.get("link", ""),
                    "snippet": result.get("snippet", "")
                })
            print(f"Added {len(web_sources)} web sources")
        else:
            print("No web results to process")

        # Combine local and web knowledge
        all_contexts = [doc.page_content for doc in relevant_docs]
        web_sources = []

        if web_results:
            print(f"Found {len(web_results)} web results")
            web_context = []
            relevant_results = []
            
            # Filter web results based on topic relevance
            topic_keywords = [keyword for keyword, _ in TOPIC_CONFIGS.get(current_topic, {}).get('relevant_keywords', [])]
            
            for result in web_results:
                content = (result.get('title', '') + ' ' + result.get('snippet', '')).lower()
                
                # For recruitment topics, be more strict about relevance
                if current_topic == 'recruitment':
                    is_relevant = any(keyword in content for keyword in ['career', 'job', 'recruitment', 'hiring', 'employment'])
                else:
                    is_relevant = not topic_keywords or any(keyword in content for keyword in topic_keywords)
                
                if is_relevant:
                    web_context.append(f"From {result['title']}: {result['snippet']}")
                    relevant_results.append({
                        "title": result["title"],
                        "url": result["link"],
                        "snippet": result["snippet"]
                    })
            
            if web_context:
                all_contexts.extend(web_context)
                web_sources = relevant_results
                print(f"Added {len(web_context)} relevant web results to context")
            else:
                print("No relevant web results found for the topic:", current_topic)
        
        combined_context = "\n\n".join(all_contexts)
        print(f"Combined context length: {len(combined_context)}")
        
        # Calculate confidence score
        confidence = calculate_confidence_score(all_contexts, query)
        
        if not combined_context:
            return {
                "answer": "I apologize, but I couldn't find any relevant information to answer your question accurately. Could you please rephrase your question or ask about a different topic?",
                "confidence": confidence,
                "sources": []
            }
        
        # Use the existing qa_chain instance
        response = await qa_chain.ainvoke({
            "query": query,
            "context": combined_context
        })
        
        # Extract answer and follow-up questions
        answer_text = response["result"]
        follow_up_questions = []
        
        # Clean up the answer text
        if "ANSWER:" in answer_text:
            parts = answer_text.split("ANSWER:")
            answer_text = parts[1].strip()
            
            # Remove any remaining CONFIDENCE: sections
            if "CONFIDENCE:" in answer_text:
                answer_text = answer_text.split("CONFIDENCE:")[0].strip()
            
            # Extract follow-up questions if they exist
            if "FOLLOW_UP:" in answer_text:
                parts = answer_text.split("FOLLOW_UP:")
                answer_text = parts[0].strip()
                if len(parts) > 1:
                    questions = parts[1].strip().split("|")
                    follow_up_questions = [q.strip() for q in questions if q.strip()]
        
        # Clean up FAQ sources before returning
        cleaned_sources = []
        for doc in relevant_docs:
            # Extract the question from the content if it exists
            content = doc.page_content
            question = ""
            if content.startswith("Q."):
                parts = content.split("\nA.", 1)
                if len(parts) > 1:
                    question = parts[0].replace("Q.", "").strip()
            
            # If no question found, use first line or part of content
            if not question:
                question = content.split("\n")[0][:100] + "..."
            
            cleaned_sources.append({
                "question": question,
                "content": content[:200] + "..." if len(content) > 200 else content  # Truncate long content
            })

        return {
            "answer": format_answer(answer_text),
            "confidence": confidence,
            "sources": cleaned_sources,
            "web_sources": web_sources,
            "follow_up_questions": follow_up_questions
        }
        
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return {
            "answer": "I apologize, but I encountered an error while processing your request. Please try again.",
            "confidence": {"score": 0, "level": "Low", "reason": f"Error: {str(e)}"},
            "sources": []
        }

# AssemblyAI configuration
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
ASSEMBLYAI_API_URL = "https://api.assemblyai.com/v2"

@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    if not ASSEMBLYAI_API_KEY:
        raise HTTPException(status_code=500, detail="AssemblyAI API key not configured")

    try:
        # Save the uploaded file temporarily
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, f"audio_{datetime.now().timestamp()}.webm")
        
        async with aiofiles.open(temp_file_path, 'wb') as out_file:
            content = await audio.read()
            await out_file.write(content)

        # Upload to AssemblyAI
        headers = {
            "authorization": ASSEMBLYAI_API_KEY,
            "content-type": "application/json"
        }

        async with httpx.AsyncClient() as client:
            # Upload the audio file
            print("Uploading audio file to AssemblyAI...")
            async with aiofiles.open(temp_file_path, 'rb') as audio_file:
                audio_data = await audio_file.read()
                upload_response = await client.post(
                    f"{ASSEMBLYAI_API_URL}/upload",
                    headers=headers,
                    content=audio_data
                )
            
            if upload_response.status_code != 200:
                raise Exception(f"Error uploading audio file: {upload_response.text}")

            upload_url = upload_response.json()["upload_url"]
            
            # Start transcription
            print("Starting transcription...")
            transcript_response = await client.post(
                f"{ASSEMBLYAI_API_URL}/transcript",
                json={"audio_url": upload_url},
                headers=headers
            )
            
            if transcript_response.status_code != 200:
                raise Exception(f"Error starting transcription: {transcript_response.text}")

            transcript_id = transcript_response.json()["id"]
            
            # Poll for transcription completion
            print("Polling for transcription completion...")
            while True:
                polling_response = await client.get(
                    f"{ASSEMBLYAI_API_URL}/transcript/{transcript_id}",
                    headers=headers
                )
                
                if polling_response.status_code != 200:
                    raise Exception(f"Error polling transcription: {polling_response.text}")
                
                polling_data = polling_response.json()
                print(f"Transcription status: {polling_data['status']}")
                
                if polling_data["status"] == "completed":
                    return {"text": polling_data["text"]}
                elif polling_data["status"] == "error":
                    raise Exception(f"Transcription failed: {polling_data.get('error', 'Unknown error')}")
                
                await asyncio.sleep(1)

    except Exception as e:
        print(f"Error in transcription: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Cleanup temporary file
        try:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except Exception as e:
            print(f"Error cleaning up temporary files: {str(e)}")
            pass

# Ensure required directories exist
Path("templates").mkdir(exist_ok=True)
Path("static").mkdir(exist_ok=True)