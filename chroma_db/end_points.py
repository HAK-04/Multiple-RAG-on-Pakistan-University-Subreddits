import json
import chromadb
from openai import OpenAI
from pathlib import Path
import time
from sentence_transformers import CrossEncoder

class RedditEmbeddingsProcessor:
    def __init__(self, openai_api_key):
        """Initialize the processor with OpenAI client and ChromaDB"""
        self.client = OpenAI(api_key=openai_api_key)
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        # Create or get collection - single collection for posts
        self.collection = self.chroma_client.get_or_create_collection(
            name="reddit_posts",
            metadata={"description": "Reddit posts from GIKI, NUST, and LUMS with comments as metadata"}
        )
        
        # Initialize cross-encoder for re-ranking (lazy loading)
        self.cross_encoder = None
        
    def generate_embedding(self, text):
        """Generate embedding for a given text using OpenAI"""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None
    
    def process_json_file(self, file_path, uni_name):
        """Process a single JSON file and add to ChromaDB"""
        print(f"\n{'='*60}")
        print(f"Processing {uni_name.upper()} data from: {file_path}")
        print(f"{'='*60}")
        
        # Read JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            posts = json.load(f)
        
        print(f"Found {len(posts)} posts to process")
        
        documents = []
        embeddings = []
        metadatas = []
        ids = []
        
        for idx, post in enumerate(posts, 1):
            # Create embedding text from title and body only
            text_to_embed = f"{post.get('title', '')} {post.get('body', '')}"
            
            # Skip if no content
            if not text_to_embed.strip():
                continue
            
            print(f"Processing post {idx}/{len(posts)}: {post['id']}")
            
            # Generate embedding
            embedding = self.generate_embedding(text_to_embed)
            
            if embedding is None:
                print(f"Skipping post {post['id']} due to embedding error")
                continue
            
            # Prepare data
            documents.append(text_to_embed)
            embeddings.append(embedding)
            
            # Store all metadata INCLUDING comments
            metadata = {
                "uni_name": uni_name,
                "post_id": post['id'],
                "subreddit": post.get('subreddit', ''),
                "title": post.get('title', ''),
                "body": post.get('body', ''),
                "upvotes": post.get('upvotes', 0),
                "timestamp": post.get('timestamp', ''),
                "num_comments": post.get('num_comments', 0),
                "url": post.get('url', ''),
                "comments": json.dumps(post.get('comments', []))  # Store comments as JSON string
            }
            metadatas.append(metadata)
            
            # Create unique ID
            ids.append(f"{uni_name}_{post['id']}")
            
            # Rate limiting to avoid OpenAI API limits
            time.sleep(0.1)
        
        # Add to ChromaDB in batch
        if documents:
            print(f"\nAdding {len(documents)} posts to ChromaDB...")
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            print(f"✓ Successfully added {len(documents)} posts from {uni_name.upper()}")
        else:
            print(f"No valid posts to add from {uni_name.upper()}")
    
    def process_all_files(self, file_paths):
        """Process all university JSON files"""
        for uni_name, file_path in file_paths.items():
            if Path(file_path).exists():
                self.process_json_file(file_path, uni_name)
            else:
                print(f"Warning: File not found - {file_path}")
        
        print(f"\n{'='*60}")
        print("Processing Complete!")
        print(f"Total documents in collection: {self.collection.count()}")
        print(f"{'='*60}")
    
    def query_similar_posts(self, query_text, n_results=5, uni_filter=None):
        """Query similar posts from the collection (searches based on title+body embeddings)"""
        query_embedding = self.generate_embedding(query_text)
        
        if query_embedding is None:
            return None
        
        where_filter = {"uni_name": uni_filter} if uni_filter else None
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter
        )
        
        # Parse comments back from JSON string
        if results and results['metadatas']:
            for metadata_list in results['metadatas']:
                for metadata in metadata_list:
                    metadata['comments'] = json.loads(metadata['comments'])
        
        return results
    
    def ask_llm_with_context(self, user_query, n_results=5, uni_filter=None, system_prompt=None):
        """
        Get relevant documents and ask LLM with context
        
        Args:
            user_query: The user's question
            n_results: Number of relevant posts to retrieve
            uni_filter: Filter by university (optional)
            system_prompt: Custom system prompt (optional)
        
        Returns:
            LLM response string
        """
        
        # Default system prompt if none provided
        if system_prompt is None:
            system_prompt = """You are a helpful assistant that answers questions about university life based on Reddit posts and comments from Pakistani universities (GIKI, NUST, and LUMS).

Your task is to:
1. Analyze the provided Reddit posts and comments
2. Give accurate, helpful answers based on the context
3. Mention specific details from posts/comments when relevant
4. If the context doesn't contain enough information, say so
5. Be balanced and present multiple perspectives if they exist in the comments
6. Cite which university the information is from when relevant

Be conversational but informative."""
        
        # Get relevant posts
        print(f"\n🔍 Searching for relevant posts...")
        results = self.query_similar_posts(user_query, n_results=n_results, uni_filter=uni_filter)
        
        if not results or not results['documents'][0]:
            return "Sorry, I couldn't find any relevant posts to answer your question."
        
        # Build context from results
        context_parts = []
        for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0]), 1):
            context_parts.append(f"\n--- POST {i} ---")
            context_parts.append(f"University: {metadata['uni_name'].upper()}")
            context_parts.append(f"Title: {metadata['title']}")
            context_parts.append(f"Body: {metadata['body']}")
            context_parts.append(f"Upvotes: {metadata['upvotes']}")
            
            # Add comments
            if metadata['comments']:
                context_parts.append(f"\nComments ({len(metadata['comments'])}):")
                for j, comment in enumerate(metadata['comments'][:10], 1):  # Limit to first 10 comments
                    context_parts.append(f"{j}. {comment}")
            
            context_parts.append("")  # Empty line between posts
        
        context = "\n".join(context_parts)
        
        print(f"✓ Found {len(results['documents'][0])} relevant posts")
        print(f"\n🤖 Asking LLM...\n")
        
        # Call OpenAI API
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context from Reddit posts:\n\n{context}\n\nUser Question: {user_query}"}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content
            return answer
            
        except Exception as e:
            return f"Error calling LLM: {e}"

    # ============================================================================
    # NEW FUNCTION 1: Query Rewriting with LLM
    # ============================================================================
    
    def ask_with_query_rewriting(self, user_query, n_results=5, uni_filter=None):
        """
        First rewrites the query using LLM to make it more effective for retrieval,
        then performs the search and generates answer.
        
        Args:
            user_query: Original user question
            n_results: Number of relevant posts to retrieve
            uni_filter: Filter by university (optional)
            
        Returns:
            Dict with rewritten_query and final answer
        """
        
        print(f"\n{'='*60}")
        print("🔄 QUERY REWRITING MODE")
        print(f"{'='*60}")
        print(f"Original Query: {user_query}\n")
        
        # Step 1: Rewrite the query using LLM
        rewrite_prompt = """You are a query optimization expert for a Reddit search system containing posts from three Pakistani universities: GIKI, NUST, and LUMS.

Your task is to rewrite user queries to make them more effective for semantic search. The search system uses embeddings to find relevant Reddit posts and comments.

Guidelines for rewriting:
1. Expand abbreviations and add context (e.g., "profs" → "professors teaching quality")
2. Make implicit information explicit (e.g., "is it good?" → "what are student experiences and opinions")
3. Add relevant keywords that would appear in relevant posts
4. If query is about comparison, make that explicit
5. If query mentions a specific university, keep that focus
6. Keep the rewritten query concise but information-rich (2-3 sentences max)

Output ONLY the rewritten query, nothing else."""

        try:
            rewrite_response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": rewrite_prompt},
                    {"role": "user", "content": f"Rewrite this query for better search results: {user_query}"}
                ],
                temperature=0.3,
                max_tokens=150
            )
            
            rewritten_query = rewrite_response.choices[0].message.content.strip()
            print(f"🔄 Rewritten Query: {rewritten_query}\n")
            
        except Exception as e:
            print(f"⚠️ Error rewriting query: {e}")
            print("Falling back to original query\n")
            rewritten_query = user_query
        
        # Step 2: Use rewritten query for retrieval
        print(f"🔍 Searching with rewritten query...")
        results = self.query_similar_posts(rewritten_query, n_results=n_results, uni_filter=uni_filter)
        
        if not results or not results['documents'][0]:
            return {
                "rewritten_query": rewritten_query,
                "answer": "Sorry, I couldn't find any relevant posts to answer your question."
            }
        
        # Build context
        context_parts = []
        for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0]), 1):
            context_parts.append(f"\n--- POST {i} ---")
            context_parts.append(f"University: {metadata['uni_name'].upper()}")
            context_parts.append(f"Title: {metadata['title']}")
            context_parts.append(f"Body: {metadata['body']}")
            context_parts.append(f"Upvotes: {metadata['upvotes']}")
            
            if metadata['comments']:
                context_parts.append(f"\nComments ({len(metadata['comments'])}):")
                for j, comment in enumerate(metadata['comments'][:10], 1):
                    context_parts.append(f"{j}. {comment}")
            
            context_parts.append("")
        
        context = "\n".join(context_parts)
        
        print(f"✓ Found {len(results['documents'][0])} relevant posts")
        print(f"\n🤖 Generating answer...\n")
        
        # Step 3: Generate final answer
        system_prompt = """You are a helpful assistant answering questions about Pakistani universities (GIKI, NUST, LUMS) based on Reddit posts and comments.

Provide accurate, balanced answers that:
- Reference specific details from the context
- Present multiple perspectives if they exist
- Cite which university information is from
- Acknowledge if information is limited

Be conversational and informative."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context:\n{context}\n\nOriginal Question: {user_query}\n\nAnswer the original question based on the context provided."}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content
            
            return {
                "rewritten_query": rewritten_query,
                "answer": answer
            }
            
        except Exception as e:
            return {
                "rewritten_query": rewritten_query,
                "answer": f"Error generating answer: {e}"
            }

    # ============================================================================
    # NEW FUNCTION 2: Tool-based Retrieval with LLM
    # ============================================================================
    
    def ask_with_tool_calls(self, user_query, n_results=5):
        """
        LLM decides which university/universities to search using tool calls.
        Tools directly call ask_llm_with_context for each university.
        
        Args:
            user_query: User's question
            n_results: Number of results per tool call
            
        Returns:
            Final synthesized answer based on tool call results
        """
        
        print(f"\n{'='*60}")
        print("🛠️  TOOL-BASED RETRIEVAL MODE")
        print(f"{'='*60}")
        print(f"Query: {user_query}\n")
        
        # Define tools for the LLM
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_giki",
                    "description": "Search and get answers from GIKI (Ghulam Ishaq Khan Institute) Reddit posts and comments. Use this when the user asks about GIKI specifically or when comparing universities.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The question to answer using GIKI posts"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_nust",
                    "description": "Search and get answers from NUST (National University of Sciences and Technology) Reddit posts and comments. Use this when the user asks about NUST specifically or when comparing universities.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The question to answer using NUST posts"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_lums",
                    "description": "Search and get answers from LUMS (Lahore University of Management Sciences) Reddit posts and comments. Use this when the user asks about LUMS specifically or when comparing universities.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The question to answer using LUMS posts"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
        
        # System prompt for tool selection
        tool_selection_prompt = """You are an intelligent assistant helping users find information about Pakistani universities (GIKI, NUST, LUMS) from Reddit.

You have access to three search tools that retrieve and analyze Reddit posts:
- search_giki: Get information about GIKI
- search_nust: Get information about NUST  
- search_lums: Get information about LUMS

Based on the user's question, decide which tool(s) to call:
- If asking about ONE specific university → call that one tool
- If COMPARING universities → call multiple tools (e.g., both search_nust and search_lums)
- If asking a GENERAL question → call all three tools to provide comprehensive information

Be strategic - only call the tools you actually need."""

        messages = [
            {"role": "system", "content": tool_selection_prompt},
            {"role": "user", "content": user_query}
        ]
        
        try:
            # Step 1: LLM decides which tools to call
            print("🤖 LLM deciding which universities to search...\n")
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.3
            )
            
            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls
            
            if not tool_calls:
                # LLM responded directly without tools
                return response_message.content
            
            print(f"✓ LLM decided to search {len(tool_calls)} universitie(s):")
            for tool_call in tool_calls:
                uni_name = tool_call.function.name.replace('search_', '').upper()
                print(f"  - {uni_name}")
            print()
            
            # Step 2: Execute each tool call using ask_llm_with_context
            messages.append(response_message)
            tool_results = []
            
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                query = function_args['query']
                
                # Extract university name from function name
                uni_name = function_name.replace('search_', '')
                
                print(f"🔍 Searching {uni_name.upper()}...")
                print(f"   Query: {query}\n")
                
                # Call ask_llm_with_context for this university
                answer = self.ask_llm_with_context(
                    user_query=query,
                    n_results=n_results,
                    uni_filter=uni_name,
                    system_prompt=f"""You are analyzing Reddit posts from {uni_name.upper()}.

Provide a focused answer about {uni_name.upper()} based on the provided posts and comments.
Be specific, balanced, and cite details from the posts.
If information is limited, say so clearly."""
                )
                
                # Store result
                tool_results.append({
                    'university': uni_name.upper(),
                    'answer': answer
                })
                
                # Add tool response to conversation
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": answer
                })
                
                print(f"✓ Retrieved answer from {uni_name.upper()}\n")
            
            # Step 3: LLM synthesizes final answer from all tool results
            print("🤖 Synthesizing final answer from all sources...\n")
            
            synthesis_prompt = """Now synthesize a comprehensive answer based on the information gathered from the university searches.

Guidelines:
- If comparing universities, clearly contrast them
- Present information in a balanced, organized way
- Cite which university each piece of information comes from
- Highlight similarities and differences if multiple universities were searched
- Be concise but thorough"""
            
            messages.append({
                "role": "system",
                "content": synthesis_prompt
            })
            
            final_response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.7,
                max_tokens=1500
            )
            
            final_answer = final_response.choices[0].message.content
            
            # Optional: Return structured results
            return {
                'answer': final_answer,
                'individual_results': tool_results,
                'universities_searched': [r['university'] for r in tool_results]
            }
            
        except Exception as e:
            return f"Error in tool-based retrieval: {e}"

    # ============================================================================
    # NEW FUNCTION 3: Two-Stage Retrieval with Cross-Encoder Re-ranking
    # ============================================================================
    
    def _load_cross_encoder(self):
        """Lazy load the cross-encoder model"""
        if self.cross_encoder is None:
            print("📦 Loading cross-encoder model (first time only)...")
            # Using MS MARCO cross-encoder - excellent for semantic relevance
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            print("✓ Cross-encoder loaded!\n")
        return self.cross_encoder
    
    def ask_with_reranking(self, user_query, initial_candidates=20, top_k=5, uni_filter=None):
        """
        Two-stage retrieval: 
        1. Retrieve many candidates using embeddings (fast but less precise)
        2. Re-rank using cross-encoder (slower but more precise)
        
        Args:
            user_query: User's question
            initial_candidates: Number of initial documents to retrieve (default: 20)
            top_k: Number of top documents to keep after re-ranking (default: 5)
            uni_filter: Filter by university (optional)
            
        Returns:
            Final answer based on re-ranked documents
        """
        
        print(f"\n{'='*60}")
        print("🎯 TWO-STAGE RETRIEVAL WITH RE-RANKING")
        print(f"{'='*60}")
        print(f"Query: {user_query}")
        print(f"Stage 1: Retrieving {initial_candidates} candidates")
        print(f"Stage 2: Re-ranking and keeping top {top_k}\n")
        
        # Stage 1: Retrieve many candidates using embedding similarity
        print("📊 Stage 1: Initial retrieval with embeddings...")
        results = self.query_similar_posts(
            user_query, 
            n_results=initial_candidates, 
            uni_filter=uni_filter
        )
        
        if not results or not results['documents'][0]:
            return "Sorry, I couldn't find any relevant posts to answer your question."
        
        print(f"✓ Retrieved {len(results['documents'][0])} candidates\n")
        
        # Stage 2: Re-rank using cross-encoder
        print("🔄 Stage 2: Re-ranking with cross-encoder...")
        
        # Load cross-encoder
        cross_encoder = self._load_cross_encoder()
        
        # Prepare pairs for cross-encoder: (query, document)
        pairs = []
        for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
            # Combine title and body for re-ranking
            doc_text = f"{metadata['title']} {metadata['body']}"
            pairs.append([user_query, doc_text])
        
        # Get relevance scores from cross-encoder
        print("   Computing relevance scores...")
        scores = cross_encoder.predict(pairs)
        
        # Sort by scores (descending)
        scored_results = list(zip(
            scores,
            results['documents'][0],
            results['metadatas'][0]
        ))
        scored_results.sort(key=lambda x: x[0], reverse=True)
        
        # Keep only top_k
        top_results = scored_results[:top_k]
        
        print(f"✓ Re-ranked and selected top {top_k} documents")
        print(f"\n📊 Re-ranking Scores:")
        for i, (score, _, metadata) in enumerate(top_results, 1):
            print(f"   {i}. Score: {score:.4f} | {metadata['uni_name'].upper()} | {metadata['title'][:60]}...")
        
        # Build context from re-ranked results
        context_parts = []
        for i, (score, doc, metadata) in enumerate(top_results, 1):
            context_parts.append(f"\n--- POST {i} (Relevance Score: {score:.4f}) ---")
            context_parts.append(f"University: {metadata['uni_name'].upper()}")
            context_parts.append(f"Title: {metadata['title']}")
            context_parts.append(f"Body: {metadata['body']}")
            context_parts.append(f"Upvotes: {metadata['upvotes']}")
            
            # Add comments
            if metadata['comments']:
                context_parts.append(f"\nComments ({len(metadata['comments'])}):")
                for j, comment in enumerate(metadata['comments'][:8], 1):
                    context_parts.append(f"{j}. {comment}")
            
            context_parts.append("")
        
        context = "\n".join(context_parts)
        
        # Generate answer with LLM
        print(f"\n🤖 Generating answer with top {top_k} re-ranked posts...\n")
        
        system_prompt = """You are a helpful assistant answering questions about Pakistani universities (GIKI, NUST, LUMS) based on Reddit posts and comments.

The posts provided have been carefully re-ranked for relevance using a cross-encoder model, so they should be highly relevant to the question.

Provide accurate, balanced answers that:
- Reference specific details from the context
- Present multiple perspectives if they exist
- Cite which university information is from
- Acknowledge if information is limited

Be conversational and informative."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context (re-ranked for relevance):\n{context}\n\nUser Question: {user_query}"}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content
            
            return {
                "answer": answer,
                "reranking_scores": [
                    {
                        "rank": i,
                        "score": float(score),
                        "university": metadata['uni_name'].upper(),
                        "title": metadata['title']
                    }
                    for i, (score, _, metadata) in enumerate(top_results, 1)
                ]
            }
            
        except Exception as e:
            return {
                "answer": f"Error generating answer: {e}",
                "reranking_scores": []
            }


# # Main execution
# if __name__ == "__main__":
#     # Set your OpenAI API key
#     OPENAI_API_KEY = "sk-svcacct-z_HuXCdWCKM-2PI9YQ76xSkjVVTp8ODdoYa35DMPVOvCAKuHTD1IfZWuiwYBkRFGIQlJT3BlbkFJ9HE9V7RutRYeN2swdUL-FVd9jnEDJQ2k_PUzEyOldAet2j4ZiZMEFLjhRs730wQN2PoA"  # Replace with your actual key
    
#     # Define file paths
#     file_paths = {
#         "giki": "/Users/mhmh/Downloads/BDA part 1/giki_data.json",
#         "nust": "/Users/mhmh/Downloads/BDA part 1/nust_data.json",
#         "lums": "/Users/mhmh/Downloads/BDA part 1/lums_data.json"
#     }
    
#     # Initialize processor
#     processor = RedditEmbeddingsProcessor(OPENAI_API_KEY)
    
#     # Process all files (uncomment to run)
#     # processor.process_all_files(file_paths)
    
#     # Example usage of different methods
#     print("\n" + "="*80)
#     print("DEMONSTRATION OF ALL THREE METHODS")
#     print("="*80)
    
#     # METHOD 3: Tool-based Retrieval
#     print("\n\n" + "="*80)
#     print("METHOD 3: TOOL-BASED RETRIEVAL")
#     print("="*80)
    
#     answer3 = processor.ask_with_tool_calls(
#         user_query="Compare the campus life at NUST and LUMS",
#         n_results=3
#     )
    
#     print("\nAnswer:")
#     print("-" * 80)
#     if isinstance(answer3, dict):
#         print(answer3['answer'])
#     else:
#         print(answer3)
#     print("-" * 80)