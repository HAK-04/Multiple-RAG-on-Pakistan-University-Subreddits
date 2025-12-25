import streamlit as st
# from dotenv import load_dotenv
import os
import io
from contextlib import redirect_stdout

from chroma_db.end_points import RedditEmbeddingsProcessor

# load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Master RAG Systems",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for minimal, modern design
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        padding-top: 2rem;
    }
    
    /* Title styling */
    .main-title {
        font-size: 4rem;
        font-weight: 700;
        color: #000000;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .subtitle {
        font-size: 1.8rem;
        text-align: center;
        background: linear-gradient(90deg, #3b82f6, #0ea5e9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 3rem;
        font-weight: 500;
    }
    
    /* RAG cards styling */
    .rag-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid #f0f0f0;
        transition: transform 0.2s;
        height: 100%;
    }
    
    .rag-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    }
    
    .icon-box {
        width: 50px;
        height: 50px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 28px;
        margin-bottom: 1rem;
    }
    
    /* Team member styling */
    .team-member {
        text-align: center;
        padding: 1rem;
    }
    
    .team-member img {
        width: 120px;
        height: 120px;
        border-radius: 50%;
        object-fit: cover;
        border: 3px solid #000000;
        margin-bottom: 0.5rem;
    }
    
    /* Steps styling */
    .step-item {
        padding: 1rem 1.25rem;
        margin: 0.75rem 0;
        background: #ffffff;
        border-radius: 10px;
        border-left: 4px solid #3b82f6;
        font-size: 1.05rem;
        font-weight: 500;
        color: #1f2937;
        line-height: 1.6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        animation: slideIn 0.3s ease-out;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(-10px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* Answer box styling */
    .answer-box {
        background: #f9fafb;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e5e7eb;
        margin-top: 1.5rem;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processor' not in st.session_state:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        st.error("⚠️ OPENAI_API_KEY not found in environment variables!")
        st.stop()
    st.session_state.processor = RedditEmbeddingsProcessor(OPENAI_API_KEY)

processor = st.session_state.processor

# Header Section
st.markdown('<h1 class="main-title">Master RAG Systems</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Through Interactive Learning</p>', unsafe_allow_html=True)

# RAG Engines Section
st.markdown("### Four RAG Engines to Explore")
st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

rag_engines = [
    {"icon": "⚡", "name": "Vanilla RAG", "color": "#f59e0b", "desc": "Basic retrieval and generation. Perfect for understanding the fundamentals."},
    {"icon": "◎", "name": "Iterative RAG", "color": "#3b82f6", "desc": "Deep dive with multiple retrieval rounds for comprehensive answers."},
    {"icon": "✨", "name": "Re-Ranking RAG", "color": "#a855f7", "desc": "Refines and prioritizes top relevant documents before generating answers."},
    {"icon": "🧠", "name": "Agentic RAG", "color": "#10b981", "desc": "Smart assistant that decides when and how to retrieve information."}
]

for i, engine in enumerate(rag_engines):
    with [col1, col2, col3, col4][i]:
        st.markdown(f"""
            <div class="rag-card">
                <div class="icon-box" style="background: {engine['color']};">
                    {engine['icon']}
                </div>
                <h3 style="margin: 0 0 0.5rem 0; font-size: 1.2rem; color: #111827;">{engine['name']}</h3>
                <p style="margin: 0; color: #6b7280; font-size: 0.9rem; line-height: 1.5;">{engine['desc']}</p>
            </div>
        """, unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# Team Section
st.markdown("### Our Team")
st.markdown("<br>", unsafe_allow_html=True)

team_col1, team_col2, team_col3, team_col4 = st.columns(4)

team_members = [
    {"name": "Muhammad Ahmad", "role": "AI Engineer", "image": "static/images/Muhammad Ahmed.jpg"},
    {"name": "Hammad Ahmad", "role": "Database Engineer", "image": "static/images/Hammad Ahmad.jpeg"},
    {"name": "Husnain Mushtaq", "role": "Frontend Engineer", "image": "static/images/Husnain Mushtaq.PNG"},
    {"name": "Hasham Nadeem", "role": "DevOps and Backend Engineer", "image": "static/images/Hasham Nadeem.jpeg"}
]

for i, member in enumerate(team_members):
    with [team_col1, team_col2, team_col3, team_col4][i]:
        try:
            st.image(member["image"], width=120)
            st.markdown(f"**{member['name']}**")
            st.caption(member["role"])
        except:
            st.markdown(f"**{member['name']}**")
            st.caption(member["role"])

st.markdown("<br><br>", unsafe_allow_html=True)

# Main Query Interface
st.markdown("---")

# RAG Engine Selection
rag_mode = st.selectbox(
    "Select RAG Engine:",
    options=["vanilla", "iterative", "reranking", "agentic"],
    format_func=lambda x: {
        "vanilla": "⚡ Vanilla RAG",
        "iterative": "◎ Iterative RAG",
        "reranking": "✨ Re-Ranking RAG",
        "agentic": "🧠 Agentic RAG"
    }[x],
    index=0
)

# Query Input
query = st.text_area(
    "Enter your question:",
    placeholder="Type your question here...",
    height=120
)

# Initialize session state for results
if 'show_results' not in st.session_state:
    st.session_state.show_results = False
if 'processing_steps' not in st.session_state:
    st.session_state.processing_steps = []
if 'answer' not in st.session_state:
    st.session_state.answer = None
if 'reranked_posts' not in st.session_state:
    st.session_state.reranked_posts = None

# Submit Button
if st.button("🚀 Start", type="primary", use_container_width=True):
    if not query.strip():
        st.warning("⚠️ Please enter a question!")
        st.session_state.show_results = False
    else:
        # Reset and start processing
        st.session_state.show_results = True
        st.session_state.processing_steps = []
        st.session_state.answer = None
        st.session_state.reranked_posts = None
        
        # Show loading spinner
        with st.spinner("Processing your query..."):
            try:
                # Show initial step
                st.session_state.processing_steps.append("🚀 Starting RAG process...")
                
                # Capture stdout
                captured_output = io.StringIO()
            
                with redirect_stdout(captured_output):
                    if rag_mode == "vanilla":
                        st.session_state.processing_steps.append("🔍 Searching for relevant posts...")
                        st.session_state.answer = processor.ask_llm_with_context(query)
                        st.session_state.processing_steps.append("✓ Found relevant posts")
                        st.session_state.processing_steps.append("🤖 Generating answer with LLM...")
                        st.session_state.processing_steps.append("✓ Answer generated successfully")
                        
                    elif rag_mode == "iterative":
                        st.session_state.processing_steps.append("🔄 Rewriting query for better search results...")
                        result = processor.ask_with_query_rewriting(query)
                        st.session_state.processing_steps.append("✓ Query rewritten successfully")
                        st.session_state.processing_steps.append("🔍 Searching with improved query...")
                        st.session_state.processing_steps.append("✓ Found relevant posts")
                        st.session_state.processing_steps.append("🤖 Generating enhanced answer...")
                        st.session_state.processing_steps.append("✓ Answer generated successfully")
                        st.session_state.answer = result["answer"]
                        
                    elif rag_mode == "reranking":
                        st.session_state.processing_steps.append("📊 Retrieving initial candidate documents...")
                        st.session_state.processing_steps.append("✓ Retrieved candidates")
                        st.session_state.processing_steps.append("🔄 Re-ranking documents with LLM...")
                        result = processor.ask_with_reranking(query)
                        st.session_state.processing_steps.append("✓ Documents re-ranked by relevance")
                        st.session_state.processing_steps.append("🤖 Generating answer from top documents...")
                        st.session_state.processing_steps.append("✓ Answer generated successfully")
                        st.session_state.answer = result["answer"]
                        st.session_state.reranked_posts = result.get("reranked_posts", [])
                        
                    elif rag_mode == "agentic":
                        st.session_state.processing_steps.append("🤖 LLM analyzing query to decide which tools to use...")
                        st.session_state.processing_steps.append("✓ Tools selected")
                        st.session_state.processing_steps.append("🔍 Calling university-specific search tools...")
                        result = processor.ask_with_tool_calls(query)
                        st.session_state.processing_steps.append("✓ Retrieved information from selected universities")
                        st.session_state.processing_steps.append("🤖 Synthesizing comprehensive answer...")
                        st.session_state.processing_steps.append("✓ Final answer synthesized")
                        st.session_state.answer = result["answer"] if isinstance(result, dict) else result
                    
                # Get captured output and parse additional steps
                output = captured_output.getvalue()
                for line in output.split('\n'):
                    line = line.strip()
                    if line and ('✓' in line or '🔍' in line or '🤖' in line or '🔄' in line or '📊' in line or 'Step' in line):
                        if line not in st.session_state.processing_steps:
                            st.session_state.processing_steps.append(line)
                
                # Final step
                st.session_state.processing_steps.append("✅ Process completed successfully!")

            except Exception as e:
                import traceback
                error_msg = f"❌ Error: {str(e)}"
                st.session_state.processing_steps.append(error_msg)
                st.error(f"An error occurred: {str(e)}")
                with st.expander("See error details"):
                    st.code(traceback.format_exc())

# Display results if available
if st.session_state.show_results and st.session_state.processing_steps:
    st.markdown("---")
    st.markdown('<h2 style="color: #1f2937; font-size: 1.5rem; font-weight: 600; margin-bottom: 1rem;">Processing Steps:</h2>', unsafe_allow_html=True)
    steps_html = ""
    for step in st.session_state.processing_steps:
        steps_html += f'<div class="step-item">{step}</div>'
    st.markdown(steps_html, unsafe_allow_html=True)
    
    # Display re-ranked posts if available (for reranking mode)
    if st.session_state.reranked_posts and len(st.session_state.reranked_posts) > 0:
        st.markdown("---")
        st.markdown('<h2 style="color: #1f2937; font-size: 1.5rem; font-weight: 600; margin-bottom: 1rem;">Re-ranked Posts:</h2>', unsafe_allow_html=True)
        for post in st.session_state.reranked_posts:
            with st.expander(f"Post {post['rank']} - {post['title'][:60]}... (Score: {post['score']:.4f})"):
                st.markdown(f"**University:** {post['university']}")
                st.markdown(f"**Title:** {post['title']}")
                st.markdown(f"**Body:** {post['body']}")
                st.markdown(f"**Upvotes:** {post['upvotes']}")
                st.markdown(f"**Relevance Score:** {post['score']:.4f}")
    
    if st.session_state.answer:
        st.markdown("---")
        st.markdown('<h2 style="color: #1f2937; font-size: 1.5rem; font-weight: 600; margin-bottom: 1rem;">Answer:</h2>', unsafe_allow_html=True)
        st.markdown(st.session_state.answer)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #6b7280; padding: 1rem;'>Built with ❤️ using Streamlit</div>",
    unsafe_allow_html=True
)
