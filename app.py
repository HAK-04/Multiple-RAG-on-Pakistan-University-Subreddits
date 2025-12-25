from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import os
import io
import sys
from contextlib import redirect_stdout

from chroma_db.end_points import RedditEmbeddingsProcessor

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

processor = RedditEmbeddingsProcessor(OPENAI_API_KEY)


@app.route("/")
def home():
    return render_template("front.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.json or {}

    query = data.get("query")
    mode = data.get("mode", "basic")
    print(f"Received query: {query} with mode: {mode}")
    if not query:
        return jsonify({"error": "query is required"}), 400

    # Capture stdout to get print statements
    captured_output = io.StringIO()
    steps = []
    
    try:
        # Redirect stdout to capture print statements
        with redirect_stdout(captured_output):
            if mode == "vanilla":  # BASIC
                steps.append("🔍 Searching for relevant posts...")
                answer = processor.ask_llm_with_context(query)
                steps.append("✓ Found relevant posts")
                steps.append("🤖 Generating answer with LLM...")
                steps.append("✓ Answer generated successfully")
                
            elif mode == "iterative":  # QUERY REWRITE
                print("Starting iterative query rewriting process...")
                steps.append("🔄 Rewriting query for better search results...")
                result = processor.ask_with_query_rewriting(query)
                steps.append("✓ Query rewritten successfully")
                steps.append("🔍 Searching with improved query...")
                steps.append("✓ Found relevant posts")
                steps.append("🤖 Generating enhanced answer...")
                steps.append("✓ Answer generated successfully")
                answer = result["answer"]
                
            elif mode == "reranking":  # RE-RANK
                steps.append("📊 Retrieving initial candidate documents...")
                steps.append("✓ Retrieved candidates")
                steps.append("🔄 Re-ranking documents with cross-encoder...")
                steps.append("✓ Documents re-ranked by relevance")
                steps.append("🤖 Generating answer from top documents...")
                result = processor.ask_with_reranking(query)
                steps.append("✓ Answer generated successfully")
                answer = result["answer"]
                
            elif mode == "agentic":  # TOOL-BASED
                steps.append("🤖 LLM analyzing query to decide which tools to use...")
                steps.append("✓ Tools selected")
                steps.append("🔍 Calling university-specific search tools...")
                result = processor.ask_with_tool_calls(query)
                steps.append("✓ Retrieved information from selected universities")
                steps.append("🤖 Synthesizing comprehensive answer...")
                steps.append("✓ Final answer synthesized")
                answer = result["answer"] if isinstance(result, dict) else result
                
            else:
                return jsonify({"error": "Invalid RAG mode"}), 400

        # Get captured output
        output = captured_output.getvalue()
        
        # Parse important lines from captured output
        for line in output.split('\n'):
            line = line.strip()
            if line and ('✓' in line or '🔍' in line or '🤖' in line or '🔄' in line or '📊' in line or 'Step' in line):
                if line not in steps:
                    steps.append(line)
        print("Captured Steps:", steps)
        return jsonify({
            "answer": answer,
            "steps": steps,
            "raw_output": output  # Optional: for debugging
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "steps": steps + [f"❌ Error: {str(e)}"]
        }), 500


if __name__ == "__main__":
    app.run(debug=True)