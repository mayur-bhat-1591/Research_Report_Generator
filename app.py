#!/usr/bin/env python
# =============================================================================
# app.py
#
# Enhanced Business Pitch Presentation Generator with Standard LaTeX Output,
# Improved UI with Fun Animations, and PDF Generation Capabilities
#
# Production-grade implementation with robust state management and automatic file saving
#
# =============================================================================

import os
import re
import time
import json
import random
import streamlit as st
import tempfile
import subprocess
import base64
import datetime
from pathlib import Path
import concurrent.futures
import threading
import traceback
import uuid
import sys

# Try to load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("[DEBUG] Loaded environment variables from .env file")
except ImportError:
    print("[DEBUG] python-dotenv not installed, skipping .env file loading")

# Try to import the pdflatex library
try:
    import pdflatex
    PDFLATEX_AVAILABLE = True
    print("[DEBUG] PDFLaTeX library is available")
except ImportError:
    PDFLATEX_AVAILABLE = False
    print("[DEBUG] PDFLaTeX library not available, using subprocess fallback")

# ------------------------------------------------------------------------------
# Initialize session state for persistent storage
# ------------------------------------------------------------------------------
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.generated_documents = {}
    st.session_state.current_session_id = None
    st.session_state.celebrated = False
    st.session_state.api_keys_configured = False

# Create output directory if it doesn't exist
OUTPUT_DIR = Path("generated_documents")
OUTPUT_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------------------------
# Helper function to extract JSON from text (removes markdown code fences)
# ------------------------------------------------------------------------------
def extract_json(text: str) -> str:
    match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    return text.strip()

# ------------------------------------------------------------------------------
# Import Agno Agent framework modules and models
# ------------------------------------------------------------------------------
try:
    from agno.agent import Agent
    from agno.models.openai import OpenAIChat
    AGNO_AVAILABLE = True
except ImportError:
    st.error("Failed to import Agno Agent framework. Please install it with: pip install agno")
    Agent = None
    OpenAIChat = None
    AGNO_AVAILABLE = False

# ------------------------------------------------------------------------------
# Import the Anthropic Claude model.
# ------------------------------------------------------------------------------
try:
    from agno.models.anthropic import Claude
    debug_msg = "Using official Anthropic Claude implementation."
    CLAUDE_AVAILABLE = True
except ImportError:
    try:
        class Claude(OpenAIChat):
            def __init__(self, id):
                super().__init__("gpt-4")
                self.original_id = id
                self.show_tool_calls = None
                self.tool_choice = None
        debug_msg = "Anthropic Claude not available; using fallback dummy Claude (GPT-4)."
        CLAUDE_AVAILABLE = False
    except:
        Claude = None
        debug_msg = "Failed to create Claude fallback. API functionality will be limited."
        CLAUDE_AVAILABLE = False
print(f"[DEBUG] {debug_msg}")

# ------------------------------------------------------------------------------
# Import internet access tools for research and data gathering
# ------------------------------------------------------------------------------
try:
    from agno.tools.duckduckgo import DuckDuckGoTools
    from agno.tools.newspaper4k import Newspaper4kTools
    RESEARCH_TOOLS_AVAILABLE = True
except ImportError:
    DuckDuckGoTools = None
    Newspaper4kTools = None
    RESEARCH_TOOLS_AVAILABLE = False
    print("[DEBUG] Research tools not available.")

# =============================================================================
# API Key Management
# =============================================================================
def get_api_keys():
    """Get API keys from environment or session state"""
    openai_key = os.environ.get("OPENAI_API_KEY", "") or st.session_state.get("openai_api_key", "")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "") or st.session_state.get("anthropic_api_key", "")
    claude_model = os.environ.get("CLAUDE_MODEL_ID", "claude-sonnet-4-20250514")

    return openai_key, anthropic_key, claude_model

def configure_api_keys():
    """Display API key configuration UI"""
    st.sidebar.header("🔑 API Configuration")

    openai_key, anthropic_key, claude_model = get_api_keys()

    # OpenAI API Key
    openai_input = st.sidebar.text_input(
        "OpenAI API Key",
        value=openai_key[:8] + "..." if openai_key else "",
        type="password",
        help="Enter your OpenAI API key"
    )

    # Anthropic API Key
    anthropic_input = st.sidebar.text_input(
        "Anthropic API Key",
        value=anthropic_key[:8] + "..." if anthropic_key else "",
        type="password",
        help="Enter your Anthropic API key"
    )

    # Save keys to session state if provided
    if openai_input and not openai_input.endswith("..."):
        st.session_state.openai_api_key = openai_input
        os.environ["OPENAI_API_KEY"] = openai_input

    if anthropic_input and not anthropic_input.endswith("..."):
        st.session_state.anthropic_api_key = anthropic_input
        os.environ["ANTHROPIC_API_KEY"] = anthropic_input

    # Check if keys are configured
    final_openai, final_anthropic, _ = get_api_keys()

    if final_openai and final_anthropic:
        st.sidebar.success("✅ API keys configured")
        st.session_state.api_keys_configured = True
        return True
    else:
        st.sidebar.warning("⚠️ Please configure both API keys")
        st.session_state.api_keys_configured = False
        return False

# =============================================================================
# Global Configuration and Environment Variables
# =============================================================================
OPENAI_API_KEY, ANTHROPIC_API_KEY, CLAUDE_MODEL_ID = get_api_keys()

# Constants for API timeouts and retries
API_TIMEOUT = 120  # seconds
MAX_RETRIES = 3    # maximum number of retry attempts
RETRY_DELAY = 5    # seconds between retries

# Default LaTeX template (standard article class)
DEFAULT_LATEX_TEMPLATE = r"""
\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{microtype}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{tikz}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{enumitem}
\usepackage{geometry}
\usepackage{fancyhdr}
\usepackage{listings}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{float}

% Page setup
\geometry{a4paper,margin=1in}
\pagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0.4pt}
\fancyhead[L]{\slshape DOCUMENT_TITLE}
\fancyhead[R]{\slshape \thepage}
\fancyfoot[C]{\slshape DOCUMENT_FOOTER}

% Color definitions
\definecolor{primarycolor}{RGB}{0,128,128}
\definecolor{secondarycolor}{RGB}{70,130,180}
\definecolor{accentcolor}{RGB}{46,139,87}
\definecolor{codebg}{RGB}{240,240,240}
\definecolor{codeframe}{RGB}{200,200,200}

% Code listing setup
\lstset{
    basicstyle=\small\ttfamily,
    backgroundcolor=\color{codebg},
    frame=single,
    rulecolor=\color{codeframe},
    breaklines=true,
    breakatwhitespace=true,
}

\hypersetup{
  colorlinks=true,
  linkcolor=primarycolor,
  filecolor=primarycolor,
  urlcolor=primarycolor,
  citecolor=primarycolor
}

\title{DOCUMENT_TITLE}
\author{DOCUMENT_AUTHOR}
\date{\today}

\begin{document}

\maketitle
\tableofcontents
\newpage

% CONTENT_PLACEHOLDER

\end{document}
"""

# Fun facts to display during processing
LATEX_FUN_FACTS = [
    "LaTeX was created by Leslie Lamport in the 1980s as an extension of TeX, which was created by Donald Knuth.",
    "The 'La' in LaTeX comes from Lamport, and the 'TeX' comes from the Greek word 'τέχνη' (techne), meaning 'art' and 'craft'.",
    "LaTeX is pronounced 'Lah-tech' or 'Lay-tech', not 'Lay-teks'.",
    "The official LaTeX logo includes specific capitalization: 'LaTeX'.",
    "LaTeX is widely used in academia, especially for mathematical and scientific documents.",
    "NASA, IEEE, and many academic journals require submissions in LaTeX format.",
    "LaTeX's first release was in 1984, making it older than many popular software tools!",
    "LaTeX automatically handles numbering, cross-references, citations, and bibliographies.",
    "LaTeX uses a declarative approach - you describe what you want, not how to achieve it.",
    "LaTeX's typesetting algorithms for mathematics are considered superior to most word processors.",
    "The TeX family of software is famously stable - files from the 1980s still compile today.",
    "For each bug found in TeX, Donald Knuth would double his reward. The current amount is $327.68.",
    "LaTeX's equation system inspired many online mathematics editors like MathJax.",
    "LaTeX can create presentation slides, posters, books, articles, letters, and even business cards.",
    "The LaTeX community maintains thousands of packages for specialized needs.",
    "The AMS (American Mathematical Society) has its own LaTeX extensions called AMS-LaTeX.",
    "LaTeX stores files in plain text, making them ideal for version control systems like Git.",
    "The 'rubber duck debugging' technique works well with LaTeX - explain your code to a rubber duck!",
    "LaTeX can generate beautiful diagrams using TikZ and PGF packages.",
    "The 'microtype' package in LaTeX improves typography by adjusting letter spacing and margins.",
    "LaTeX automatically hyphenates words across lines using sophisticated language-specific algorithms.",
    "LaTeX uses ligatures to improve typography, converting character pairs like 'fi' into single glyphs.",
    "LaTeX can generate accessible PDF documents with proper structure for screen readers.",
    "The BibTeX system for bibliography management was created in 1985 and is still widely used.",
    "LaTeX's successor, LaTeX3, has been in development since the early 1990s.",
    "LaTeX uses Knuth's revolutionary Computer Modern font family by default.",
    "Overleaf, an online LaTeX editor, has over 10 million users worldwide.",
    "LaTeX's error messages are famously cryptic but very precise once you learn to read them!",
    "LaTeX can create stunning mathematical diagrams using packages like commutative-diagrams.",
    "The largest known LaTeX document might be Edward Kmett's thesis, with over 10,000 pages."
]

# =============================================================================
# Utility Functions and Debug Logging
# =============================================================================
def debug_log(message: str):
    """Log debug messages to console with timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[DEBUG {timestamp}] {message}")

# =============================================================================
# File management functions
# =============================================================================
def save_file_to_disk(content, filename, extension):
    """
    Save content to a file in the output directory with timestamp.

    Args:
        content (str): Content to save
        filename (str): Base filename
        extension (str): File extension

    Returns:
        Path: Path to the saved file
    """
    # Remove special characters from filename
    safe_filename = re.sub(r'[^\w\-_\. ]', '_', filename)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create filename with timestamp
    full_filename = f"{safe_filename}_{timestamp}.{extension}"
    file_path = OUTPUT_DIR / full_filename

    # Save file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    debug_log(f"Saved file to {file_path}")
    return file_path

def generate_file_link(file_path, link_text):
    """Generate HTML link to local file"""
    return f'<a href="file://{file_path.absolute()}" target="_blank">{link_text}</a>'

# =============================================================================
# Timeout handler
# =============================================================================
class TimeoutError(Exception):
    """Exception raised when a function call times out"""
    pass

def timeout_handler(seconds):
    """Decorator to handle timeouts for function calls"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = [None]
            error = [None]

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    error[0] = e

            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(seconds)

            if thread.is_alive():
                debug_log(f"Function {func.__name__} timed out after {seconds} seconds")
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")

            if error[0]:
                raise error[0]

            return result[0]

        return wrapper
    return decorator

# =============================================================================
# Token counting and chunking utilities
# =============================================================================
def estimate_tokens(text: str) -> int:
    """Rough estimation of tokens in text (1 token ≈ 4 characters)"""
    return len(text) // 4

def chunk_content_for_latex(content: str, max_tokens: int = 8000) -> list:
    """
    Split content into chunks that fit within token limits for LaTeX conversion.
    """
    if estimate_tokens(content) <= max_tokens:
        return [content]

    # Split by sections first
    sections = content.split("--- SECTION")
    chunks = []
    current_chunk = ""

    for i, section in enumerate(sections):
        if i == 0:  # First part before any section
            current_chunk = section
            continue

        section_content = "--- SECTION" + section
        section_tokens = estimate_tokens(section_content)
        current_tokens = estimate_tokens(current_chunk)

        if current_tokens + section_tokens > max_tokens and current_chunk:
            # Save current chunk and start new one
            chunks.append(current_chunk.strip())
            current_chunk = section_content
        else:
            current_chunk += section_content

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

# =============================================================================
# IMPROVED: Function to compile LaTeX to PDF with better error handling
# =============================================================================
def compile_latex_to_pdf(latex_code: str, base_filename: str = None) -> tuple:
    """
    Compile LaTeX code to PDF using multiple methods with enhanced error handling.

    Args:
        latex_code (str): LaTeX source code
        base_filename (str, optional): Base filename for the PDF. Defaults to None.

    Returns:
        tuple: (success: bool, pdf_base64: str, log: str, pdf_path: Path or None)
    """
    debug_log("Starting PDF compilation...")

    # Create safe filename
    safe_filename = "document"
    if base_filename:
        safe_filename = re.sub(r'[^\w\-_]', '_', base_filename)

    # Method 1: Try using the pdflatex Python library (preferred)
    if PDFLATEX_AVAILABLE:
        try:
            debug_log("Attempting PDF compilation using pdflatex library...")

            # Create PDFLaTeX instance from binary string
            pdfl = pdflatex.PDFLaTeX.from_binarystring(
                latex_code.encode('utf-8'),
                safe_filename
            )

            # Compile to PDF
            pdf_data, log_data, completed_process = pdfl.create_pdf(
                keep_pdf_file=False,
                keep_log_file=False
            )

            # Check if compilation was successful
            if completed_process.returncode == 0 and pdf_data:
                debug_log("PDF compilation successful using pdflatex library!")

                # Convert PDF to base64
                pdf_base64 = base64.b64encode(pdf_data).decode('utf-8')

                # Save PDF to output directory
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"{safe_filename}_{timestamp}.pdf"
                pdf_path = OUTPUT_DIR / output_filename
                pdf_path.write_bytes(pdf_data)
                debug_log(f"Saved PDF to {pdf_path}")

                return True, pdf_base64, log_data, pdf_path
            else:
                error_msg = f"PDFLaTeX library compilation failed with return code {completed_process.returncode}"
                if log_data:
                    error_msg += f"\n\nLog:\n{log_data}"
                debug_log(f"PDFLaTeX library failed: {error_msg}")
                # Continue to fallback method

        except Exception as library_error:
            debug_log(f"PDFLaTeX library failed with exception: {library_error}")
            # Continue to fallback method

    # Method 2: Fallback to subprocess method with improved handling
    debug_log("Attempting PDF compilation using subprocess...")

    try:
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            debug_log(f"Using temporary directory: {temp_path}")

            # Write LaTeX code to a temporary file
            tex_file = temp_path / "document.tex"
            tex_file.write_text(latex_code, encoding='utf-8')
            debug_log(f"Wrote LaTeX file: {tex_file}")

            # Run pdflatex with better error handling
            log_output = ""
            success = False

            # First compilation
            try:
                debug_log("Running first pdflatex compilation...")
                process = subprocess.run(
                    [
                        "pdflatex",
                        "-interaction=nonstopmode",
                        "-halt-on-error",
                        "-output-directory", str(temp_path),
                        "document.tex"
                    ],
                    cwd=str(temp_path),
                    capture_output=True,
                    text=True,
                    timeout=90,
                    env=dict(os.environ, TEXMFCACHE=str(temp_path))
                )

                log_output += f"=== First Compilation ===\n"
                log_output += f"Return code: {process.returncode}\n"
                log_output += f"STDOUT:\n{process.stdout}\n"
                log_output += f"STDERR:\n{process.stderr}\n\n"

                debug_log(f"First compilation return code: {process.returncode}")

                if process.returncode == 0:
                    # Second compilation for references
                    debug_log("Running second pdflatex compilation...")
                    process2 = subprocess.run(
                        [
                            "pdflatex",
                            "-interaction=nonstopmode",
                            "-halt-on-error",
                            "-output-directory", str(temp_path),
                            "document.tex"
                        ],
                        cwd=str(temp_path),
                        capture_output=True,
                        text=True,
                        timeout=90,
                        env=dict(os.environ, TEXMFCACHE=str(temp_path))
                    )

                    log_output += f"=== Second Compilation ===\n"
                    log_output += f"Return code: {process2.returncode}\n"
                    log_output += f"STDOUT:\n{process2.stdout}\n"
                    log_output += f"STDERR:\n{process2.stderr}\n\n"

                    debug_log(f"Second compilation return code: {process2.returncode}")

                    if process2.returncode == 0:
                        success = True

            except subprocess.TimeoutExpired:
                log_output += "=== Compilation Timeout ===\nPDFLaTeX compilation timed out after 90 seconds.\n\n"
                debug_log("PDFLaTeX compilation timed out")
            except Exception as e:
                log_output += f"=== Compilation Error ===\nError running pdflatex: {str(e)}\n\n"
                debug_log(f"Error running pdflatex: {e}")

            # Check if PDF was created
            pdf_file = temp_path / "document.pdf"
            debug_log(f"Checking for PDF file: {pdf_file}")
            debug_log(f"PDF file exists: {pdf_file.exists()}")

            if success and pdf_file.exists():
                debug_log("PDF compilation successful using subprocess!")

                # Convert PDF to base64
                pdf_data = pdf_file.read_bytes()
                pdf_base64 = base64.b64encode(pdf_data).decode('utf-8')

                # Save PDF to output directory
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"{safe_filename}_{timestamp}.pdf"
                pdf_path = OUTPUT_DIR / output_filename
                pdf_path.write_bytes(pdf_data)
                debug_log(f"Saved PDF to {pdf_path}")

                return True, pdf_base64, log_output, pdf_path
            else:
                debug_log("PDF compilation failed - no PDF file generated")
                return False, "", log_output, None

    except subprocess.TimeoutExpired:
        error_message = "Error: LaTeX compilation timed out after 90 seconds."
        debug_log(error_message)
        return False, "", error_message, None
    except FileNotFoundError:
        error_message = (
            "Error: pdflatex command not found. Please install a LaTeX distribution:\n"
            "- TeX Live (Linux/Windows): https://www.tug.org/texlive/\n"
            "- MacTeX (macOS): https://www.tug.org/mactex/\n"
            "- MiKTeX (Windows): https://miktex.org/\n"
            "Or install the pdflatex Python library: pip install pdflatex"
        )
        debug_log(error_message)
        return False, "", error_message, None
    except Exception as e:
        error_message = f"Error compiling LaTeX: {str(e)}\n{traceback.format_exc()}"
        debug_log(error_message)
        return False, "", error_message, None

# =============================================================================
# Function to create a download link for the PDF in Streamlit
# =============================================================================
def get_pdf_download_link(pdf_base64: str, file_name: str = "document.pdf") -> str:
    """Create a download link for the PDF."""
    href = f'<a href="data:application/pdf;base64,{pdf_base64}" download="{file_name}">Download {file_name}</a>'
    return href

# =============================================================================
# Fun animation and status updates
# =============================================================================
def show_fun_fact():
    """Display a random LaTeX fun fact"""
    fact = random.choice(LATEX_FUN_FACTS)
    return fact

def create_loading_animation():
    """Create a simple loading animation placeholder that will be updated"""
    # Create a container for the animation
    container = st.empty()
    return container

def update_loading_animation(container, step=0, total_steps=10, fun_fact=None):
    """Update the loading animation with progress information and a fun fact"""
    emojis = ["🧠", "⚙️", "🔍", "📝", "✨", "🚀", "🎯", "🔮", "💡", "🎨"]
    progress = min(step / total_steps, 1.0)
    emoji = emojis[step % len(emojis)]

    # Calculate progress bar
    bar_length = 20
    filled_length = int(bar_length * progress)
    bar = "█" * filled_length + "░" * (bar_length - filled_length)

    # Create progress message
    progress_pct = int(progress * 100)
    message = f"{emoji} Processing: [{bar}] {progress_pct}%"

    # Add fun fact if provided
    if fun_fact:
        message += f"\n\n**Did you know?** {fun_fact}"

    # Update the container
    container.markdown(message)

    return container

def typewriter_text(text, placeholder=None, delay=0.01):
    """Create a typewriter effect for text"""
    if placeholder is None:
        placeholder = st.empty()

    for i in range(len(text) + 1):
        placeholder.markdown(f"**{text[:i]}**")
        time.sleep(delay)

    return placeholder

# =============================================================================
# Function to create combined markdown document
# =============================================================================
def create_combined_markdown(designed_sections, outline_data, topic):
    """
    Combines all designed sections into a single Markdown document with proper formatting.

    Args:
        designed_sections (list): List of dictionaries containing designed content for each section
        outline_data (dict): Dictionary containing the document outline
        topic (str): The document topic

    Returns:
        str: Combined markdown document
    """
    sections = sorted(designed_sections, key=lambda x: x['section_number'])
    sections_dict = {s['section_number']: s for s in sections}

    # Get section titles from outline
    section_titles = {}
    for section in outline_data.get('sections', []):
        section_titles[section.get('section_number')] = section.get('title', f"Section {section.get('section_number')}")

    # Create markdown document
    md_content = f"# {topic}\n\n"
    md_content += "## Table of Contents\n\n"

    # Generate TOC
    for section_number in sorted(section_titles.keys()):
        md_content += f"{section_number}. [{section_titles.get(section_number)}](#section-{section_number})\n"

    md_content += "\n---\n\n"

    # Add each section
    for section_number in sorted(section_titles.keys()):
        title = section_titles.get(section_number)
        md_content += f"## {section_number}. {title} <a id='section-{section_number}'></a>\n\n"

        if section_number in sections_dict:
            section_content = sections_dict[section_number].get('designed_content', '')
            md_content += f"{section_content}\n\n"
        else:
            md_content += "*Content for this section is not available.*\n\n"

        md_content += "---\n\n"

    # Add footer
    md_content += f"\n\n*Generated by AI Document Assistant*\n"

    return md_content

# =============================================================================
# Generate default sections if API calls fail
# =============================================================================
def generate_default_sections(num_sections, topic):
    """Generate default sections if API calls fail"""
    sections = []
    for i in range(1, num_sections + 1):
        title = f"Section {i}"
        if i == 1:
            title = "Introduction"
        elif i == num_sections:
            title = "Conclusion"

        sections.append({
            "section_number": i,
            "title": title,
            "description": f"This is a placeholder for {title}."
        })
    return {"sections": sections}

def generate_default_prompts(outline_data):
    """Generate default prompts if API calls fail"""
    prompts = []
    for section in outline_data.get("sections", []):
        section_number = section.get("section_number")
        title = section.get("title")
        description = section.get("description")
        prompt = f"Generate content for the section titled '{title}'. {description} Include relevant information, examples, and analysis."
        prompts.append({"section_number": section_number, "prompt": prompt})
    return {"section_prompts": prompts}

# =============================================================================
# FIXED: AgentManager Class
# =============================================================================
class AgentManager:
    def __init__(self, topic=None, user_input=None, detail_level=None):
        self.agents = {}
        self.topic = topic
        self.user_input = user_input
        self.detail_level = detail_level
        debug_log(f"AgentManager initialized with topic: {self.topic}")

    def create_planning_agents(self):
        try:
            openai_key, anthropic_key, claude_model = get_api_keys()

            if not openai_key:
                raise Exception("OpenAI API key not configured")
            if not anthropic_key:
                raise Exception("Anthropic API key not configured")

            self.agents['outline_generator'] = Agent(
                name="Outline Generator",
                model=OpenAIChat(id="gpt-4o-mini"),
                instructions="",
                markdown=False,
                show_tool_calls=True,
            )
            debug_log("Created Outline Generator agent using OpenAIChat.")

            self.agents['prompt_generator'] = Agent(
                name="Prompt Generator",
                model=Claude(id=claude_model),
                instructions="",
                markdown=False,
                show_tool_calls=True,
            )
            debug_log("Created Prompt Generator agent using Anthropic Claude with model id: " + claude_model)
            return True
        except Exception as e:
            debug_log(f"Error creating planning agents: {str(e)}")
            st.error(f"Failed to initialize AI agents: {str(e)}")
            return False

    def create_validation_agent(self):
        try:
            openai_key, anthropic_key, claude_model = get_api_keys()

            if not anthropic_key:
                raise Exception("Anthropic API key not configured")

            self.agents['validator'] = Agent(
                name="Story Validator",
                model=Claude(id=claude_model),
                instructions="",
                markdown=False,
                show_tool_calls=True,
            )
            debug_log("Created Story Validator agent using Anthropic Claude with model id: " + claude_model)
            return True
        except Exception as e:
            debug_log(f"Error creating validation agent: {str(e)}")
            return False

    def create_data_gatherer_agent(self, section_number: int):
        try:
            openai_key, _, _ = get_api_keys()

            if not openai_key:
                raise Exception("OpenAI API key not configured")

            agent_name = f"Section {section_number} Content Generator"
            tools = []
            if RESEARCH_TOOLS_AVAILABLE:
                if DuckDuckGoTools:
                    tools.append(DuckDuckGoTools())
                if Newspaper4kTools:
                    tools.append(Newspaper4kTools())

            agent = Agent(
                name=agent_name,
                model=OpenAIChat(id="gpt-4o-mini"),
                tools=tools,
                instructions="",
                markdown=True,
                show_tool_calls=True,
            )
            debug_log(f"Created Data Gatherer agent for {agent_name}.")
            return agent
        except Exception as e:
            debug_log(f"Error creating data gatherer agent: {str(e)}")
            return None

    def create_design_agent(self, section_number: int):
        try:
            openai_key, _, _ = get_api_keys()

            if not openai_key:
                raise Exception("OpenAI API key not configured")

            agent_name = f"Section {section_number} Design Agent"
            # FIXED: Define tools properly
            tools = []
            if RESEARCH_TOOLS_AVAILABLE:
                if DuckDuckGoTools:
                    tools.append(DuckDuckGoTools())
                if Newspaper4kTools:
                    tools.append(Newspaper4kTools())

            agent = Agent(
                name=agent_name,
                model=OpenAIChat(id="gpt-4o-mini"),
                tools=tools,  # FIXED: Now properly defined
                instructions="",
                markdown=True,
                show_tool_calls=True,
            )
            debug_log(f"Created Design Agent for {agent_name}.")
            return agent
        except Exception as e:
            debug_log(f"Error creating design agent: {str(e)}")
            return None

    def create_latex_formatter_agent(self):
        try:
            openai_key, _, _ = get_api_keys()

            if not openai_key:
                raise Exception("OpenAI API key not configured")

            # Use GPT-4o-mini for LaTeX formatting to avoid token limits
            self.agents['latex_formatter'] = Agent(
                name="LaTeX Formatter",
                model=OpenAIChat(id="gpt-4o-mini"),
                instructions="",
                markdown=False,
                show_tool_calls=True,
            )
            debug_log("Created LaTeX Formatter agent using GPT-4o-mini.")
            return self.agents['latex_formatter']
        except Exception as e:
            debug_log(f"Error creating LaTeX formatter agent: {str(e)}")
            return None

# =============================================================================
# Default Run Prompt
# =============================================================================
DEFAULT_RUN_PROMPT = "Please generate your response."

# =============================================================================
# Planning Team Functions
# =============================================================================
def planning_team(user_input: str, detail_level: str, topic: str, animation_container) -> dict:
    """
    The Planning Team generates a detailed outline and corresponding section prompts.
    """
    # Determine number of sections based on detail level
    if detail_level == "TL;DR":
        num_sections = 5
    elif detail_level == "Executive Summary":
        num_sections = 10
    else:  # Detailed Summary
        num_sections = 15

    # Start animation
    update_loading_animation(animation_container, 0, 4, show_fun_fact())

    # Initialize manager with topic and user_input to ensure they're available
    manager = AgentManager(topic=topic, user_input=user_input, detail_level=detail_level)
    if not manager.create_planning_agents():
        st.error("Failed to create planning agents. Using fallback outline.")
        # Generate fallback outline
        outline_data = generate_default_sections(num_sections, topic)
        prompt_data = generate_default_prompts(outline_data)
        return {
            "outline": outline_data,
            "section_prompts": prompt_data,
            "raw_outline": json.dumps(outline_data),
            "raw_prompts": json.dumps(prompt_data)
        }

    # Outline Generator Instructions
    outline_instructions = f"""
You are an expert document designer specializing in structured content organization. Your task is to analyze the provided text and generate a detailed, coherent outline for a LaTeX document.

User Input:
```
{user_input}
```

Topic: "{topic}"
Detail Level: {detail_level}

## ANALYSIS APPROACH:
1. First, identify the main themes and key concepts in the user's text
2. Organize these concepts into a logical progression that tells a complete story
3. Ensure appropriate coverage of introductory material, core content, and concluding thoughts
4. Balance theoretical/conceptual sections with practical/application sections

## OUTPUT REQUIREMENTS:
1. Produce exactly {num_sections} sections based on the detail level
2. For each section, output a JSON object with the following keys:
   - section_number (integer): Sequential numbering starting from 1
   - title (string): Clear, concise section title (5-8 words maximum)
   - description (string): 1-2 sentences explaining the section's focus and importance
3. First section should introduce the topic; final section should provide conclusion/next steps
4. Middle sections should follow a logical progression of ideas
5. Use descriptive, specific titles rather than generic ones
6. Do not include any explanatory text outside the JSON structure
7. Output only valid JSON following this schema:
    {{{{ "sections": [{{"section_number": 1, "title": "Section Title", "description": "Brief description."}}, ...] }}}}

## IMPORTANT:
- Ensure JSON is properly formatted with no syntax errors
- Section numbers must be sequential integers
- Each section title should be unique and descriptive
- Descriptions should clearly communicate the section's purpose
- The outline should comprehensively cover the topic
"""
    manager.agents['outline_generator'].instructions = outline_instructions
    debug_log("Set instructions for Outline Generator.")

    # Show a typewriter effect for the status
    status = st.empty()
    typewriter_text("Analyzing your text and generating document outline...", status)

    # Try to get outline with timeout and retry logic
    outline_data = {"sections": []}
    for attempt in range(MAX_RETRIES):
        try:
            # Start a separate thread for the API call with timeout
            with st.spinner(f"Generating outline (attempt {attempt+1}/{MAX_RETRIES})..."):
                outline_future = concurrent.futures.ThreadPoolExecutor().submit(
                    manager.agents['outline_generator'].run, DEFAULT_RUN_PROMPT
                )
                outline_response = outline_future.result(timeout=API_TIMEOUT)

                raw_outline = extract_json(outline_response.content)
                try:
                    outline_data = json.loads(raw_outline)
                    debug_log("Outline generated successfully.")
                    break  # Success, break out of retry loop
                except Exception as e:
                    debug_log(f"Error parsing outline JSON: {e}")
                    # If last attempt, use fallback
                    if attempt == MAX_RETRIES - 1:
                        outline_data = generate_default_sections(num_sections, topic)
                        raw_outline = json.dumps(outline_data)
                    else:
                        time.sleep(RETRY_DELAY)  # Wait before retrying
        except (concurrent.futures.TimeoutError, TimeoutError):
            debug_log(f"Outline generation timed out on attempt {attempt+1}")
            if attempt == MAX_RETRIES - 1:
                st.warning("Outline generation timed out. Using fallback outline.")
                outline_data = generate_default_sections(num_sections, topic)
                raw_outline = json.dumps(outline_data)
            else:
                time.sleep(RETRY_DELAY)  # Wait before retrying
        except Exception as e:
            debug_log(f"Error during outline generation: {e}")
            if attempt == MAX_RETRIES - 1:
                st.warning(f"Error generating outline: {str(e)}. Using fallback outline.")
                outline_data = generate_default_sections(num_sections, topic)
                raw_outline = json.dumps(outline_data)
            else:
                time.sleep(RETRY_DELAY)  # Wait before retrying

    # Update animation
    update_loading_animation(animation_container, 1, 4, show_fun_fact())

    # Update status and show a new fun fact
    status.empty()
    typewriter_text("Outline complete! Now creating detailed content prompts...", status)

    # Prompt Generator Instructions
    prompt_instructions = f"""
You are an expert prompt designer. Based on the following outline and user input, generate a detailed prompt for each section that instructs a content generator to produce comprehensive LaTeX content.
Requirements:
1. For each section in the outline, output a JSON object with the following keys:
   - section_number (integer)
   - prompt (string instructing to produce detailed LaTeX content for the document section)
2. Do not include any extra text.
3. Output only valid JSON following this schema:
    {{{{ "section_prompts": [{{"section_number": 1, "prompt": "Detailed prompt for section 1."}}, ...] }}}}
User Input:
```
{user_input}
```
Topic: "{topic}"
Detail Level: {detail_level}

Outline:
{json.dumps(outline_data, indent=2)}
"""
    manager.agents['prompt_generator'].instructions = prompt_instructions
    debug_log("Set instructions for Prompt Generator.")

    # Try to get prompts with timeout and retry logic
    prompt_data = {"section_prompts": []}
    for attempt in range(MAX_RETRIES):
        try:
            # Start a separate thread for the API call with timeout
            with st.spinner(f"Generating prompts (attempt {attempt+1}/{MAX_RETRIES})..."):
                prompt_future = concurrent.futures.ThreadPoolExecutor().submit(
                    manager.agents['prompt_generator'].run, DEFAULT_RUN_PROMPT
                )
                prompt_response = prompt_future.result(timeout=API_TIMEOUT)

                raw_prompts = extract_json(prompt_response.content)
                try:
                    prompt_data = json.loads(raw_prompts)
                    debug_log("Section prompts generated successfully.")
                    break  # Success, break out of retry loop
                except Exception as e:
                    debug_log(f"Error parsing prompt JSON: {e}")
                    # If last attempt, use fallback
                    if attempt == MAX_RETRIES - 1:
                        prompt_data = generate_default_prompts(outline_data)
                        raw_prompts = json.dumps(prompt_data)
                    else:
                        time.sleep(RETRY_DELAY)  # Wait before retrying
        except (concurrent.futures.TimeoutError, TimeoutError):
            debug_log(f"Prompt generation timed out on attempt {attempt+1}")
            if attempt == MAX_RETRIES - 1:
                st.warning("Prompt generation timed out. Using fallback prompts.")
                prompt_data = generate_default_prompts(outline_data)
                raw_prompts = json.dumps(prompt_data)
            else:
                time.sleep(RETRY_DELAY)  # Wait before retrying
        except Exception as e:
            debug_log(f"Error during prompt generation: {e}")
            if attempt == MAX_RETRIES - 1:
                st.warning(f"Error generating prompts: {str(e)}. Using fallback prompts.")
                prompt_data = generate_default_prompts(outline_data)
                raw_prompts = json.dumps(prompt_data)
            else:
                time.sleep(RETRY_DELAY)  # Wait before retrying

    # Update animation
    update_loading_animation(animation_container, 2, 4, show_fun_fact())

    # Clear status
    status.empty()

    return {
        "outline": outline_data,
        "section_prompts": prompt_data,
        "raw_outline": raw_outline,
        "raw_prompts": raw_prompts
    }

# =============================================================================
# Validation Team Functions
# =============================================================================
def validate_outline(planning_data: dict, detail_level: str, topic: str, animation_container) -> dict:
    """
    Optionally validates the outline and section prompts. If valid data exists, skip validation.
    """
    if planning_data.get("outline", {}).get("sections") and planning_data.get("section_prompts", {}).get("section_prompts"):
        debug_log("Validation skipped; planning data is valid.")
        return planning_data
    else:
        manager = AgentManager(topic=topic, detail_level=detail_level)
        if not manager.create_validation_agent():
            st.warning("Failed to create validation agent. Proceeding with unvalidated data.")
            return planning_data

        # Determine number of sections based on detail level
        if detail_level == "TL;DR":
            num_sections = 5
        elif detail_level == "Executive Summary":
            num_sections = 10
        else:  # Detailed Summary
            num_sections = 15

        # Update animation
        update_loading_animation(animation_container, 2, 4, show_fun_fact())

        # Show status message
        status = st.empty()
        typewriter_text("Validating document structure...", status)

        validation_instructions = f"""
You are a narrative validator. Your task is to review the following document outline and section prompts.
Requirements:
1. Ensure the outline contains exactly {num_sections} sections, each with keys: section_number, title, and description.
2. Ensure each section prompt is detailed and suitable for creating LaTeX content.
3. If any part is missing, generate a revised version that meets these requirements.
4. Output only valid JSON with the following structure:
    For the outline:
    {{{{ "sections": [{{"section_number": 1, "title": "Section Title", "description": "Brief description."}}, ...] }}}}
    For the prompts:
    {{{{ "section_prompts": [{{"section_number": 1, "prompt": "Detailed prompt for section 1."}}, ...] }}}}

Topic: "{topic}"
Detail Level: {detail_level}

Outline:
{json.dumps(planning_data.get("outline", {}), indent=2)}
Section Prompts:
{json.dumps(planning_data.get("section_prompts", {}), indent=2)}
"""
        manager.agents['validator'].instructions = validation_instructions
        debug_log("Set instructions for Validator.")

        # Try validation with timeout and retry
        validated_data = planning_data
        for attempt in range(MAX_RETRIES):
            try:
                with st.spinner(f"Validating document structure (attempt {attempt+1}/{MAX_RETRIES})..."):
                    validation_future = concurrent.futures.ThreadPoolExecutor().submit(
                        manager.agents['validator'].run, DEFAULT_RUN_PROMPT
                    )
                    validation_response = validation_future.result(timeout=API_TIMEOUT)

                    raw_validated = extract_json(validation_response.content)
                    try:
                        validated_data = json.loads(raw_validated)
                        validated_data["raw_validated"] = raw_validated
                        debug_log("Validation complete successfully.")
                        break  # Success
                    except Exception as e:
                        debug_log(f"Error parsing validated JSON: {e}")
                        if attempt == MAX_RETRIES - 1:
                            validated_data = planning_data
                            validated_data["raw_validated"] = json.dumps(planning_data)
                        else:
                            time.sleep(RETRY_DELAY)
            except (concurrent.futures.TimeoutError, TimeoutError):
                debug_log(f"Validation timed out on attempt {attempt+1}")
                if attempt == MAX_RETRIES - 1:
                    st.warning("Validation timed out. Proceeding with unvalidated data.")
                    validated_data = planning_data
                    validated_data["raw_validated"] = json.dumps(planning_data)
                else:
                    time.sleep(RETRY_DELAY)
            except Exception as e:
                debug_log(f"Error during validation: {e}")
                if attempt == MAX_RETRIES - 1:
                    st.warning(f"Error during validation: {str(e)}. Proceeding with unvalidated data.")
                    validated_data = planning_data
                    validated_data["raw_validated"] = json.dumps(planning_data)
                else:
                    time.sleep(RETRY_DELAY)

        # Update animation
        update_loading_animation(animation_container, 3, 4, show_fun_fact())

        # Clear status
        status.empty()

        return validated_data

# =============================================================================
# Data Gatherer Functions (Section-by-Section)
# =============================================================================
def gather_single_section(section: dict, manager: AgentManager, topic: str) -> dict:
    """
    Generates raw content for a single section using the data gatherer agent.
    """
    section_number = section.get("section_number")
    prompt = section.get("prompt")
    title = section.get("title", f"Section {section_number}")

    # Show status
    status = st.empty()
    status.info(f"📝 Gathering content for section {section_number}: {title}")

    # Create fallback content
    fallback_content = f"""
# {title}

This is auto-generated content for section {section_number} on the topic of {topic}.

## Key Points

* This section covers important aspects of {title}
* It includes relevant information and examples
* It addresses the main points requested in the prompt

## Details

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

## Conclusion

This section provides valuable insights into {title} and connects to the overall document theme.
"""

    agent = manager.create_data_gatherer_agent(section_number)
    if not agent:
        debug_log(f"Failed to create agent for section {section_number}. Using fallback content.")
        status.empty()
        return {"section_number": section_number, "raw_content": fallback_content, "title": title}

    instructions = f"""
You are the Content Generator for section {section_number} ({title}). Using the following prompt, produce comprehensive content that will eventually be converted to LaTeX for a document.

Topic: "{topic}"

Prompt: "{prompt}"

Requirements:
1. Generate detailed content with headings, bullet points, and citations.
2. Focus on clarity and conciseness.
3. Format in a way that will be easy to convert to LaTeX later.
4. Output only the final content.
"""
    agent.instructions = instructions

    # Try to generate content with timeout and retry
    raw_content = fallback_content
    for attempt in range(MAX_RETRIES):
        try:
            with st.spinner(f"Generating content for section {section_number} (attempt {attempt+1}/{MAX_RETRIES})..."):
                response_future = concurrent.futures.ThreadPoolExecutor().submit(
                    agent.run, DEFAULT_RUN_PROMPT
                )
                response = response_future.result(timeout=API_TIMEOUT)

                raw_content = response.content.strip()
                if raw_content:
                    debug_log(f"Raw content for section {section_number} generated successfully.")
                    break  # Success
                else:
                    debug_log(f"Empty content received for section {section_number}")
                    if attempt == MAX_RETRIES - 1:
                        raw_content = fallback_content
                    else:
                        time.sleep(RETRY_DELAY)
        except (concurrent.futures.TimeoutError, TimeoutError):
            debug_log(f"Content generation for section {section_number} timed out on attempt {attempt+1}")
            if attempt == MAX_RETRIES - 1:
                st.warning(f"Content generation for section {section_number} timed out. Using fallback content.")
                raw_content = fallback_content
            else:
                time.sleep(RETRY_DELAY)
        except Exception as e:
            debug_log(f"Error generating content for section {section_number}: {e}")
            if attempt == MAX_RETRIES - 1:
                st.warning(f"Error generating content for section {section_number}: {str(e)}. Using fallback content.")
                raw_content = fallback_content
            else:
                time.sleep(RETRY_DELAY)

    # Clear status message
    status.empty()

    return {"section_number": section_number, "raw_content": raw_content, "title": title}

def gather_section_data(validated_data: dict, topic: str, animation_container) -> list:
    """
    Loops over each section prompt and generates raw content for each section.
    """
    sections = []
    section_prompts = validated_data.get("section_prompts", {}).get("section_prompts", [])
    debug_log(f"Found {len(section_prompts)} section prompts for content generation.")
    manager = AgentManager(topic=topic)

    # If no section prompts, create fallback
    if not section_prompts:
        st.warning("No section prompts found. Using fallback content.")
        outline_sections = validated_data.get("outline", {}).get("sections", [])
        for section in outline_sections:
            section_number = section.get("section_number")
            title = section.get("title", f"Section {section_number}")
            fallback_content = f"""
# {title}

This is auto-generated content for section {section_number} on the topic of {topic}.

## Key Points

* This section covers important aspects of {title}
* It includes relevant information and examples
* It addresses the main points in the outline

## Details

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

## Conclusion

This section provides valuable insights into {title} and connects to the overall document theme.
"""
            sections.append({"section_number": section_number, "raw_content": fallback_content, "title": title})
        return sections

    # Create progress bar
    progress_bar = st.progress(0)

    # Show fun fact
    fact_placeholder = st.empty()
    current_fact = show_fun_fact()
    fact_placeholder.info(f"**Did you know?** {current_fact}")

    for i, section in enumerate(section_prompts):
        # Update animation
        update_loading_animation(animation_container, 3, 4, None)

        # Update fun fact every 3 sections
        if i % 3 == 0 and i > 0:
            current_fact = show_fun_fact()
            fact_placeholder.info(f"**Did you know?** {current_fact}")

        # Add title to section data
        section_number = section.get("section_number")
        title = None
        for outline_section in validated_data.get("outline", {}).get("sections", []):
            if outline_section.get("section_number") == section_number:
                title = outline_section.get("title")
                break
        if title:
            section["title"] = title

        result = gather_single_section(section, manager, topic)
        sections.append(result)

        # Update progress
        progress = (i + 1) / len(section_prompts)
        progress_bar.progress(progress)

        time.sleep(0.5)

    # Clear fact placeholder
    fact_placeholder.empty()

    debug_log("Data Gatherer workflow complete.")
    return sections

# =============================================================================
# Design Team Functions
# =============================================================================
def design_sections(sections: list, topic: str, animation_container) -> list:
    """
    Enhances raw section content into well-formatted content.
    """
    manager = AgentManager(topic=topic)
    designed_sections = []

    # Create progress bar
    progress_bar = st.progress(0)

    # Show fun fact
    fact_placeholder = st.empty()
    current_fact = show_fun_fact()
    fact_placeholder.info(f"**Did you know?** {current_fact}")

    for i, section in enumerate(sections):
        # Update animation
        update_loading_animation(animation_container, 3, 4, None)

        # Update fun fact every 3 sections
        if i % 3 == 0 and i > 0:
            current_fact = show_fun_fact()
            fact_placeholder.info(f"**Did you know?** {current_fact}")

        section_number = section.get("section_number")
        title = section.get("title", f"Section {section_number}")
        raw_content = section.get("raw_content", "")

        status = st.empty()
        status.info(f"✨ Enhancing content for section {section_number}: {title}...")

        design_agent = manager.create_design_agent(section_number)
        if not design_agent:
            debug_log(f"Failed to create design agent for section {section_number}. Using raw content.")
            designed_sections.append({
                "section_number": section_number,
                "designed_content": raw_content,
                "raw_content": raw_content,
                "design_raw": raw_content,
                "title": title
            })
            # Update progress
            progress = (i + 1) / len(sections)
            progress_bar.progress(progress)
            # Clear status
            status.empty()
            continue

        design_instructions = f"""
You are the Document Design Agent for section {section_number} ({title}). Enhance the following content to produce a well-structured section for a professional LaTeX document.

Topic: "{topic}"

Content to enhance:
\"\"\"{raw_content}\"\"\"

Requirements:
1. Improve formatting and structure.
2. Use markdown that can be easily converted to LaTeX.
3. Suggest image placeholders or diagrams where appropriate.
4. Use engaging headings, bullet points, and call-out boxes.
5. Preserve all content, formatting, and data from the original text.
6. Output only the final enhanced content.
"""
        design_agent.instructions = design_instructions

        # Try to design content with timeout and retry
        designed_content = raw_content
        design_raw = raw_content
        for attempt in range(MAX_RETRIES):
            try:
                with st.spinner(f"Enhancing content for section {section_number} (attempt {attempt+1}/{MAX_RETRIES})..."):
                    design_future = concurrent.futures.ThreadPoolExecutor().submit(
                        design_agent.run, DEFAULT_RUN_PROMPT
                    )
                    design_response = design_future.result(timeout=API_TIMEOUT)

                    design_raw = design_response.content.strip()
                    designed_content = design_raw
                    if designed_content:
                        debug_log(f"Designed content for section {section_number} generated successfully.")
                        break  # Success
                    else:
                        debug_log(f"Empty design content received for section {section_number}")
                        if attempt == MAX_RETRIES - 1:
                            designed_content = raw_content
                            design_raw = raw_content
                        else:
                            time.sleep(RETRY_DELAY)
            except (concurrent.futures.TimeoutError, TimeoutError):
                debug_log(f"Design for section {section_number} timed out on attempt {attempt+1}")
                if attempt == MAX_RETRIES - 1:
                    st.warning(f"Design for section {section_number} timed out. Using raw content.")
                    designed_content = raw_content
                    design_raw = raw_content
                else:
                    time.sleep(RETRY_DELAY)
            except Exception as e:
                debug_log(f"Error designing content for section {section_number}: {e}")
                if attempt == MAX_RETRIES - 1:
                    st.warning(f"Error designing content for section {section_number}: {str(e)}. Using raw content.")
                    designed_content = raw_content
                    design_raw = raw_content
                else:
                    time.sleep(RETRY_DELAY)

        designed_sections.append({
            "section_number": section_number,
            "designed_content": designed_content,
            "raw_content": raw_content,
            "design_raw": design_raw,
            "title": title
        })

        # Update progress
        progress = (i + 1) / len(sections)
        progress_bar.progress(progress)

        # Clear status
        status.empty()

        time.sleep(0.5)

    # Clear fact placeholder
    fact_placeholder.empty()

    return designed_sections

# =============================================================================
# FIXED: LaTeX Formatter Function with Token Management
# =============================================================================
def format_to_latex(designed_sections: list, topic: str, user_input: str, detail_level: str, animation_container) -> str:
    """
    Takes the designed sections and converts them to a complete LaTeX document with token management.
    """
    manager = AgentManager(topic=topic, user_input=user_input, detail_level=detail_level)
    latex_agent = manager.create_latex_formatter_agent()

    # Combine all designed section content
    combined_content = ""
    for section in sorted(designed_sections, key=lambda x: x['section_number']):
        title = section.get("title", f"Section {section['section_number']}")
        combined_content += f"\n\n--- SECTION {section['section_number']}: {title} ---\n\n"
        combined_content += section.get("designed_content", "")

    # Update animation
    update_loading_animation(animation_container, 4, 4, show_fun_fact())

    # Show status
    status = st.empty()
    typewriter_text("Converting content to LaTeX format... This may take a moment.", status)

    # Fallback LaTeX document
    fallback_latex = DEFAULT_LATEX_TEMPLATE.replace("DOCUMENT_TITLE", topic)
    fallback_latex = fallback_latex.replace("DOCUMENT_AUTHOR", "Generated by AI Document Assistant")
    fallback_latex = fallback_latex.replace("DOCUMENT_FOOTER", "Generated Document")

    # Create basic section content for fallback
    section_content = ""
    for section in sorted(designed_sections, key=lambda x: x['section_number']):
        section_number = section.get("section_number")
        title = section.get("title", f"Section {section_number}")
        content = section.get("designed_content", "").strip()

        # Convert markdown headings to LaTeX sections
        content = re.sub(r"^# (.+)$", r"\\section{\1}", content, flags=re.MULTILINE)
        content = re.sub(r"^## (.+)$", r"\\subsection{\1}", content, flags=re.MULTILINE)
        content = re.sub(r"^### (.+)$", r"\\subsubsection{\1}", content, flags=re.MULTILINE)

        # Convert markdown bullet points to LaTeX itemize
        content = re.sub(r"^\* (.+)$", r"\\item \1", content, flags=re.MULTILINE)
        content = content.replace("\n* ", "\n\\item ")

        # Wrap bullet points in itemize environment
        if "\\item" in content:
            content = "\\begin{itemize}\n" + content + "\n\\end{itemize}\n"

        # Escape special LaTeX characters (basic escaping)
        escapes = [("&", "\\&"), ("%", "\\%"), ("$", "\\$"), ("#", "\\#"),
                  ("_", "\\_"), ("{", "\\{"), ("}", "\\}")]
        for old, new in escapes:
            content = content.replace(old, new)

        section_content += f"\\section{{{title}}}\n\n{content}\n\n"

    fallback_latex = fallback_latex.replace("% CONTENT_PLACEHOLDER", section_content)

    if not latex_agent:
        debug_log("Failed to create LaTeX formatter agent. Using fallback LaTeX document.")
        status.empty()
        return fallback_latex

    # Check if content is too large for a single request
    content_tokens = estimate_tokens(combined_content)
    if content_tokens > 8000:
        st.warning(f"Content is large ({content_tokens} estimated tokens). Using chunked processing...")

        # Process sections in smaller chunks
        latex_document = process_sections_in_chunks(designed_sections, topic, latex_agent)
        status.empty()
        return latex_document

    # Prepare the LaTeX formatter instructions
    latex_instructions = f"""
You are a LaTeX expert. Convert the following section content to a complete LaTeX document.

DOCUMENT TOPIC: {topic}
DETAIL LEVEL: {detail_level}

LATEX TEMPLATE:
```latex
{DEFAULT_LATEX_TEMPLATE}
```

SECTION CONTENT TO CONVERT:
```
{combined_content}
```

Requirements:
1. Replace DOCUMENT_TITLE with an appropriate title based on the topic: "{topic}".
2. Replace DOCUMENT_AUTHOR with "Generated by AI Document Assistant".
3. Replace DOCUMENT_FOOTER with an appropriate footer.
4. Replace CONTENT_PLACEHOLDER with properly formatted LaTeX content for each section.
5. Each section should be a proper LaTeX section with appropriate subsections.
6. Format content with proper LaTeX syntax (bullet lists, emphasis, tables where appropriate).
7. For any charts/figures mentioned, include suitable TikZ or pgfplots code to create them.
8. Include all content from the markdown - ensure no information is lost during conversion.
9. Handle code blocks properly using the listings package.
10. Preserve all formatting including headings, lists, emphasis, tables, etc.
11. Properly escape all special LaTeX characters like %, $, _, etc.
12. Output a complete, compilable LaTeX document.

IMPORTANT: Ensure ALL content from the markdown sections is converted to LaTeX. Do not summarize or omit content.
"""
    latex_agent.instructions = latex_instructions

    # Try to format LaTeX with timeout and retry
    latex_document = fallback_latex
    for attempt in range(MAX_RETRIES):
        try:
            with st.spinner(f"Converting to LaTeX (attempt {attempt+1}/{MAX_RETRIES})..."):
                latex_future = concurrent.futures.ThreadPoolExecutor().submit(
                    latex_agent.run, DEFAULT_RUN_PROMPT
                )
                latex_response = latex_future.result(timeout=API_TIMEOUT * 2)  # Double timeout for LaTeX conversion

                latex_content = latex_response.content.strip()

                # Extract the complete LaTeX document (if it's wrapped in code fences)
                latex_match = re.search(r"```(?:latex)?\s*(.*?)```", latex_content, re.DOTALL)
                if latex_match:
                    latex_document = latex_match.group(1)
                else:
                    latex_document = latex_content

                if latex_document and "\\begin{document}" in latex_document:
                    debug_log("LaTeX conversion complete successfully.")
                    break  # Success
                else:
                    debug_log("Invalid LaTeX document received")
                    if attempt == MAX_RETRIES - 1:
                        latex_document = fallback_latex
                    else:
                        time.sleep(RETRY_DELAY)
        except (concurrent.futures.TimeoutError, TimeoutError):
            debug_log(f"LaTeX conversion timed out on attempt {attempt+1}")
            if attempt == MAX_RETRIES - 1:
                st.warning("LaTeX conversion timed out. Using fallback LaTeX document.")
                latex_document = fallback_latex
            else:
                time.sleep(RETRY_DELAY)
        except Exception as e:
            debug_log(f"Error during LaTeX conversion: {e}")
            if attempt == MAX_RETRIES - 1:
                st.warning(f"Error during LaTeX conversion: {str(e)}. Using fallback LaTeX document.")
                latex_document = fallback_latex
            else:
                time.sleep(RETRY_DELAY)

    # Clear status
    status.empty()

    # Show a celebration animation
    st.balloons()

    return latex_document

def process_sections_in_chunks(designed_sections: list, topic: str, latex_agent) -> str:
    """
    Process sections in smaller chunks to avoid token limits.
    """
    # Create the base LaTeX document
    base_latex = DEFAULT_LATEX_TEMPLATE.replace("DOCUMENT_TITLE", topic)
    base_latex = base_latex.replace("DOCUMENT_AUTHOR", "Generated by AI Document Assistant")
    base_latex = base_latex.replace("DOCUMENT_FOOTER", "Generated Document")

    # Process sections individually and combine
    all_section_content = ""

    for section in sorted(designed_sections, key=lambda x: x['section_number']):
        section_number = section.get("section_number")
        title = section.get("title", f"Section {section_number}")
        content = section.get("designed_content", "").strip()

        # Convert this single section
        section_latex = convert_single_section_to_latex(title, content, latex_agent)
        all_section_content += section_latex + "\n\n"

    # Replace the placeholder with all section content
    final_latex = base_latex.replace("% CONTENT_PLACEHOLDER", all_section_content)
    return final_latex

def convert_single_section_to_latex(title: str, content: str, latex_agent) -> str:
    """
    Convert a single section to LaTeX format.
    """
    if not latex_agent:
        # Fallback conversion
        content = re.sub(r"^# (.+)$", r"\\subsection{\1}", content, flags=re.MULTILINE)
        content = re.sub(r"^## (.+)$", r"\\subsubsection{\1}", content, flags=re.MULTILINE)
        content = re.sub(r"^\* (.+)$", r"\\item \1", content, flags=re.MULTILINE)

        if "\\item" in content:
            content = "\\begin{itemize}\n" + content + "\n\\end{itemize}\n"

        # Basic escaping
        escapes = [("&", "\\&"), ("%", "\\%"), ("$", "\\$"), ("#", "\\#"),
                  ("_", "\\_"), ("{", "\\{"), ("}", "\\}")]
        for old, new in escapes:
            content = content.replace(old, new)

        return f"\\section{{{title}}}\n\n{content}"

    # Use agent for conversion
    instructions = f"""
Convert the following single section to LaTeX format:

Section Title: {title}
Content:
```
{content}
```

Requirements:
1. Start with \\section{{{title}}}
2. Convert markdown to proper LaTeX
3. Use \\subsection and \\subsubsection for headings
4. Convert bullet points to itemize environments
5. Escape special characters properly
6. Output only the LaTeX code for this section
"""

    try:
        latex_agent.instructions = instructions
        response = latex_agent.run(DEFAULT_RUN_PROMPT)

        section_latex = response.content.strip()
        # Clean up if wrapped in code fences
        latex_match = re.search(r"```(?:latex)?\s*(.*?)```", section_latex, re.DOTALL)
        if latex_match:
            section_latex = latex_match.group(1)

        return section_latex
    except Exception as e:
        debug_log(f"Error converting section '{title}' to LaTeX: {e}")
        # Fallback to basic conversion
        return convert_single_section_to_latex(title, content, None)

# =============================================================================
# Document generation workflow
# =============================================================================
def generate_document(user_input, topic, detail_level):
    """Complete document generation workflow that saves state at each step"""

    # Generate a session ID for this document generation session
    session_id = str(uuid.uuid4())
    st.session_state.current_session_id = session_id

    # Initialize session data
    st.session_state.generated_documents[session_id] = {
        "topic": topic,
        "detail_level": detail_level,
        "user_input": user_input,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "started",
        "steps_completed": [],
        "data": {}
    }

    try:
        # Create animation container
        animation_container = create_loading_animation()

        # Step 1: Planning Team
        stage_header = st.subheader("1. Planning Stage")
        with st.spinner("Planning document structure..."):
            planning_data = planning_team(user_input, detail_level, topic, animation_container)
            st.session_state.generated_documents[session_id]["data"]["planning"] = planning_data
            st.session_state.generated_documents[session_id]["steps_completed"].append("planning")
        st.success("✅ Document planning complete!")

        # Step 2: Validation
        stage_header.write("2. Validation Stage")
        with st.spinner("Validating document structure..."):
            validated_data = validate_outline(planning_data, detail_level, topic, animation_container)
            st.session_state.generated_documents[session_id]["data"]["validation"] = validated_data
            st.session_state.generated_documents[session_id]["steps_completed"].append("validation")
        st.success("✅ Document structure validated!")

        # Step 3: Data Gathering
        stage_header.write("3. Content Generation Stage")
        with st.spinner("Generating section content..."):
            raw_sections = gather_section_data(validated_data, topic, animation_container)
            st.session_state.generated_documents[session_id]["data"]["raw_sections"] = raw_sections
            st.session_state.generated_documents[session_id]["steps_completed"].append("content")
        st.success("✅ Section content generated!")

        # Step 4: Design
        stage_header.write("4. Design Enhancement Stage")
        with st.spinner("Enhancing content design..."):
            designed_sections = design_sections(raw_sections, topic, animation_container)
            st.session_state.generated_documents[session_id]["data"]["designed_sections"] = designed_sections
            st.session_state.generated_documents[session_id]["steps_completed"].append("design")
        st.success("✅ Content design enhancements complete!")

        # Step 5: LaTeX Formatting
        stage_header.write("5. LaTeX Conversion Stage")
        with st.spinner("Converting to LaTeX format..."):
            latex_document = format_to_latex(designed_sections, topic, user_input, detail_level, animation_container)
            st.session_state.generated_documents[session_id]["data"]["latex"] = latex_document
            st.session_state.generated_documents[session_id]["steps_completed"].append("latex")

            # Create combined markdown document
            markdown_document = create_combined_markdown(designed_sections, planning_data.get("outline", {}), topic)
            st.session_state.generated_documents[session_id]["data"]["markdown"] = markdown_document

            # Save files to disk
            safe_topic = re.sub(r'[^\w\-_\. ]', '_', topic)
            md_path = save_file_to_disk(markdown_document, safe_topic, "md")
            latex_path = save_file_to_disk(latex_document, safe_topic, "tex")
            st.session_state.generated_documents[session_id]["file_paths"] = {
                "markdown": str(md_path),
                "latex": str(latex_path)
            }

        st.success("✅ LaTeX conversion complete!")

        # Clear animation container
        animation_container.empty()

        # Mark document as complete
        st.session_state.generated_documents[session_id]["status"] = "complete"

        # Return the data needed for display
        return {
            "session_id": session_id,
            "designed_sections": designed_sections,
            "planning_data": planning_data,
            "markdown_document": markdown_document,
            "latex_document": latex_document,
            "file_paths": {
                "markdown": md_path,
                "latex": latex_path
            }
        }

    except Exception as e:
        st.session_state.generated_documents[session_id]["status"] = "error"
        st.session_state.generated_documents[session_id]["error"] = str(e)
        st.session_state.generated_documents[session_id]["traceback"] = traceback.format_exc()
        raise e

# =============================================================================
# Display Functions
# =============================================================================
def display_sections(sections: list):
    st.header("Document Content")
    for section in sorted(sections, key=lambda x: x['section_number']):
        section_title = section.get("title", f"Section {section['section_number']}")
        st.subheader(section_title)
        st.markdown(section.get("designed_content", ""), unsafe_allow_html=True)
        st.markdown("---")

# =============================================================================
# IMPROVED: Display Document Result Function with Enhanced PDF Generation
# =============================================================================
def display_document_result(result):
    """Display the generated document results with enhanced PDF generation"""

    # Display the designed content
    display_sections(result["designed_sections"])

    # Display the LaTeX code without using an expander
    st.header("LaTeX Code")
    show_latex = st.checkbox("Show LaTeX Source Code", value=False)
    if show_latex:
        st.code(result["latex_document"], language="latex")

    # Download and file paths section
    st.header("Document Files")

    # Show file paths
    md_path = result["file_paths"]["markdown"]
    latex_path = result["file_paths"]["latex"]

    st.info(f"""
    Files have been automatically saved to the project directory:

    - **Markdown**: {md_path}
    - **LaTeX**: {latex_path}
    """)

    # Download buttons in columns to save space
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Download Markdown",
            data=result["markdown_document"],
            file_name=f"{os.path.basename(md_path)}",
            mime="text/markdown",
            key=f"md_{result['session_id']}"
        )

    with col2:
        st.download_button(
            label="Download LaTeX Source",
            data=result["latex_document"],
            file_name=f"{os.path.basename(latex_path)}",
            mime="text/plain",
            key=f"latex_{result['session_id']}"
        )

    # PDF section with enhanced error handling
    st.header("PDF Preview")

    # Add PDF compilation button
    if st.button("🔄 Compile PDF", type="primary"):
        with st.spinner("Compiling LaTeX to PDF..."):
            # Generate a safe filename from topic
            safe_topic = re.sub(r'[^\w\-_\. ]', '_', result.get("planning_data", {}).get("outline", {}).get("title", result.get("planning_data", {}).get("topic", "document")))

            debug_log(f"Starting PDF compilation for topic: {safe_topic}")

            # Try to compile PDF with improved function
            success, pdf_base64, log, pdf_path = compile_latex_to_pdf(
                result["latex_document"],
                safe_topic
            )

            if success:
                st.success("🎉 PDF compilation successful!")
                debug_log("PDF compilation successful in Streamlit")

                # Show PDF path
                if pdf_path:
                    st.info(f"PDF saved to: {pdf_path}")

                # Download button for PDF
                st.download_button(
                    label="📥 Download PDF",
                    data=base64.b64decode(pdf_base64),
                    file_name=f"{os.path.basename(str(pdf_path)) if pdf_path else 'document.pdf'}",
                    mime="application/pdf",
                    key=f"pdf_{result['session_id']}"
                )

                # Display PDF preview
                st.subheader("PDF Preview")
                pdf_display = f'<iframe src="data:application/pdf;base64,{pdf_base64}" width="100%" height="600" type="application/pdf"></iframe>'
                st.markdown(pdf_display, unsafe_allow_html=True)

                # Show balloons for celebration
                st.balloons()

            else:
                st.error("❌ PDF compilation failed.")
                debug_log("PDF compilation failed in Streamlit")

                # Show troubleshooting information
                with st.expander("📋 Troubleshooting Information & Solutions"):
                    st.markdown("### Compilation Log")
                    st.code(log, language="text")

                    st.markdown("### Quick Solutions")
                    st.markdown("""
                    **Option 1: Install PDFLaTeX Python Library (Recommended)**
                    ```bash
                    pip install pdflatex
                    ```

                    **Option 2: Install LaTeX Distribution**
                    - **Windows**: Install [MiKTeX](https://miktex.org/) or [TeX Live](https://www.tug.org/texlive/)
                    - **macOS**: Install [MacTeX](https://www.tug.org/mactex/)
                    - **Linux**: `sudo apt-get install texlive-full` (Ubuntu/Debian)

                    **Option 3: Use Online LaTeX Editor**
                    1. Copy the LaTeX source code above
                    2. Go to [Overleaf](https://www.overleaf.com)
                    3. Create a new project and paste the code
                    4. Compile and download the PDF

                    **Option 4: Docker Solution**
                    ```bash
                    docker run --rm -v $(pwd):/workspace texlive/texlive pdflatex document.tex
                    ```
                    """)

                    st.markdown("### Common Issues")
                    st.markdown("""
                    - **Missing packages**: The LaTeX template uses common packages, but some may be missing
                    - **Special characters**: Some text may contain characters that need escaping
                    - **Math expressions**: Complex math might need additional packages
                    - **Encoding issues**: Try saving with UTF-8 encoding
                    - **Permissions**: Make sure the output directory is writable
                    """)

                    st.markdown("### Debug Information")
                    st.markdown(f"""
                    - **PDFLaTeX Library Available**: {PDFLATEX_AVAILABLE}
                    - **Output Directory**: {OUTPUT_DIR.absolute()}
                    - **LaTeX Document Length**: {len(result["latex_document"])} characters
                    """)

# =============================================================================
# MAIN: Streamlit Application Function with API Key Management
# =============================================================================
def main():
    st.title("Enhanced Document Generator")
    st.write(
        "Convert your text into a professional LaTeX document with just a few clicks."
    )

    # Configure API keys in sidebar
    api_keys_configured = configure_api_keys()

    st.sidebar.title("Options")
    st.sidebar.info(
        "This application uses a multi-agent workflow to generate a comprehensive, LaTeX-formatted document."
    )

    # Check dependencies and show status
    if not AGNO_AVAILABLE:
        st.error("⚠️ Agno framework not available. Please install with: `pip install agno`")
        return

    if not api_keys_configured:
        st.warning("🔑 Please configure your API keys in the sidebar to continue.")
        st.info("""
        **Required API Keys:**
        - OpenAI API Key (for GPT-4 models)
        - Anthropic API Key (for Claude models)

        You can get these from:
        - OpenAI: https://platform.openai.com/api-keys
        - Anthropic: https://console.anthropic.com/
        """)
        return

    # Check if we're displaying a previous generation or starting new
    if st.session_state.current_session_id and "generated_documents" in st.session_state:
        # Show "Start New" button at the top
        if st.button("Start New Document", type="primary"):
            # Reset current session ID but keep document history
            st.session_state.current_session_id = None
            st.rerun()

        # Display the current document
        current_doc = st.session_state.generated_documents[st.session_state.current_session_id]

        if current_doc["status"] == "complete":
            st.header(f"Document: {current_doc['topic']}")
            st.subheader(f"Generated on {current_doc['timestamp']}")

            # Display the document content
            if "data" in current_doc and all(k in current_doc["data"] for k in ["designed_sections", "planning", "markdown", "latex"]):
                result = {
                    "session_id": st.session_state.current_session_id,
                    "designed_sections": current_doc["data"]["designed_sections"],
                    "planning_data": current_doc["data"]["planning"],
                    "markdown_document": current_doc["data"]["markdown"],
                    "latex_document": current_doc["data"]["latex"],
                    "file_paths": {
                        "markdown": Path(current_doc["file_paths"]["markdown"]) if "file_paths" in current_doc else None,
                        "latex": Path(current_doc["file_paths"]["latex"]) if "file_paths" in current_doc else None,
                    }
                }
                display_document_result(result)
            else:
                st.error("Document data incomplete. Please generate a new document.")
        elif current_doc["status"] == "error":
            st.error(f"An error occurred during generation: {current_doc.get('error', 'Unknown error')}")

            with st.expander("Error Details"):
                st.code(current_doc.get("traceback", "No traceback available"))

            if st.button("Start Over"):
                st.session_state.current_session_id = None
                st.rerun()
        else:
            st.info("Document generation in progress...")

    else:
        # Input area for user text
        topic = st.text_input("Document Topic", value="ChatGPT for Marketing")
        user_input = st.text_area("Input Text (up to 20,000 characters)",
                                height=300,
                                max_chars=20000,
                                placeholder="Paste your text here...")

        # Detail level selection
        detail_level = st.radio(
            "Select Detail Level",
            ["TL;DR", "Executive Summary", "Detailed Summary"],
            index=1,
            help="TL;DR (5 sections), Executive Summary (10 sections), Detailed Summary (15 sections)"
        )

        # Generate document button
        if st.button("Generate Document", type="primary"):
            if not user_input:
                st.error("Please enter some text to generate a document.")
                return

            # Check if topic is provided
            if not topic:
                st.error("Please provide a document topic.")
                return

            st.markdown("### 🚀 Processing your document!")
            st.markdown("We'll transform your input into a professionally formatted LaTeX document.")

            try:
                # Generate document using our workflow function
                result = generate_document(user_input, topic, detail_level)

                # Display the document
                display_document_result(result)

                st.success("🎉 Document generation complete!")

                # Final celebration
                if not st.session_state.get('celebrated', False):
                    st.balloons()
                    st.session_state['celebrated'] = True

            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
                debug_log(f"Unexpected error: {traceback.format_exc()}")
                st.info("Try using a smaller document or the TL;DR option to reduce complexity.")

    st.markdown(
        """
        ---
        **About This Application:**

        This application leverages a multi-agent workflow to generate a professional LaTeX document from your text.
        The process includes:
          1. **Planning Team:** Analyzes your text and generates an outline.
          2. **Data Validator:** Reviews and refines the outline and prompts.
          3. **Content Generation:** Produces raw content for each section.
          4. **Design Enhancement:** Improves the content for better structure.
          5. **LaTeX Formatting:** Converts the content to LaTeX format.
          6. **PDF Generation:** Compiles the LaTeX code to a downloadable PDF.

        **Note:** The generation process may take several minutes depending on the text length and detail level.
        Documents are automatically saved to the "generated_documents" folder in the project directory.
        """
    )

# =============================================================================
# IMPROVED: Diagnostics Function with Enhanced Checks
# =============================================================================
def run_diagnostics_sidebar():
    # Use checkbox instead of expander for diagnostics
    show_diagnostics = st.sidebar.checkbox("Show Diagnostics Checklist", value=False)
    if show_diagnostics:
        st.sidebar.markdown("### Diagnostics Checklist")
        diag_results = []

        # Check API keys
        openai_key, anthropic_key, claude_model = get_api_keys()

        if openai_key:
            diag_results.append("- ✅ OPENAI_API_KEY is set.")
        else:
            diag_results.append("- ❌ OPENAI_API_KEY is missing.")
        if anthropic_key:
            diag_results.append("- ✅ ANTHROPIC_API_KEY is set.")
        else:
            diag_results.append("- ❌ ANTHROPIC_API_KEY is missing.")
        if claude_model:
            diag_results.append(f"- ✅ CLAUDE_MODEL_ID is set to `{claude_model}`.")
        else:
            diag_results.append("- ❌ CLAUDE_MODEL_ID is missing.")

        # Check for pdflatex Python library
        if PDFLATEX_AVAILABLE:
            diag_results.append("- ✅ PDFLaTeX Python library detected.")
        else:
            diag_results.append("- ⚠️ PDFLaTeX Python library missing. Install with: pip install pdflatex")

        # Check for LaTeX installation
        try:
            result = subprocess.run(["pdflatex", "--version"], capture_output=True, timeout=5)
            if result.returncode == 0:
                diag_results.append("- ✅ LaTeX installation detected.")
            else:
                diag_results.append("- ⚠️ LaTeX installation check failed.")
        except subprocess.TimeoutExpired:
            diag_results.append("- ⚠️ LaTeX installation check timed out.")
        except FileNotFoundError:
            diag_results.append("- ⚠️ LaTeX installation not found. Install TeX Live, MacTeX, or MiKTeX.")
        except Exception as e:
            diag_results.append(f"- ⚠️ LaTeX check error: {str(e)}")

        # Check required libraries
        if AGNO_AVAILABLE:
            diag_results.append("- ✅ Agno library detected.")
        else:
            diag_results.append("- ❌ Agno library missing.")

        if RESEARCH_TOOLS_AVAILABLE:
            diag_results.append("- ✅ Research tools available.")
        else:
            diag_results.append("- ⚠️ Research tools missing (optional).")

        try:
            import concurrent.futures
            diag_results.append("- ✅ Concurrent futures library detected.")
        except ImportError:
            diag_results.append("- ❌ Concurrent futures library missing.")

        # Check output directory
        if OUTPUT_DIR.exists():
            diag_results.append(f"- ✅ Output directory exists at {OUTPUT_DIR.absolute()}")
        else:
            diag_results.append(f"- ❌ Output directory missing")

        # Check Python version
        python_version = sys.version_info
        if python_version >= (3, 8):
            diag_results.append(f"- ✅ Python {python_version.major}.{python_version.minor} detected.")
        else:
            diag_results.append(f"- ⚠️ Python {python_version.major}.{python_version.minor} detected. Python 3.8+ recommended.")

        for item in diag_results:
            st.sidebar.markdown(item)

    # Show document history
    if "generated_documents" in st.session_state and st.session_state.generated_documents:
        st.sidebar.markdown("### Document History")

        # List all documents by date
        for session_id, doc in sorted(
            st.session_state.generated_documents.items(),
            key=lambda x: x[1]["timestamp"],
            reverse=True
        ):
            if doc["status"] == "complete":
                if st.sidebar.button(f"{doc['topic']} ({doc['timestamp']})", key=f"history_{session_id}"):
                    st.session_state.current_session_id = session_id
                    st.rerun()

# =============================================================================
# Execution Entry Point
# =============================================================================
if __name__ == "__main__":
    # Run diagnostics sidebar
    run_diagnostics_sidebar()
    main()

# =============================================================================
# End of File: app.py
# =============================================================================
