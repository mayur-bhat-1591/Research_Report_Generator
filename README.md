A sophisticated multi-agent AI application that transforms raw text into professionally formatted LaTeX documents with automatic PDF generation. Built with Streamlit and powered by OpenAI GPT-4 and Anthropic Claude models.
🌟 Features

Multi-Agent Workflow: Utilizes specialized AI agents for planning, content generation, validation, design enhancement, and LaTeX formatting
Flexible Detail Levels: Choose from TL;DR (5 sections), Executive Summary (10 sections), or Detailed Summary (15 sections)
Professional Output: Generates publication-ready LaTeX documents with proper formatting, tables, figures, and citations
PDF Generation: Automatic compilation to PDF with multiple fallback options
Web Interface: User-friendly Streamlit interface with real-time progress tracking
Document Management: Automatic file saving with session history and document recovery
Research Integration: Optional web research capabilities using DuckDuckGo and Newspaper4k
Robust Error Handling: Comprehensive fallback mechanisms and retry logic
Diagnostics: Built-in system diagnostics and troubleshooting tools

🚀 Quick Start
Prerequisites

Python 3.8 or higher
OpenAI API key
Anthropic API key
(Optional) LaTeX distribution for local PDF compilation

Installation

Clone the repository
bashgit clone https://github.com/yourusername/enhanced-document-generator.git
cd enhanced-document-generator

Install dependencies
bashpip install -r requirements.txt

Set up environment variables
Create a .env file in the project root:
envOPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
CLAUDE_MODEL_ID=claude-sonnet-4-20250514

Run the application
bashstreamlit run app.py

Open your browser
Navigate to http://localhost:8501

📦 Dependencies
Core Requirements
txtstreamlit>=1.28.0
agno>=0.1.0
openai>=1.0.0
anthropic>=0.25.0
python-dotenv>=1.0.0
Optional Dependencies
txtpdflatex>=0.1.3          # For enhanced PDF generation
duckduckgo-search>=3.0.0 # For web research capabilities
newspaper4k>=0.9.0       # For article parsing
System Dependencies (Optional)
For local PDF compilation, install a LaTeX distribution:

Windows: MiKTeX or TeX Live
macOS: MacTeX
Linux: sudo apt-get install texlive-full (Ubuntu/Debian)

🏗️ Architecture
The application follows a multi-agent architecture with specialized AI agents:
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Planning Team  │    │ Validation Team │    │ Content Team    │
│                 │    │                 │    │                 │
│ • Outline Gen   │───▶│ • Story Valid   │───▶│ • Data Gatherer │
│ • Prompt Gen    │    │                 │    │ • Design Agent  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │  LaTeX Team     │
                                               │                 │
                                               │ • LaTeX Format  │
                                               │ • PDF Compiler  │
                                               └─────────────────┘
Workflow Stages

Planning Stage: Analyzes input text and generates document outline
Validation Stage: Reviews and refines document structure
Content Generation: Creates detailed content for each section
Design Enhancement: Improves formatting and presentation
LaTeX Conversion: Transforms content to professional LaTeX format
PDF Generation: Compiles LaTeX to downloadable PDF

🎯 Usage
Basic Usage

Enter Document Topic: Provide a clear topic for your document
Input Text: Paste your raw text (up to 20,000 characters)
Select Detail Level: Choose the appropriate complexity level
Generate Document: Click the generate button and wait for processing

Detail Levels
LevelSectionsBest ForTL;DR5 sectionsQuick summaries, brief reportsExecutive Summary10 sectionsBusiness reports, presentationsDetailed Summary15 sectionsAcademic papers, comprehensive analyses
Output Formats

Markdown: Human-readable format for editing
LaTeX: Professional typesetting format
PDF: Publication-ready document

🔧 Configuration
Environment Variables
VariableDescriptionDefaultOPENAI_API_KEYOpenAI API keyRequiredANTHROPIC_API_KEYAnthropic API keyRequiredCLAUDE_MODEL_IDClaude model identifierclaude-sonnet-4-20250514
Advanced Configuration
Modify these constants in app.py for fine-tuning:
pythonAPI_TIMEOUT = 120      # API request timeout (seconds)
MAX_RETRIES = 3        # Maximum retry attempts
RETRY_DELAY = 5        # Delay between retries (seconds)
🛠️ Troubleshooting
Common Issues
PDF Generation Fails

Install the pdflatex Python library: pip install pdflatex
Install a LaTeX distribution (see System Dependencies)
Use the online LaTeX editor fallback (Overleaf)

API Timeout Errors

Check your internet connection
Verify API keys are correct and have sufficient credits
Try reducing the detail level for large documents

Memory Issues with Large Documents

Use TL;DR mode for very large inputs
Break large documents into smaller sections
Increase system memory allocation

Diagnostics
The application includes built-in diagnostics accessible via the sidebar:

API key validation
LaTeX installation detection
Library dependency checks
Output directory verification

🗂️ File Structure
enhanced-document-generator/
├── app.py                 # Main application file
├── .env                   # Environment variables (create this)
├── requirements.txt       # Python dependencies
├── generated_documents/   # Output directory (auto-created)
├── README.md             # This file
└── .gitignore           # Git ignore file
🤝 Contributing
We welcome contributions! Please follow these steps:

Fork the repository
Create a feature branch (git checkout -b feature/amazing-feature)
Commit your changes (git commit -m 'Add amazing feature')
Push to the branch (git push origin feature/amazing-feature)
Open a Pull Request

Development Setup
bash# Clone your fork
git clone https://github.com/yourusername/enhanced-document-generator.git

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If available

# Run tests
python -m pytest tests/  # If tests are available

# Run the application
streamlit run app.py
📈 Performance
Typical Processing Times
Detail LevelSmall Text (1-2 pages)Medium Text (3-5 pages)Large Text (5+ pages)TL;DR30-60 seconds60-90 seconds90-120 secondsExecutive60-120 seconds120-180 seconds180-300 secondsDetailed120-240 seconds240-360 seconds360-600 seconds
Times may vary based on API response times and system performance
🔒 Security & Privacy

API Keys: Store securely in .env file, never commit to repository
Data Processing: Text is processed by OpenAI and Anthropic APIs
Local Storage: Generated documents are saved locally only
No Data Retention: The application doesn't store your data permanently

📚 API Documentation
Key Classes
AgentManager: Manages AI agent lifecycle and coordination
pythonmanager = AgentManager(topic="My Topic", user_input="...", detail_level="Executive Summary")
Document Generation: Main workflow function
pythonresult = generate_document(user_input, topic, detail_level)
Utility Functions

compile_latex_to_pdf(): Converts LaTeX to PDF
save_file_to_disk(): Handles file I/O operations
extract_json(): Parses JSON from AI responses

🎨 Customization
LaTeX Template
Modify the DEFAULT_LATEX_TEMPLATE in app.py to customize document appearance:

Change fonts, colors, and styling
Add custom packages
Modify page layout and margins

AI Instructions
Customize agent instructions to change output style:

Adjust tone and formality
Modify section structure
Change content focus areas

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
🙏 Acknowledgments

Streamlit - Web application framework
OpenAI - GPT-4 language model
Anthropic - Claude language model
Agno - Agent framework
LaTeX - Document preparation system

📞 Support

Issues: GitHub Issues
Discussions: GitHub Discussions
Email: your.email@example.com

🗺️ Roadmap

 Support for additional AI models (GPT-4 Turbo, Claude Opus)
 Custom template system
 Batch processing capabilities
 API endpoint for programmatic access
 Integration with cloud storage services
 Multi-language support
 Real-time collaboration features
