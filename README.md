# Enhanced Document Generator

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A sophisticated multi-agent AI application that transforms raw text into professionally formatted LaTeX documents with automatic PDF generation. Built with Streamlit and powered by OpenAI GPT-4 and Anthropic Claude models.

## 🌟 Features

- **Multi-Agent Workflow**: Utilizes specialized AI agents for planning, content generation, validation, design enhancement, and LaTeX formatting
- **Flexible Detail Levels**: Choose from TL;DR (5 sections), Executive Summary (10 sections), or Detailed Summary (15 sections)
- **Professional Output**: Generates publication-ready LaTeX documents with proper formatting, tables, figures, and citations
- **PDF Generation**: Automatic compilation to PDF with multiple fallback options
- **Web Interface**: User-friendly Streamlit interface with real-time progress tracking
- **Document Management**: Automatic file saving with session history and document recovery
- **Research Integration**: Optional web research capabilities using DuckDuckGo and Newspaper4k
- **Robust Error Handling**: Comprehensive fallback mechanisms and retry logic
- **Diagnostics**: Built-in system diagnostics and troubleshooting tools

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Anthropic API key
- (Optional) LaTeX distribution for local PDF compilation

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/enhanced-document-generator.git
   cd enhanced-document-generator
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   CLAUDE_MODEL_ID=claude-sonnet-4-20250514
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   Navigate to `http://localhost:8501`

## 📦 Dependencies

### Core Requirements

```txt
streamlit>=1.28.0
agno>=0.1.0
openai>=1.0.0
anthropic>=0.25.0
python-dotenv>=1.0.0
```

### Optional Dependencies

```txt
pdflatex>=0.1.3          # For enhanced PDF generation
duckduckgo-search>=3.0.0 # For web research capabilities
newspaper4k>=0.9.0       # For article parsing
```

### System Dependencies (Optional)

For local PDF compilation, install a LaTeX distribution:

- **Windows**: [MiKTeX](https://miktex.org/) or [TeX Live](https://www.tug.org/texlive/)
- **macOS**: [MacTeX](https://www.tug.org/mactex/)
- **Linux**: `sudo apt-get install texlive-full` (Ubuntu/Debian)

## 🏗️ Architecture

The application follows a multi-agent architecture with specialized AI agents:

```
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
```

### Workflow Stages

1. **Planning Stage**: Analyzes input text and generates document outline
2. **Validation Stage**: Reviews and refines document structure
3. **Content Generation**: Creates detailed content for each section
4. **Design Enhancement**: Improves formatting and presentation
5. **LaTeX Conversion**: Transforms content to professional LaTeX format
6. **PDF Generation**: Compiles LaTeX to downloadable PDF

## 🎯 Usage

### Basic Usage

1. **Enter Document Topic**: Provide a clear topic for your document
2. **Input Text**: Paste your raw text (up to 20,000 characters)
3. **Select Detail Level**: Choose the appropriate complexity level
4. **Generate Document**: Click the generate button and wait for processing

### Detail Levels

| Level | Sections | Best For |
|-------|----------|----------|
| TL;DR | 5 sections | Quick summaries, brief reports |
| Executive Summary | 10 sections | Business reports, presentations |
| Detailed Summary | 15 sections | Academic papers, comprehensive analyses |

### Output Formats

- **Markdown**: Human-readable format for editing
- **LaTeX**: Professional typesetting format
- **PDF**: Publication-ready document

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `ANTHROPIC_API_KEY` | Anthropic API key | Required |
| `CLAUDE_MODEL_ID` | Claude model identifier | `claude-sonnet-4-20250514` |

### Advanced Configuration

Modify these constants in `app.py` for fine-tuning:

```python
API_TIMEOUT = 120      # API request timeout (seconds)
MAX_RETRIES = 3        # Maximum retry attempts
RETRY_DELAY = 5        # Delay between retries (seconds)
```

## 🛠️ Troubleshooting

### Common Issues

**PDF Generation Fails**
- Install the `pdflatex` Python library: `pip install pdflatex`
- Install a LaTeX distribution (see System Dependencies)
- Use the online LaTeX editor fallback (Overleaf)

**API Timeout Errors**
- Check your internet connection
- Verify API keys are correct and have sufficient credits
- Try reducing the detail level for large documents

**Memory Issues with Large Documents**
- Use TL;DR mode for very large inputs
- Break large documents into smaller sections
- Increase system memory allocation

### Diagnostics

The application includes built-in diagnostics accessible via the sidebar:
- API key validation
- LaTeX installation detection
- Library dependency checks
- Output directory verification

## 🗂️ File Structure

```
enhanced-document-generator/
├── app.py                 # Main application file
├── .env                   # Environment variables (create this)
├── requirements.txt       # Python dependencies
├── generated_documents/   # Output directory (auto-created)
├── README.md             # This file
└── .gitignore           # Git ignore file
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/enhanced-document-generator.git

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If available

# Run tests
python -m pytest tests/  # If tests are available

# Run the application
streamlit run app.py
```

## 📈 Performance

### Typical Processing Times

| Detail Level | Small Text (1-2 pages) | Medium Text (3-5 pages) | Large Text (5+ pages) |
|--------------|-------------------------|-------------------------|-----------------------|
| TL;DR | 30-60 seconds | 60-90 seconds | 90-120 seconds |
| Executive | 60-120 seconds | 120-180 seconds | 180-300 seconds |
| Detailed | 120-240 seconds | 240-360 seconds | 360-600 seconds |

*Times may vary based on API response times and system performance*

## 🔒 Security & Privacy

- **API Keys**: Store securely in `.env` file, never commit to repository
- **Data Processing**: Text is processed by OpenAI and Anthropic APIs
- **Local Storage**: Generated documents are saved locally only
- **No Data Retention**: The application doesn't store your data permanently

## 📚 API Documentation

### Key Classes

**AgentManager**: Manages AI agent lifecycle and coordination
```python
manager = AgentManager(topic="My Topic", user_input="...", detail_level="Executive Summary")
```

**Document Generation**: Main workflow function
```python
result = generate_document(user_input, topic, detail_level)
```

### Utility Functions

- `compile_latex_to_pdf()`: Converts LaTeX to PDF
- `save_file_to_disk()`: Handles file I/O operations
- `extract_json()`: Parses JSON from AI responses

## 🎨 Customization

### LaTeX Template

Modify the `DEFAULT_LATEX_TEMPLATE` in `app.py` to customize document appearance:
- Change fonts, colors, and styling
- Add custom packages
- Modify page layout and margins

### AI Instructions

Customize agent instructions to change output style:
- Adjust tone and formality
- Modify section structure
- Change content focus areas

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) - Web application framework
- [OpenAI](https://openai.com/) - GPT-4 language model
- [Anthropic](https://www.anthropic.com/) - Claude language model
- [Agno](https://github.com/agno-ai/agno) - Agent framework
- [LaTeX](https://www.latex-project.org/) - Document preparation system

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/enhanced-document-generator/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/enhanced-document-generator/discussions)
- **Email**: your.email@example.com

## 🗺️ Roadmap

- [ ] Support for additional AI models (GPT-4 Turbo, Claude Opus)
- [ ] Custom template system
- [ ] Batch processing capabilities
- [ ] API endpoint for programmatic access
- [ ] Integration with cloud storage services
- [ ] Multi-language support
- [ ] Real-time collaboration features

---

Made with ❤️ by [Your Name](https://github.com/yourusername)
