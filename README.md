# 🎬 StreamScribe-AI

A powerful YouTube video summarizer powered by local AI models through Ollama. Get comprehensive, detailed summaries of any YouTube video with just a URL.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B.svg)
![Ollama](https://img.shields.io/badge/Ollama-latest-green.svg)

## ✨ Features

- **Instant Video Summaries**: Generate detailed summaries from YouTube video transcripts
- **Model Selection**: Choose from multiple local Ollama models
- **Custom System Prompts**: Customize AI behavior with editable system prompts
- **Markdown Formatting**: Well-formatted, structured summaries with highlights
- **User-Friendly Interface**: Clean Streamlit UI with model recommendations

## 🚀 Quick Start

1. **Clone the repository**

   ```bash
   git clone https://github.com/ahmed2231web/StreamScribe-AI.git
   cd StreamScribe-AI
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Install Ollama and download models**

   Follow instructions at [ollama.ai](https://ollama.ai) to install Ollama, then:

   ```bash
   ollama pull gemma3:4b   # Recommended for summarization
   ollama pull llama3.2:3b  # Alternative model
   ```

4. **Run the application**

   ```bash
   streamlit run main.py
   ```

## 🔧 Usage

1. Enter a YouTube URL in the input field
2. Select your preferred AI model from the dropdown
3. Click "Generate Summary"
4. View the detailed, structured analysis of the video content

## 🧩 Technology Stack

- **Streamlit**: Frontend UI
- **Ollama**: Local AI model integration
- **YouTube Transcript API**: Transcript extraction
- **Pydantic**: Data validation and settings management

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! 
