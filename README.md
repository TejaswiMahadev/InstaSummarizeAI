# 🎬 Video Insight AI

**Video Insight AI** is a powerful Streamlit-based web application that allows users to analyze and summarize video content from **YouTube**, **Google Drive**, or **local uploads**. It uses **Google Gemini (Generative AI)** to perform transcription, translation, video analysis, and interactive Q&A.

---

## ✨ Features

### 🔍 Video Input Options
- 📁 Upload a video file (`.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`)
- ☁️ Google Drive link (auto download supported)
- 🎥 YouTube URL (automatic transcript retrieval)

### 🧠 AI Capabilities
- 📝 Transcription & Translation (multilingual support)
- 📊 Content Classification (Educational vs Entertainment)
- 🎯 Audience-Based Summaries
- 🗺️ Learning Roadmap for educational content
- 🎬 Genre, Mood & Similar Video Suggestions
- 💬 AI-Powered Chat about the video content

---

## 🚀 Technologies Used

| Technology            | Purpose                                  |
|------------------------|------------------------------------------|
| `Streamlit`            | Web-based UI framework                   |
| `google-generativeai`  | Gemini API integration                   |
| `youtube-transcript-api` | Fetching YouTube transcripts           |
| `dotenv`               | Manage environment variables             |
| `requests`             | Download videos from Google Drive        |
| `re`, `json`, `os`, etc.| Utility operations                      |

---

## 🛠️ Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/TejaswiMahadev/VidAI.git
cd VidAI
```
### 2. Install all the requirements

```bash
pip install -r requirements.txt
```

### 3.  Setup API Key

```bash
GEMINI_API=your_google_gemini_api_key
```

### 4. Run the application

```bash
streamlit run main.py
```


