import streamlit as st
import google.generativeai as genai
import os
import shutil
from dotenv import load_dotenv
import json
from streamlit_option_menu import option_menu
import re
import numpy as np
import requests
import time
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API"))

content_model = genai.GenerativeModel(
    'gemini-2.0-flash',
    system_instruction="You are the content generator, who can deliver the required information based on the transcript given."
)

vision_model = genai.GenerativeModel(
    'gemini-2.0-flash', 
    system_instruction = "You are a very good video analyzer and information extractor from video"
)

transcription_model = genai.GenerativeModel(
    'gemini-2.0-flash',
    system_instruction = "You are the transcription model, who can convert the given video into text format, and also able to tranlate the transcript"
)

def extract_youtube_id(url):
    """Extract YouTube video ID from various YouTube URL formats"""
    try:
        parsed_url = urlparse(url)
        
        # Handle different YouTube URL formats
        if 'youtube.com' in parsed_url.netloc:
            if '/watch' in parsed_url.path:
                return parse_qs(parsed_url.query)['v'][0]
            elif '/embed/' in parsed_url.path:
                return parsed_url.path.split('/embed/')[1].split('?')[0]
        elif 'youtu.be' in parsed_url.netloc:
            return parsed_url.path[1:].split('?')[0]
        
        return None
    except Exception as e:
        st.error(f"Error extracting YouTube ID: {str(e)}")
        return None

def get_youtube_transcript(video_id):
    """Get YouTube transcript using youtube-transcript-api"""
    try:
        with st.spinner("Fetching YouTube transcript..."):
            # Try to get transcript in English first
            try:
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
                language = 'English'
            except:
                # If English not available, get any available transcript
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
                language = 'Auto-detected'
            
            # Combine all transcript segments
            full_transcript = ' '.join([segment['text'] for segment in transcript_list])
            
            return {
                'Original_text': full_transcript,
                'Translated_text': full_transcript,
                'Original_language': language
            }
    
    except Exception as e:
        st.error(f"Failed to get YouTube transcript: {str(e)}")
        st.info("This might happen if the video doesn't have captions or is private/restricted.")
        return None

def get_youtube_video_info(video_id):
    """Get basic YouTube video information"""
    try:
        # You can use YouTube Data API here if you have the API key
        # For now, we'll return basic info
        return {
            'title': f'YouTube Video {video_id}',
            'url': f'https://www.youtube.com/watch?v={video_id}'
        }
    except Exception as e:
        st.error(f"Error getting video info: {str(e)}")
        return None

def get_local_path(filename):
    safe_filename = os.path.basename(filename).replace(" ", "_")
    downloads_folder = os.path.join("downloads")
    os.makedirs(downloads_folder, exist_ok=True)    
    return os.path.join(downloads_folder, safe_filename)  

@st.cache_data
def load_url(drive_url):
    file_id = drive_url.split("/d/")[1].split("/")[0]
    direct_link = f"https://drive.google.com/uc?export=download&id={file_id}"
    file_path = get_local_path(f"video_{file_id}.mp4")

    with st.spinner("Checking Google Drive Link..."):
        try:
            response = requests.get(direct_link)  
            if response.status_code == 200:
                with open(file_path, "wb") as f:
                    f.write(response.content)
                return file_path
            else:
                st.error("Failed to download file from Google Drive")
                return None
        except Exception as e:
            st.error(f"Google Drive download error: {str(e)}")
            return None

def get_from_uploads(video_file):
    file_path = get_local_path(video_file.name)
    with open(file_path, "wb") as f:
        while chunk := video_file.read(1024 * 1024):  
            f.write(chunk)
    return file_path 

def wait_for_file_active(file, max_retries=5, delay=5):
    retries = 0
    while retries < max_retries:
        status = file.state.name
        if status == "ACTIVE":
            return True
        elif status == "FAILED":
            st.error("File upload failed!")
            return False
        time.sleep(delay)
        retries += 1
        file = genai.get_file(file.name)  
    st.error("File activation timed out!")
    return False

def transcribe_video(my_file):
    try:
        response = transcription_model.generate_content([
            """Transcribe this video content. If the video content or transcript is in another language, translate it to English, else don't need to translate. Return STRICT JSON format with: 
            {'Original_text' : string, 
             'Translated_text' : string, 
             'Original_language' : string}""",
            my_file
        ])
        return parse_gemini_response(response.text)

    except Exception as e:
        st.error(f"Transcription failed: {str(e)}")
        return None

def analyze_with_vision(my_file):
    try:
        response = vision_model.generate_content([ 
            """Analyze this video content for any purposes, related to the video (It can be educational, entertainment, etc). Return STRICT JSON format with: 
            {genre : string, 
            mood : string, 
            similar_content_suggestions : array of strings, 
            key_elements : array of strings,
            audience_options : array of strings}""",
            my_file
        ])
        my_file.delete()
        return parse_gemini_response(response.text)

    except Exception as e:
        st.error(f"Vision analysis failed: {str(e)}")
        return None

def analyze_youtube_content(transcript_text):
    """Analyze YouTube video content based on transcript"""
    try:
        response = vision_model.generate_content(
            f"""Analyze this YouTube video transcript content for categorization and recommendations. 
            Return STRICT JSON format with: 
            {{
                "genre": "string", 
                "mood": "string", 
                "similar_content_suggestions": ["array", "of", "strings"], 
                "key_elements": ["array", "of", "strings"],
                "audience_options": ["array", "of", "strings"]
            }}
            
            Transcript: {transcript_text}"""
        )
        return parse_gemini_response(response.text)

    except Exception as e:
        st.error(f"YouTube content analysis failed: {str(e)}")
        return None

def analyze_type(transcript):
    prompt = f"""By the help of the following text, identify the score of whether it is related educational content or not, the score provided should be in the range of 0 to 1, where higher represents education content
    DO NOT encourage any vulgar or UNWANTED content required for entertainment purposes like comedy, singing, dancing and all, 
    Always try to look straight on the point, and the tone of the text too, if it is professional and is knowledge related thing, consider it to be educational
    The output format should be a STRICT JSON FORMAT as provided

    {{"Score": 0.0}}

    Replace 0.0 with your calculated score as a number (not string).

    This is the following text : {transcript}"""

    try:
        response = content_model.generate_content(prompt)
        result = parse_gemini_response(response.text)
        
        # Ensure Score is a float
        if result and 'Score' in result:
            try:
                result['Score'] = float(result['Score'])
            except (ValueError, TypeError):
                result['Score'] = 0.0
        
        return result
    except Exception as e:
        st.error(f"Error in content type analysis: {str(e)}")
        return {"Score": 0.0}

def parse_gemini_response(response_text):
    try:
        try:
            cleaned = re.sub(r'```json|```', '', response_text)
            cleaned = re.sub(r'[\x00-\x1F\x7F-\x9F]', ' ', cleaned)
            json_str = re.search(r'\{[\s\S]*\}', cleaned)
            if json_str:
                return json.loads(json_str.group())
            return None

        except Exception as e:
            cleaned = re.sub(r'```json|```', '', response_text)
            json_str = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if json_str:
                return json.loads(json_str.group())
            return None

    except Exception as e:
        st.error(f"Failed to parse response: {str(e)}")
        return None

# Streamlit UI
st.set_page_config(page_title="Video Insight AI", layout="wide")

# Initialize session state
if 'current_video' not in st.session_state:
    st.session_state.current_video = {'id': None, 'type': None, 'source': None}
if 'analysis' not in st.session_state:
    st.session_state.analysis = None
if 'transcript' not in st.session_state:
    st.session_state.transcript = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = {}
if 'audience' not in st.session_state:
    st.session_state.audience = "General"
if 'youtube_info' not in st.session_state:
    st.session_state.youtube_info = None

st.title("🎬 Video Insight AI")
st.markdown("*Analyze videos from multiple sources: Upload files, Google Drive, or YouTube URLs*")

video_file_path = None
video_id = None
source_type = None

with st.sidebar:
    st.title("📹 Video Sources")
    
    # Enhanced source selection
    option = st.selectbox(
        "Choose video source:", 
        ["Upload a file", "Google Drive URL", "YouTube URL"],
        help="Select how you want to provide the video"
    )
    
    if option == "Upload a file":
        video_file = st.file_uploader(
            "Upload a video file", 
            type=["mp4", "mov", "avi", "mkv", "webm"], 
            help="Upload a video file from your device"
        )
        if video_file:
            try:
                video_file_path = get_from_uploads(video_file)
                video_id = f"upload_{video_file.name}"
                source_type = "upload"
            except Exception as e:
                st.error(f"Error loading file: {e}")

    elif option == "Google Drive URL":
        video_url = st.text_input(
            "Enter Google Drive URL", 
            help="Paste your Google Drive shareable link here",
            placeholder="https://drive.google.com/file/d/..."
        )
        
        if video_url:
            try:
                video_file_path = load_url(video_url)
                video_id = f"gdrive_{video_url.split('/d/')[1].split('/')[0]}"
                source_type = "gdrive"
            except Exception as e:
                st.error(f"Google Drive access error: {e}")

    elif option == "YouTube URL":
        youtube_url = st.text_input(
            "Enter YouTube URL", 
            help="Paste YouTube video URL here",
            placeholder="https://www.youtube.com/watch?v=..."
        )
        
        if youtube_url:
            youtube_id = extract_youtube_id(youtube_url)
            if youtube_id:
                video_id = f"youtube_{youtube_id}"
                source_type = "youtube"
                st.success(f"✅ YouTube Video ID: {youtube_id}")
                
                # Get video info
                st.session_state.youtube_info = get_youtube_video_info(youtube_id)
                if st.session_state.youtube_info:
                    st.info(f"📺 Video: {st.session_state.youtube_info['title']}")
            else:
                st.error("❌ Invalid YouTube URL format")

    # Process button
    if video_id:
        st.markdown("---")
        with st.expander("🔍 Video Processing", expanded=True):
            if st.session_state.current_video['id'] != video_id:
                st.session_state.chat_history[video_id] = []
                st.session_state.current_video = {
                    'id': video_id, 
                    'type': None, 
                    'source': source_type
                }
                st.info(f"📝 New video detected: {source_type.upper()}")
        
        if st.button("🚀 Process Video", type="primary", use_container_width=True):
            try:
                if source_type == "youtube":
                    # Process YouTube video
                    youtube_id = extract_youtube_id(youtube_url)
                    
                    # Get transcript
                    with st.spinner("📜 Fetching YouTube transcript..."):
                        st.session_state.transcript = get_youtube_transcript(youtube_id)
                    
                    if st.session_state.transcript:
                        # Analyze content type
                        with st.spinner("🔬 Analyzing content..."):
                            type_analysis = analyze_type(st.session_state.transcript['Translated_text'])
                            score = 0
                            if type_analysis and 'Score' in type_analysis:
                                try:
                                    score = float(type_analysis['Score'])
                                except (ValueError, TypeError):
                                    score = 0
                            st.session_state.current_video['type'] = (
                                "Knowledge Analytics" if score > 0.5 
                                else "Entertainment Analytics"
                            )
                        
                        # Get content analysis
                        with st.spinner("🎯 Getting content insights..."):
                            st.session_state.analysis = analyze_youtube_content(
                                st.session_state.transcript['Translated_text']
                            )
                        
                        st.success("✅ YouTube video processed successfully!")
                    else:
                        st.error("❌ Failed to get YouTube transcript")
                        st.stop()
                
                else:
                    # Process uploaded/Google Drive video
                    if not video_file_path:
                        st.error("❌ No video file available")
                        st.stop()
                    
                    with st.spinner("📤 Uploading file..."):
                        my_file = genai.upload_file(video_file_path)
                        if my_file.state.name == "FAILED":
                            st.info("🔄 Re-uploading file...")
                            my_file = genai.upload_file(video_file_path)

                        if not wait_for_file_active(my_file):
                            st.stop()

                    with st.spinner("📝 Getting transcript..."):
                        st.session_state.transcript = transcribe_video(my_file)

                    with st.spinner("🔍 Analyzing video..."):
                        # Determine content type
                        if st.session_state.transcript:
                            type_analysis = analyze_type(st.session_state.transcript['Translated_text'])
                            score = 0
                            if type_analysis and 'Score' in type_analysis:
                                try:
                                    score = float(type_analysis['Score'])
                                except (ValueError, TypeError):
                                    score = 0
                            st.session_state.current_video['type'] = (
                                "Knowledge Analytics" if score > 0.5 
                                else "Entertainment Analytics"
                            )
                        
                        # Get visual analysis
                        st.session_state.analysis = analyze_with_vision(my_file)

                    st.success("✅ Video processed successfully!")
                    
            except Exception as e:
                st.error(f"❌ Processing failed: {str(e)}")

# Main content area
if st.session_state.analysis and st.session_state.transcript:
    
    # Audience selection
    if st.session_state.analysis and 'audience_options' in st.session_state.analysis:
        st.session_state.audience = st.selectbox(
            "🎯 Select Target Audience:",
            options=st.session_state.analysis.get('audience_options', ['General']),
            key='audience_selector'
        )

    # Content type indicator
    if st.session_state.current_video['type'] == "Knowledge Analytics":
        st.success("🎓 This video contains educational content")
        menu_options = ["📊 Summary", "🗺️ Roadmap", "📝 Transcript", "💬 Chat"]
        menu_icons = ["bar-chart", "map", "card-text", "chat"]
    else:
        st.info("🎭 This video contains entertainment content")
        menu_options = ["📊 Summary", "🎬 Similar Content", "📝 Transcript", "💬 Chat"]
        menu_icons = ["bar-chart", "film", "card-text", "chat"]

    # Navigation menu
    selected = option_menu(
        menu_title=None,
        options=menu_options,
        icons=menu_icons,
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "#31333F", "font-size": "18px"}, 
            "nav-link": {"font-size": "16px", "text-align": "center", "margin":"0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#02ab21"},
        }
    )

    # Content sections
    if selected == "📊 Summary":
        st.header("📊 Video Summary")
        
        if st.session_state.current_video['source'] == 'youtube' and st.session_state.youtube_info:
            st.info(f"🎬 **YouTube Video**: {st.session_state.youtube_info['title']}")
        
        if st.button("✨ Generate Summary", type="primary"):
            with st.spinner("🔄 Generating comprehensive summary..."):
                if st.session_state.current_video['type'] == "Knowledge Analytics":
                    prompt = f"Provide a comprehensive and detailed educational summary for {st.session_state.audience} audience"
                    content = st.session_state.transcript['Translated_text']
                else:
                    prompt = "Provide a comprehensive and detailed entertainment analysis summary"
                    content = json.dumps(st.session_state.analysis)
                
                response = content_model.generate_content(f"{prompt} for the provided content:\n\n{content}")
                
                st.markdown("### 📋 Summary")
                st.markdown(response.text)
    
    elif selected == "🗺️ Roadmap" and st.session_state.current_video['type'] == "Knowledge Analytics":
        st.header("🗺️ Learning Roadmap")
        
        if st.button("🎯 Generate Learning Path", type="primary"):
            with st.spinner("🔄 Creating personalized roadmap..."):
                response = content_model.generate_content(
                    f"Create a detailed learning roadmap for {st.session_state.audience} audience based on this content. "
                    f"Include prerequisites, main topics, practice exercises, and next steps.\n\n"
                    f"Content: {st.session_state.transcript['Translated_text']}"
                )
                
                st.markdown("### 🎯 Your Learning Path")
                st.markdown(response.text)
    
    elif selected == "🎬 Similar Content" and st.session_state.current_video['type'] == "Entertainment Analytics":
        st.header("🎬 Similar Content Recommendations")
        
        if st.session_state.analysis:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🎭 Content Details")
                st.markdown(f"**Genre:** {st.session_state.analysis.get('genre', 'Not specified')}")
                st.markdown(f"**Mood:** {st.session_state.analysis.get('mood', 'Not specified')}")
                
                st.markdown("### 🔑 Key Elements")
                for element in st.session_state.analysis.get('key_elements', []):
                    st.markdown(f"• {element}")
            
            with col2:
                st.markdown("### 🎯 Recommended Similar Content")
                for item in st.session_state.analysis.get('similar_content_suggestions', []):
                    st.markdown(f"🎬 {item}")
    
    elif selected == "📝 Transcript":
        st.header("📝 Video Transcript")
        
        if st.session_state.current_video['source'] == 'youtube':
            st.info("📺 Transcript obtained from YouTube captions")
        
        if st.session_state.transcript:
            original_lang = st.session_state.transcript.get('Original_language', '').lower()
            
            if original_lang not in ['en', 'english'] and st.session_state.transcript.get('Original_text') != st.session_state.transcript.get('Translated_text'):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### 🌍 English Translation")
                    st.text_area("", st.session_state.transcript['Translated_text'], height=400, key="english_transcript")
                with col2:
                    st.markdown(f"### 🗣️ Original ({st.session_state.transcript['Original_language']})")
                    st.text_area("", st.session_state.transcript['Original_text'], height=400, key="original_transcript")
            else:
                st.markdown(f"### 📄 Transcript ({st.session_state.transcript.get('Original_language', 'English')})")
                st.text_area("", st.session_state.transcript.get('Translated_text', st.session_state.transcript.get('Original_text', '')), height=400, key="single_transcript")
    
    elif selected == "💬 Chat":
        st.header("💬 AI Video Assistant")
        
        if st.session_state.current_video['source'] == 'youtube':
            st.info("🤖 Ask me anything about this YouTube video!")
        
        current_chat = st.session_state.chat_history.get(video_id, [])
        
        # Display chat history
        for message in current_chat:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("💭 Ask about the video content..."):
            current_chat.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    if st.session_state.current_video['type'] == "Knowledge Analytics":
                        context = st.session_state.transcript['Translated_text']
                        system_prompt = "educational content"
                    else:
                        context = json.dumps(st.session_state.analysis)
                        system_prompt = "entertainment content"
                    
                    response = content_model.generate_content(
                        f"Answer the question based on the {system_prompt} with respect to the {st.session_state.audience} audience.\n\n"
                        f"Question: {prompt}\n\n"
                        f"Context: {context}"
                    )
                    
                    current_chat.append({"role": "assistant", "content": response.text})
                    st.session_state.chat_history[video_id] = current_chat
                    
                    st.markdown(response.text)

else:
    # Welcome screen
    st.markdown("""
    <div style="text-align: center; padding: 2rem;">
        <h2>🎬 Welcome to Video Insight AI</h2>
        <p>Upload a video file, provide a Google Drive link, or enter a YouTube URL to get started!</p>
        <p>📹 <strong>Supported Sources:</strong></p>
        <ul style="list-style: none; padding: 0;">
            <li>📁 File Upload (MP4, MOV, AVI, MKV, WebM)</li>
            <li>☁️ Google Drive Links</li>
            <li>🎭 YouTube URLs (with captions)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature highlights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🎓 Educational Content
        - Detailed summaries
        - Learning roadmaps
        - Q&A assistance
        """)
    
    with col2:
        st.markdown("""
        ### 🎭 Entertainment Content
        - Content analysis
        - Similar recommendations
        - Mood & genre detection
        """)
    
    with col3:
        st.markdown("""
        ### 🌍 Multi-language Support
        - Auto-translation
        - Original & translated text
        - Language detection
        """)
