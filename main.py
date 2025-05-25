import re
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
import ollama
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
import html

class YouTubeVideo(BaseModel):
    video_id: str = Field(..., description="The YouTube video ID")
    url: str = Field(..., description="The full YouTube video URL")
    
    @field_validator('video_id')
    @classmethod
    def validate_video_id(cls, v):
        if not v or not re.match(r'^[A-Za-z0-9_-]{11}$', v):
            raise ValueError("Invalid YouTube video ID format")
        return v

class Summary(BaseModel):
    main_points_discussed: List[str] = Field(..., description="Comprehensive main points and key takeaways from the video")
    conclusion: str = Field(..., description="A detailed conclusion of the video")

class TranscriptSummary(BaseModel):
    video_id: str
    url: str
    summary: Summary
    raw_response: Optional[str] = None

########################################################################
# Extracts the video ID from a YouTube URL using regex patterns
# Handles standard YouTube URLs, embed URLs, and shortened youtu.be URLs
# Raises ValueError if extraction fails
########################################################################
def extract_video_id(url: str) -> str:
    """
    Extract the video ID from a YouTube URL
    
    Args:
        url: The YouTube URL
        
    Returns:
        The video ID
    """
    try:
        patterns = [
            r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',  # Standard YouTube URLs
            r'(?:embed\/)([0-9A-Za-z_-]{11})',  # Embed URLs
            r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})'  # Shortened youtu.be URLs
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        raise ValueError("Could not extract video ID from URL")
    except Exception as e:
        raise ValueError(f"Error extracting video ID: {str(e)}")

########################################################################
# Retrieves the transcript of a YouTube video using YouTubeTranscriptApi
# Combines transcript segments into a single text string
# Raises Exception if transcript fetching fails
########################################################################
def get_transcript(video_id: str) -> str:
    """
    Get the transcript from a YouTube video ID
    
    Args:
        video_id: The YouTube video ID
        
    Returns:
        The combined transcript text
    """
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([item["text"] for item in transcript_list])
        return transcript_text
    except Exception as e:
        raise Exception(f"Error fetching transcript: {str(e)}")

########################################################################
# Uses Ollama API to generate a summary of the video transcript
# Sends system prompt and transcript to the specified AI model
# Returns the AI-generated summary text
########################################################################
def summarize_with_ollama(transcript_text: str, model_name: str, system_prompt: str) -> str:
    """
    Use Ollama to summarize the transcript with the selected model.
    """
    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": f"Please summarize the following YouTube video transcript according to the structure and guidelines provided:\\n\\n{transcript_text}"
                }
            ]
        )
        
        return response['message']['content']
    except Exception as e:
        raise Exception(f"Error summarizing transcript with {model_name}: {str(e)}")

########################################################################
# Parses and structures the raw AI output into a standardized Summary format
# Uses regex to extract main points and conclusion sections
# Implements multiple fallback approaches if parsing fails
########################################################################
def clean_ollama_output(raw_text: str) -> Summary:
    """
    Clean and structure the Ollama output into the new Summary format.
    """
    try:
        cleaned_text = html.unescape(raw_text.strip())
        
        main_points_discussed = []
        conclusion = ""
        
        # More robust regex patterns to find sections
        # Look for "Main Points Discussed" section (with or without ##)
        main_points_pattern = r"(?:##\s*)?(?:Main Points Discussed|MAIN POINTS DISCUSSED|Main Points|MAIN POINTS)[\s:]*\n?(.*?)(?=(?:##\s*)?(?:Conclusion|CONCLUSION|$))"
        main_points_match = re.search(main_points_pattern, cleaned_text, re.IGNORECASE | re.DOTALL)
        
        # Look for "Conclusion" section (with or without ##)
        conclusion_pattern = r"(?:##\s*)?(?:Conclusion|CONCLUSION)[\s:]*\n?(.*)"
        conclusion_match = re.search(conclusion_pattern, cleaned_text, re.IGNORECASE | re.DOTALL)

        if main_points_match:
            points_text = main_points_match.group(1).strip()
            
            # Try multiple approaches to extract bullet points
            # Method 1: Look for lines starting with bullet characters
            bullet_lines = []
            for line in points_text.split('\n'):
                line = line.strip()
                # Check if line starts with bullet characters or numbers
                if line and (line.startswith(('•', '*', '-', '◦', '▪', '‣')) or re.match(r'^\d+\.?\s', line)):
                    # Clean up the bullet point
                    clean_point = re.sub(r'^[\•\*\-◦▪‣\d\.]+\s*', '', line).strip()
                    if clean_point:
                        bullet_lines.append(clean_point)
            
            if bullet_lines:
                main_points_discussed = bullet_lines
            else:
                # Method 2: Try regex approach for bullet points
                bullet_patterns = [
                    r'(?:^|\n)[\s]*[\•\*\-◦▪‣]+\s*([^\n]+)',  # Bullet characters
                    r'(?:^|\n)[\s]*\d+\.?\s*([^\n]+)',        # Numbered points
                    r'(?:^|\n)[\s]*([A-Z][^\n]*[.!?])(?=\s*[\n•\*\-\d]|$)'  # Sentences that look like points
                ]
                
                for pattern in bullet_patterns:
                    matches = re.findall(pattern, points_text, re.MULTILINE)
                    if matches:
                        main_points_discussed = [match.strip() for match in matches if match.strip()]
                        break
                
                # Method 3: If still no points, split by sentences and filter
                if not main_points_discussed:
                    sentences = re.split(r'[.!?]+', points_text)
                    main_points_discussed = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        # If still no main points found, provide fallback
        if not main_points_discussed:
            if main_points_match:
                raw_content = main_points_match.group(1).strip()[:300]
                main_points_discussed = [f"Could not parse main points. Raw content: {raw_content}"]
            else:
                main_points_discussed = ["'Main Points Discussed' section not found in the output."]

        # Process conclusion
        if conclusion_match:
            conclusion_text = conclusion_match.group(1).strip()
            # Remove any remaining markdown formatting
            conclusion = re.sub(r'[#*_`]', '', conclusion_text)
            conclusion = re.sub(r'\s+', ' ', conclusion).strip()
        
        if not conclusion:
            if conclusion_match:
                raw_content = conclusion_match.group(1).strip()[:200]
                conclusion = f"Could not parse conclusion. Raw content: {raw_content}"
            else:
                conclusion = "'Conclusion' section not found in the output."
        
        return Summary(
            main_points_discussed=main_points_discussed if main_points_discussed else ["No main points could be extracted."],
            conclusion=conclusion if conclusion else "No conclusion could be extracted."
        )
        
    except Exception as e:
        # If all parsing fails, try to extract any meaningful content
        lines = raw_text.split('\n')
        fallback_points = []
        fallback_conclusion = ""
        
        for line in lines:
            line = line.strip()
            if line and len(line) > 10:
                if any(keyword in line.lower() for keyword in ['point', 'highlight', 'key', 'important']):
                    fallback_points.append(line)
                elif any(keyword in line.lower() for keyword in ['conclusion', 'summary', 'overall']):
                    fallback_conclusion = line
        
        if not fallback_points:
            fallback_points = ["Failed to parse AI output. Please check the raw response."]
        if not fallback_conclusion:
            fallback_conclusion = "Failed to parse conclusion from AI output."
            
        return Summary(
            main_points_discussed=fallback_points,
            conclusion=fallback_conclusion
        )

########################################################################
# Main application function that creates the Streamlit UI and handles the workflow
# Sets up the page layout, configuration options, and processes user input
# Displays the generated summary with formatting
########################################################################
def main():
    st.set_page_config(
        page_title="YouTube Video Summarizer",
        page_icon="🎬",
        layout="wide"
    )
    
    st.title("🎬 YouTube Video Auto Summarizer")
    st.markdown("Enter a YouTube URL and select your preferred AI model to get a comprehensive summary of the video content.")
    
    # Sidebar for Configuration
    st.sidebar.title("⚙️ Configuration")
    
    # Default system prompt
    default_system_prompt = """
        You are a senior video analyst tasked with creating exhaustive, detailed, nuanced summaries of YouTube videos. Your summaries must be professional-grade analyses that capture every dimension of the video's content. Follow this structure meticulously:

        # COMPREHENSIVE VIDEO ANALYSIS

        ## 🎯 Core Objectives & Context
        • Identify and explain 3-5 primary objectives/purposes of the video
        • Analyze the creator's approach to achieving these objectives
        • Contextualize within the creator's body of work/industry trends

        ## 📚 Detailed Content Breakdown
        ### Key Concepts & Terminology (5-8 items)
        • List and define technical/specialist terms with timestamped examples
        • Explain complex ideas using analogies and real-world applications

        ### 🧩 Content Components Analysis
        1. **Educational Elements** (teaching methods used)
        2. **Entertainment Factors** (humor, storytelling techniques)
        3. **Visual Components** (graphics, demonstrations, screen recordings)
        4. **Data Presentation** (statistics, research citations)
        5. **Argument Structure** (persuasive techniques, evidence hierarchy)

        ## 🕒 Chronological Content Map
        Create a minute-by-minute analysis table (virtual):
        | Time Range | Segment Type | Key Events | Importance Level | Visual Cues |
        |------------|--------------|------------|------------------|-------------|
        | 0:00-2:30  | Introduction | [Details]  | High             | [Observations] |
        | [...]      | [...]        | [...]      | [...]           | [...]       |

        ## 💡 Critical Insights & Implications
        • Identify 7-10 substantive insights with:
          - Direct quotes (timestamped)
          - Contextual interpretation
          - Industry implications
          - Potential controversies
        • Cross-reference with related works/sources

        ## 🧐 Creator Analysis
        • Expertise demonstration methods
        • Bias/perspective indicators
        • Audience engagement tactics
        • Unique value proposition

        ## 🏁 Synthesis & Impact Assessment
        ### Multidimensional Conclusion
        1. **Content Efficacy** (strengths/weaknesses)
        2. **Audience Value** (beginner vs expert utility)
        3. **Long-term Relevance** (projected 1-3 year impact)
        4. **Ethical Considerations** (potential misuse cases)

        ### Actionable Intelligence
        • 3-5 professional applications
        • Recommended follow-up resources
        • Critical questions raised (5-8 items)

        # FORMATTING PROTOCOL
        - Use hierarchical markdown (#### for sub-sections)
        - Combine bullet points/numbered lists/tables
        - Highlight key terms with **bold** and _italics_
        - Maintain academic tone with professional emoji use
        - Include virtual timestamps throughout (e.g., [12:45])
        - Preserve technical jargon with inline explanations

        # QUALITY CONTROL
        - Minimum 5000 word analysis
        - 50-60 substantive points minimum
        - 10+ external reference comparisons
        - Identify 10+ potential improvement areas
        - Flag any factual inconsistencies
        - Rate content density (1-5 scale)

        Your analysis should serve as a professional-grade reference document suitable for academic research, content remixing, and deep critical analysis. Prioritize depth over brevity while maintaining strict factual accuracy.
        """
    
    # Initialize the system prompt in session state if it doesn't exist
    if 'system_prompt' not in st.session_state:
        st.session_state.system_prompt = default_system_prompt
    
    # Model Selection
    try:
        models_response = ollama.list()
        models = models_response.get('models', [])
        
        # Extract model names using regex (we know this works now)
        ollama_models_list = []
        for m in models:
            model_str = str(m)
            # Extract model name from string format: model='name'
            match = re.search(r"[mM]odel='([^']+)'", model_str)
            if match:
                ollama_models_list.append(match.group(1))
        
        if not ollama_models_list:
            raise ValueError("No models found")
            
    except Exception as e:
        st.sidebar.error("⚠️ Could not connect to Ollama. Using default models.")
        ollama_models_list = ["gemma3:4b", "qwen3:1.7b", "llama3.2:3b", "nomic-embed-text:v1.5"]
    
    # Set default model preference (prioritize text generation models)
    text_models = [m for m in ollama_models_list if not m.startswith("nomic-embed")]
    if "gemma3:4b" in text_models:
        default_model = "gemma3:4b"
    elif text_models:
        default_model = text_models[0]
    else:
        default_model = ollama_models_list[0] if ollama_models_list else "gemma3:4b"
    
    default_index = ollama_models_list.index(default_model) if default_model in ollama_models_list else 0
    
    # Create nice formatted options for the selectbox
    formatted_options = []
    model_name_mapping = {}
    
    for model_name in ollama_models_list:
        if ':' in model_name:
            model_family, model_size = model_name.split(':', 1)
            if model_name.startswith("nomic-embed"):
                formatted_name = f"🔗 {model_family} ({model_size}) - Embedding Model"
            else:
                formatted_name = f"🤖 {model_family} ({model_size})"
        else:
            formatted_name = f"🤖 {model_name}"
            
        formatted_options.append(formatted_name)
        model_name_mapping[formatted_name] = model_name
    
    # Model selection dropdown
    formatted_selected = st.sidebar.selectbox(
        "🧠 Choose AI Model:",
        options=formatted_options,
        index=default_index,
        help="Select the AI model for summarization. Text generation models work best for this task."
    )
    
    selected_model = model_name_mapping[formatted_selected]
    
    # Display model info
    if selected_model.startswith("nomic-embed"):
        st.sidebar.warning("⚠️ Note: This is an embedding model. Consider using a text generation model for better summaries.")
    else:
        st.sidebar.success(f"✅ Selected: {selected_model}")
    
    # System Prompt Editor
    with st.sidebar.expander("✏️ System Prompt Editor", expanded=False):
        st.text_area(
            "Edit the system prompt for the AI:",
            value=st.session_state.system_prompt,
            height=400,
            key="prompt_editor",
            help="Customize how the AI generates summaries. Changes will apply to the next summary generation."
        )
        
        # Update the session state when the text area changes
        if st.session_state.prompt_editor != st.session_state.system_prompt:
            st.session_state.system_prompt = st.session_state.prompt_editor
            
        # Reset button
        if st.button("Reset to Default", help="Reset the prompt to the default template"):
            st.session_state.system_prompt = default_system_prompt
            st.session_state.prompt_editor = default_system_prompt
            st.rerun()
    
    # Add help section
    with st.sidebar.expander("ℹ️ Model Guide"):
        st.markdown("""
        **Recommended Models for Summarization:**
        - **gemma3:4b** - Fast and accurate
        - **qwen3:1.7b** - Lightweight and efficient  
        - **llama3.2:3b** - Good balance of speed/quality
        
        **Note:** Embedding models (nomic-embed) are designed for creating vector representations, not text generation.
        """)
    
    with st.sidebar.expander("🔧 Troubleshooting"):
        st.markdown("""
        **If you don't see your models:**
        1. Ensure Ollama is running
        2. Pull models: `ollama pull gemma3:4b`
        3. Refresh this page
        
        **Popular commands:**
        ```bash
        ollama pull gemma3:4b
        ollama pull qwen3:1.7b
        ollama pull llama3.2:3b
        ```
        """)
    
    # Main content area
    st.markdown("---")
    
    # Input form
    with st.form("youtube_url_form", clear_on_submit=False):
        youtube_url = st.text_input(
            "🔗 YouTube URL", 
            placeholder="https://www.youtube.com/watch?v=...",
            help="Paste any YouTube video URL here"
        )
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submit_button = st.form_submit_button("📝 Generate Summary", type="primary", use_container_width=True)
    
    # Processing
    if submit_button and youtube_url:
        try:
            with st.spinner("🔄 Processing your request..."):
                st.markdown("### 🔄 Process Status")
                
                # Video ID Extraction
                with st.status("Extracting Video Information...", expanded=False) as status_vid:
                    video = YouTubeVideo(url=youtube_url, video_id=extract_video_id(youtube_url))
                    st.session_state.current_url = video.url 
                    status_vid.update(label=f"✅ Video ID: {video.video_id}", state="complete")

                # Transcript Fetching
                with st.status("Fetching Video Transcript...", expanded=False) as status_transcript:
                    transcript_text = get_transcript(video.video_id)
                    word_count = len(transcript_text.split())
                    status_transcript.update(label=f"✅ Transcript Ready ({word_count:,} words)", state="complete")
                
                # Show transcript preview
                if transcript_text:
                    with st.expander("📄 Transcript Preview"):
                        st.text_area(
                            "First 1000 characters:", 
                            transcript_text[:1000] + ("..." if len(transcript_text) > 1000 else ""), 
                            height=150,
                            disabled=True
                        )

                # AI Summarization
                with st.status(f"Generating Summary with {selected_model}...", expanded=False) as status_summarize:
                    raw_summary_text = summarize_with_ollama(transcript_text, selected_model, st.session_state.system_prompt)
                    status_summarize.update(label=f"✅ Summary Generated", state="complete")

                # Processing Results
                with st.status("Finalizing Summary...", expanded=False) as status_clean:
                    st.session_state.current_summary = TranscriptSummary(
                        video_id=video.video_id,
                        url=st.session_state.current_url,
                        summary=Summary(main_points_discussed=["Direct display"], conclusion="Direct display"),
                        raw_response=raw_summary_text
                    )
                    status_clean.update(label="✅ Processing Complete!", state="complete")
                
                st.success("🎉 Summary generated successfully!")
            
        except ValueError as ve:
            st.error(f"❌ Input Error: {ve}")
        except Exception as e:
            st.error(f"❌ An unexpected error occurred: {e}")
            with st.expander("🔍 Error Details (for debugging)"):
                st.code(str(e))

    # Display Results
    if 'current_summary' in st.session_state and st.session_state.current_summary:
        summary_output = st.session_state.current_summary
        
        st.markdown("---")
        st.markdown("## 📜 Video Summary")
        
        # Video information
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**🔗 Video:** [{summary_output.url}]({summary_output.url})")
        with col2:
            st.markdown(f"**🆔 ID:** `{summary_output.video_id}`")
        
        st.markdown("---")
        
        # Display the raw response directly (since AI generates well-formatted content)
        raw_response = summary_output.raw_response
        
        # Clean up any <think> tags if present
        if '<think>' in raw_response:
            # Extract content after </think>
            parts = raw_response.split('</think>')
            if len(parts) > 1:
                clean_response = parts[1].strip()
            else:
                clean_response = raw_response
        else:
            clean_response = raw_response
        
        # Display the cleaned response as markdown
        st.markdown(clean_response)

        # Raw output option for debugging
        with st.expander("🔍 View Raw AI Response (Debug)"):
            st.markdown("**Complete raw model output:**")
            st.text_area("Raw AI Response", summary_output.raw_response, height=300, disabled=True, label_visibility="hidden")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>🤖 Powered by Ollama & Streamlit</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
