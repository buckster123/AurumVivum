import streamlit as st
import requests
import time
import os
import logging
import sqlite3
import json
import base64
import re
from pathlib import Path
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
from dotenv import load_dotenv

# Load environment variables from project root
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

# ==================== CONFIGURATION & LOGGING ====================

# Paths
AUDIO_FOLDER = os.getenv("SUNO_AUDIO_FOLDER", "sandbox/music")
DB_FILE = os.getenv("SUNO_DB_FILE", "suno_history.db")
LOG_FOLDER = os.getenv("SUNO_LOG_FOLDER", "logs")
Path(AUDIO_FOLDER).mkdir(exist_ok=True)
Path(LOG_FOLDER).mkdir(exist_ok=True)

# Logging
log_file = Path(LOG_FOLDER) / f"suno_radio_{datetime.now().strftime('%Y%m%d')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Model Character Limits
CHAR_LIMITS = {
    "V3_5": {"prompt": 3000, "style": 200, "title": 80},
    "V4": {"prompt": 3000, "style": 200, "title": 80},
    "V4_5": {"prompt": 5000, "style": 1000, "title": 100},
    "V4_5PLUS": {"prompt": 5000, "style": 1000, "title": 100},
    "V4_5ALL": {"prompt": 5000, "style": 1000, "title": 80},
    "V5": {"prompt": 5000, "style": 1000, "title": 100}
}

# Page Config
st.set_page_config(
    page_title="🎵 Aurum AI Radio",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== SESSION STATE ====================

def init_session_state():
    defaults = {
        'api_key': os.getenv("SUNO_API_KEY", ""),
        'api_provider': "sunoapi.org",
        'connection_tested': False,
        'credits': 0,
        'current_task': None,
        'queue': [],
        'auto_play': False,
        'auto_save': True,
        'stats': {'total_gens': 0, 'credits_used': 0, 'avg_time': 0},
        'history_loaded': False,
        'debug_mode': False,
        'history_page': 0,
        'history_per_page': 20,
        'generated_tracks': [],  # NEW: Store generated tracks for UI persistence
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ==================== UTILITY FUNCTIONS ====================

def sanitize_filename(text: str) -> str:
    """Sanitize text for safe filename use"""
    if not text:
        return "untitled"
    sanitized = re.sub(r'[^a-zA-Z0-9\s\-_]', '', text)
    return sanitized[:80].strip()

def get_char_limits(model: str) -> dict:
    """Get character limits for selected model"""
    return CHAR_LIMITS.get(model, CHAR_LIMITS["V5"])

def render_audio_player(filepath: str, title: str = ""):
    """Fixed: Efficient audio player using direct file streaming"""
    try:
        if not Path(filepath).exists():
            st.error(f"Audio file not found: {filepath}")
            return
        
        st.audio(str(filepath), format='audio/mpeg')
        if title:
            st.caption(title)
            
    except Exception as e:
        logger.error(f"Failed to render audio player: {str(e)}")
        st.error(f"Could not load audio: {str(e)}")

def render_image_grid(image_paths: list):
    """Render image grid"""
    try:
        if len(image_paths) == 1:
            st.image(image_paths[0], use_container_width=True)
        elif len(image_paths) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                st.image(image_paths[0], use_container_width=True)
            with col2:
                st.image(image_paths[1], use_container_width=True)
    except Exception as e:
        logger.error(f"Failed to render images: {str(e)}")

def show_character_counter(text: str, max_chars: int, label: str):
    """Show character counter with color coding"""
    count = len(text) if text else 0
    color = "red" if count > max_chars else "orange" if count > max_chars * 0.9 else "green"
    st.caption(f":{color}[{label}: {count}/{max_chars}]")

def safe_duration_display(duration_val) -> str:
    """
    FIX: Safely convert duration to display string
    Handles corrupted data (URLs, None, strings) gracefully
    """
    try:
        if duration_val is None:
            return "0.0s"
        
        if isinstance(duration_val, (int, float)):
            return f"{float(duration_val):.1f}s"
        
        if isinstance(duration_val, str):
            if 'http' in duration_val:
                logger.warning(f"Duration field contains URL: {duration_val[:50]}...")
                return "0.0s"
            
            try:
                return f"{float(duration_val):.1f}s"
            except ValueError:
                num_match = re.search(r'(\d+\.?\d*)', duration_val)
                if num_match:
                    return f"{float(num_match.group(1)):.1f}s"
                return "0.0s"
        
        return "0.0s"
    except (ValueError, TypeError) as e:
        logger.error(f"Duration conversion error for value {duration_val}: {e}")
        return "0.0s"

def safe_json_loads(json_str, default=None):
    """
    FIX: Safely parse JSON string with fallback for empty/invalid values
    Prevents JSONDecodeError on empty strings or invalid JSON
    """
    if default is None:
        default = []
    
    if not json_str or not isinstance(json_str, str):
        return default
    
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        logger.warning(f"Failed to parse JSON: {str(json_str)[:50]}...")
        return default

# ==================== DATABASE FUNCTIONS ====================

def get_db():
    """Initialize SQLite DB with migration support"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS generations (
            id TEXT PRIMARY KEY,
            clip_id TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            prompt TEXT,
            title TEXT,
            model TEXT,
            is_instrumental BOOLEAN,
            duration REAL,
            audio_url TEXT,
            audio_file TEXT,
            cover_images TEXT,
            status TEXT
        )
    """)
    
    try:
        cursor.execute("ALTER TABLE generations ADD COLUMN clip_id TEXT")
        conn.commit()
        logger.info("Database migrated: added clip_id column")
    except sqlite3.OperationalError:
        pass
    
    conn.commit()
    return conn

def save_generation(data: dict):
    """Save generation to database"""
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO generations 
            (id, clip_id, prompt, title, model, is_instrumental, duration, audio_url, audio_file, cover_images, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data['id'],
            data.get('clip_id'),
            data['prompt'],
            data['title'],
            data['model'],
            data['is_instrumental'],
            data['duration'],
            data['audio_url'],
            data['audio_file'],
            json.dumps(data.get('cover_images', [])),
            data['status']
        ))
        conn.commit()
        conn.close()
        logger.info(f"Saved generation {data['id']} to database")
    except Exception as e:
        logger.error(f"Failed to save generation: {str(e)}")

def load_history(limit: int = 100, offset: int = 0, search: str = None):
    """Load generation history with pagination and search"""
    try:
        conn = get_db()
        cursor = conn.cursor()
        
        query = "SELECT * FROM generations WHERE 1=1"
        params = []
        
        if search:
            query += " AND (prompt LIKE ? OR title LIKE ?)"
            params.extend([f"%{search}%", f"%{search}%"])
        
        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        columns = ['id', 'clip_id', 'timestamp', 'prompt', 'title', 'model', 'is_instrumental', 
                  'duration', 'audio_url', 'audio_file', 'cover_images', 'status']
        return [dict(zip(columns, row)) for row in rows]
    except Exception as e:
        logger.error(f"Failed to load history: {str(e)}")
        return []

def get_stats():
    """Get generation statistics"""
    try:
        conn = get_db()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*), SUM(duration), AVG(duration) FROM generations WHERE status = 'SUCCESS'")
        total_gens, total_duration, avg_duration = cursor.fetchone()
        
        cursor.execute("SELECT model, COUNT(*) FROM generations GROUP BY model")
        model_stats = dict(cursor.fetchall())
        
        cursor.execute("""
            SELECT DATE(timestamp), COUNT(*) 
            FROM generations 
            WHERE timestamp > datetime('now', '-30 days')
            GROUP BY DATE(timestamp)
        """)
        daily_stats = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            'total_gens': total_gens or 0,
            'total_duration': total_duration or 0,
            'avg_duration': avg_duration or 0,
            'model_stats': model_stats,
            'daily_stats': daily_stats
        }
    except Exception as e:
        logger.error(f"Failed to get stats: {str(e)}")
        return {'total_gens': 0, 'total_duration': 0, 'avg_duration': 0, 
                'model_stats': {}, 'daily_stats': {}}

# ==================== API CLIENT ====================

API_PROVIDER_URLS = {
    "sunoapi.org": "https://api.sunoapi.org/api/v1",
    "kie.ai": "https://api.kie.ai/api/v1",
}

def detect_provider(api_key: str) -> str:
    """Detect provider from API key format"""
    return "kie.ai" if api_key.startswith('sk-') and len(api_key) < 50 else "sunoapi.org"

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def test_connection(api_key: str, provider: str) -> tuple:
    """Test API connection with retry logic"""
    base_url = API_PROVIDER_URLS[provider]
    headers = {"Authorization": f"Bearer {api_key}"}
    
    endpoint = f"{base_url}/{'generate/credit' if provider == 'sunoapi.org' else 'credit'}"
    response = requests.get(endpoint, headers=headers, timeout=10)
    
    if response.status_code == 200:
        result = response.json()
        return True, result.get('data', 0)
    return False, 0

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_music(api_key: str, prompt: str, model: str, is_instrumental: bool, **kwargs) -> str:
    """Generate music with detailed error logging and retry logic"""
    provider = st.session_state.api_provider
    base_url = API_PROVIDER_URLS[provider]
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "instrumental": is_instrumental,
        "customMode": True,
        "callBackUrl": "https://example.com/callback",
        "prompt": prompt
    }
    
    if kwargs.get('title'):
        payload['title'] = kwargs['title']
    if kwargs.get('style'):
        payload['style'] = kwargs['style']
    if kwargs.get('negativeTags'):
        payload['negativeTags'] = kwargs['negativeTags']
    if kwargs.get('styleWeight'):
        payload['styleWeight'] = kwargs['styleWeight']
    if kwargs.get('weirdnessConstraint'):
        payload['weirdnessConstraint'] = kwargs['weirdnessConstraint']
    if kwargs.get('vocalGender'):
        payload['vocalGender'] = kwargs['vocalGender']
    
    logger.info("="*60)
    logger.info("GENERATION REQUEST")
    logger.info(f"Provider: {provider}")
    logger.info(f"URL: {base_url}/generate")
    logger.info(f"Payload: {json.dumps(payload, indent=2)}")
    logger.info("="*60)
    
    try:
        response = requests.post(
            f"{base_url}/generate",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        logger.info(f"Response Status Code: {response.status_code}")
        logger.info(f"Response Text: {response.text[:500]}...")
        
        if response.status_code != 200:
            raise Exception(f"HTTP Error {response.status_code}: {response.text[:300]}")
        
        result = response.json()
        logger.info(f"Parsed JSON: {json.dumps(result, indent=2)}")
        
        if result.get("code") != 200:
            error_msg = result.get('msg', 'No error message provided')
            error_data = result.get('data', {})
            raise Exception(f"API Error Code {result.get('code')}: {error_msg} | Data: {error_data}")
        
        task_id = result.get("data", {}).get("taskId")
        if not task_id:
            raise Exception(f"No taskId in response. Full response: {json.dumps(result)}")
        
        logger.info(f"Task ID received successfully: {task_id}")
        return task_id
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON response from API: {str(e)}")
        raise Exception(f"API returned invalid JSON: {response.text[:200]}")
    except requests.exceptions.Timeout:
        logger.error("Request timed out after 30 seconds")
        raise Exception("API request timed out - server took too long to respond")
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error: {str(e)}")
        raise Exception(f"Failed to connect to API: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in generate_music: {str(e)}")
        raise

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def check_status(api_key: str, task_id: str) -> dict:
    """Check generation status with retry logic"""
    provider = st.session_state.api_provider
    base_url = API_PROVIDER_URLS[provider]
    
    headers = {"Authorization": f"Bearer {api_key}"}
    
    logger.info("="*60)
    logger.info(f"STATUS CHECK for task: {task_id}")
    logger.info(f"URL: {base_url}/generate/record-info")
    logger.info("="*60)
    
    try:
        response = requests.get(
            f"{base_url}/generate/record-info",
            headers=headers,
            params={"taskId": task_id},
            timeout=10
        )
        
        logger.info(f"Status Response Code: {response.status_code}")
        logger.info(f"Status Response Text: {response.text[:500]}...")
        
        if response.status_code != 200:
            logger.warning(f"Status check returned HTTP {response.status_code}")
            return {"status": "ERROR", "error": f"HTTP {response.status_code}: {response.text[:200]}"}
        
        result = response.json()
        logger.info(f"Status JSON: {json.dumps(result, indent=2)}")
        
        if result.get("code") != 200:
            error_msg = result.get('msg', 'Unknown error in status check')
            logger.warning(f"Status API error: {error_msg}")
            return {"status": "ERROR", "error": error_msg}
        
        data = result.get("data", {})
        logger.info(f"Task status: {data.get('status')} | Full data: {json.dumps(data)[:300]}...")
        
        return data
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in status response: {str(e)}")
        return {"status": "ERROR", "error": f"Invalid JSON: {response.text[:100]}"}
    except Exception as e:
        logger.error(f"Exception in check_status: {str(e)}")
        return {"status": "ERROR", "error": str(e)}

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_cover(api_key: str, clip_id: str) -> str:
    """Generate cover art using clip_id (not task_id)"""
    provider = st.session_state.api_provider
    base_url = API_PROVIDER_URLS[provider]
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "taskId": clip_id,
        "callBackUrl": "https://example.com/callback"
    }
    
    logger.info(f"Generating cover for clip {clip_id}")
    
    try:
        response = requests.post(
            f"{base_url}/suno/cover/generate",
            headers=headers,
            json=payload,
            timeout=20
        )
        
        if response.status_code != 200:
            logger.warning(f"Cover generation HTTP {response.status_code}: {response.text[:200]}")
            return None
        
        result = response.json()
        logger.info(f"Cover generation response: {json.dumps(result)}")
        
        if result.get("code") == 409:  # Already exists
            return result["data"]["taskId"]
        elif result.get("code") == 200:
            return result["data"]["taskId"]
        
        return None
    except Exception as e:
        logger.error(f"Cover generation error: {str(e)}")
        return None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def check_cover_status(api_key: str, cover_task_id: str) -> dict:
    """Check cover generation status"""
    provider = st.session_state.api_provider
    base_url = API_PROVIDER_URLS[provider]
    
    headers = {"Authorization": f"Bearer {api_key}"}
    
    try:
        response = requests.get(
            f"{base_url}/suno/cover/details",
            headers=headers,
            params={"taskId": cover_task_id},
            timeout=10
        )
        
        if response.status_code != 200:
            return {"status": "ERROR", "error": f"HTTP {response.status_code}"}
        
        result = response.json()
        return result.get("data", {})
    except Exception as e:
        logger.error(f"Cover status check error: {str(e)}")
        return {"status": "ERROR", "error": str(e)}

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def download_file(url: str, filepath: str) -> str:
    """Download file with retry logic and error handling"""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        file_size = Path(filepath).stat().st_size
        logger.info(f"Downloaded: {filepath} ({file_size} bytes)")
        return str(filepath)
    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        raise

# ==================== MAIN APP ====================

def main():
    init_session_state()
    
    st.title("🎵 Aurum AI Music")
    st.markdown("*Your infinite AI music generator*")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🔑 API Setup")
        
        api_key = st.text_input("API Key", type="password", value=st.session_state.api_key)
        if api_key != st.session_state.api_key:
            st.session_state.api_key = api_key
            st.session_state.connection_tested = False
        
        if api_key:
            if st.button("🔍 Test Connection", type="secondary"):
                with st.spinner("Testing..."):
                    provider = detect_provider(api_key)
                    st.session_state.api_provider = provider
                    success, credits = test_connection(api_key, provider)
                    if success:
                        st.session_state.connection_tested = True
                        st.session_state.credits = credits
                        st.success(f"✅ Connected! Credits: {credits}")
                    else:
                        st.error("❌ Connection failed")
        
        if st.session_state.connection_tested:
            st.info(f"💰 Credits: {st.session_state.credits}")
        
        st.markdown("### ⚙️ Options")
        st.session_state.auto_play = st.checkbox("🔄 Auto-Play Mode")
        st.session_state.auto_save = st.checkbox("💾 Auto-save Files", value=True)
        st.session_state.debug_mode = st.checkbox("🔧 Debug Mode")
        
        if st.button("🗑️ Clear API Key"):
            st.session_state.api_key = ""
            st.session_state.connection_tested = False
            st.rerun()
        
        if st.session_state.debug_mode:
            st.markdown("### 🐛 Debug Info")
            st.text(f"Provider: {st.session_state.api_provider}")
            st.text(f"Connection: {st.session_state.connection_tested}")
            st.text(f"Log: {log_file}")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["🎼 Generate", "📜 History", "📊 Stats"])
    
    # ==================== GENERATE TAB ====================
    with tab1:
        if not (st.session_state.api_key and st.session_state.connection_tested):
            st.info("🔑 Connect your API key in the sidebar to start")
            return
        
        # Model selection
        model = st.selectbox("Model", list(CHAR_LIMITS.keys()), index=list(CHAR_LIMITS.keys()).index("V5"))
        limits = get_char_limits(model)
        
        # NEW: Manual clear button for UI
        if st.button("🗑️ Clear Recent Generation"):
            st.session_state.generated_tracks = []
            st.rerun()
        
        # Inputs
        col1, col2 = st.columns([2, 1])
        
        with col1:
            prompt = st.text_area(
                "📝 Music Prompt",
                placeholder="Describe your song...",
                height=150,
                max_chars=limits["prompt"]
            )
            show_character_counter(prompt, limits["prompt"], "Prompt")
        
        with col2:
            is_instrumental = st.checkbox("🎵 Instrumental", value=True)
            title = st.text_input("🎵 Title (opt)", max_chars=limits["title"])
            show_character_counter(title, limits["title"], "Title")
        
        # Advanced options
        with st.expander("🎚️ Advanced Options"):
            col3, col4 = st.columns(2)
            
            with col3:
                style = st.text_input("Style Tags")
                show_character_counter(style, limits["style"], "Style")
                negative_tags = st.text_input("Exclude Tags")
            
            with col4:
                style_weight = st.slider("Style Weight", 0.0, 1.0, 0.65, 0.05)
                weirdness = st.slider("Weirdness", 0.0, 1.0, 0.5, 0.05)
                vocal_gender = st.selectbox("Vocal Gender", ["auto", "m", "f"]) if not is_instrumental else "auto"
        
        # Generate button
        if st.button("🚀 Generate Music", type="primary", use_container_width=True):
            if not prompt.strip():
                st.error("❌ Prompt is required")
                return
            
            try:
                # Submit generation
                with st.spinner("🚀 Submitting request..."):
                    logger.info("User initiated generation")
                    task_id = generate_music(
                        st.session_state.api_key,
                        prompt,
                        model,
                        is_instrumental,
                        title=title or None,
                        style=style or None,
                        negativeTags=negative_tags or None,
                        styleWeight=style_weight,
                        weirdnessConstraint=weirdness,
                        vocalGender=None if vocal_gender == "auto" else vocal_gender
                    )
                    st.success(f"✅ Task submitted: `{task_id[:12]}...`")
                
                # Poll for completion
                progress = st.progress(0)
                status_container = st.empty()
                
                with st.spinner("⏳ Generating (2-4 minutes)..."):
                    while True:
                        status = check_status(st.session_state.api_key, task_id)
                        status_val = status.get("status", "UNKNOWN")
                        
                        with status_container:
                            if status_val == "PENDING":
                                progress.progress(20)
                                st.info("⏳ In queue...")
                            elif status_val == "GENERATING":
                                progress.progress(60)
                                st.info("🎼 Generating audio...")
                            elif status_val == "SUCCESS":
                                progress.progress(100)
                                st.success("✅ Generation complete!")
                                break
                            elif status_val == "ERROR":
                                error_msg = status.get('error', 'Unknown error during generation')
                                logger.error(f"Task failed: {error_msg}")
                                st.error(f"❌ Generation failed: {error_msg}")
                                return
                            else:
                                st.warning(f"🤔 Status: {status_val}")
                                logger.warning(f"Unknown status: {status_val}")
                        
                        time.sleep(5)
                
                # Process results
                logger.info("Processing successful generation results")
                suno_data = status.get("response", {}).get("sunoData", [])
                
                if not suno_data:
                    logger.error("No sunoData in response")
                    st.error("❌ No audio data received from API")
                    logger.debug(f"Full status response: {json.dumps(status)}")
                    return
                
                # Store tracks in session state for UI persistence
                tracks = []
                for i, track in enumerate(suno_data):
                    clip_id = track.get('id')
                    track_title = track.get('title', f"Track {i+1}")
                    audio_url = track.get('audioUrl')
                    duration = track.get('duration', 0)
                    
                    if not audio_url:
                        logger.warning(f"No audioUrl for track {clip_id}")
                        continue
                    
                    # Download and save
                    safe_title = sanitize_filename(track_title)
                    filename = f"{safe_title}_{clip_id[-8:]}.mp3"
                    filepath = None
                    
                    if st.session_state.auto_save:
                        try:
                            filepath = download_file(audio_url, Path(AUDIO_FOLDER) / filename)
                        except Exception as e:
                            st.error(f"Download failed: {str(e)}")
                            filepath = None
                    
                    # Save to database
                    save_generation({
                        'id': f"{task_id}_{i}",
                        'clip_id': clip_id,
                        'prompt': prompt,
                        'title': track_title,
                        'model': model,
                        'is_instrumental': is_instrumental,
                        'duration': duration,
                        'audio_url': audio_url,
                        'audio_file': filepath,
                        'cover_images': [],
                        'status': 'SUCCESS'
                    })
                    
                    tracks.append({
                        'title': track_title,
                        'filepath': filepath,
                        'clip_id': clip_id,
                        'safe_title': safe_title
                    })
                
                # Store in session state for UI persistence
                st.session_state.generated_tracks = tracks
                
            # FIX: Better Retry Error Handling
            except RetryError as re:
                last_exc = re.last_attempt.exception()
                logger.error(f"API call failed after 3 retries: {str(last_exc)}", exc_info=True)
                st.error(f"❌ API Error (retried 3x): {str(last_exc)}")
                if st.session_state.debug_mode:
                    with st.expander("🔍 Debug Details"):
                        st.code(f"Last attempt exception: {type(last_exc).__name__}: {str(last_exc)}")
                        st.text(f"Log file: {log_file}")
            except Exception as e:
                logger.error(f"Generation error: {str(e)}", exc_info=True)
                st.error(f"❌ Error: {str(e)}")
                if st.session_state.debug_mode:
                    with st.expander("🔍 Debug Details"):
                        st.code(str(e))
                        st.text(f"Log file: {log_file}")
        
        # ==================== FIX: Persistent UI Display ====================
        # Display stored tracks (persists across reruns like download button clicks)
        if st.session_state.generated_tracks:
            st.markdown("### 🎧 Generated Tracks")
            
            for i, track in enumerate(st.session_state.generated_tracks):
                col_track, col_download = st.columns([3, 1])
                
                with col_track:
                    st.markdown(f"**{track['title']}**")
                    if track['filepath'] and Path(track['filepath']).exists():
                        render_audio_player(track['filepath'], track['title'])
                
                with col_download:
                    if track['filepath'] and Path(track['filepath']).exists():
                        with open(track['filepath'], 'rb') as f:
                            st.download_button(
                                "💾 Save MP3",
                                f,
                                Path(track['filepath']).name,
                                "audio/mpeg",
                                key=f"dl_persist_{i}"  # Unique key to prevent conflicts
                            )
                
                # Cover generation button
                cover_key = f"cover_persist_{track['clip_id']}"
                if st.button(f"🎨 Generate Cover", key=cover_key):
                    with st.spinner("Creating cover art..."):
                        conn = get_db()
                        cursor = conn.cursor()
                        cursor.execute("SELECT cover_images FROM generations WHERE clip_id = ?", (track['clip_id'],))
                        result = cursor.fetchone()
                        conn.close()
                        
                        if result and result[0]:
                            st.info("Cover already exists for this track")
                            covers = safe_json_loads(result[0])
                            if covers:
                                render_image_grid(covers)
                        else:
                            cover_task_id = generate_cover(st.session_state.api_key, track['clip_id'])
                            if cover_task_id:
                                while True:
                                    cover_status = check_cover_status(st.session_state.api_key, cover_task_id)
                                    if cover_status.get('status') == 'SUCCESS':
                                        images = cover_status.get('images', [])
                                        if images:
                                            cover_paths = []
                                            for j, img_url in enumerate(images):
                                                ext = img_url.split('.')[-1].split('?')[0]
                                                cover_filename = f"{track['safe_title']}_cover_{j+1}.{ext}"
                                                
                                                if st.session_state.auto_save:
                                                    cover_path = download_file(img_url, Path(AUDIO_FOLDER) / cover_filename)
                                                    cover_paths.append(cover_path)
                                            
                                            # Update DB
                                            conn = get_db()
                                            cursor = conn.cursor()
                                            cursor.execute(
                                                "UPDATE generations SET cover_images = ? WHERE clip_id = ?",
                                                (json.dumps(cover_paths), track['clip_id'])
                                            )
                                            conn.commit()
                                            conn.close()
                                            
                                            render_image_grid(cover_paths)
                                        break
                                    elif cover_status.get('status') == 'ERROR':
                                        st.error("Cover generation failed")
                                        break
                                    time.sleep(3)
                            else:
                                st.warning("Cover generation failed or already exists")
        # ========================================================================
    
    # ==================== HISTORY TAB ====================
    with tab2:
        st.subheader("📜 Generation History")
        
        search = st.text_input("🔍 Search", placeholder="Search prompts or titles...")
        
        col_page1, col_page2, col_page3 = st.columns([2, 2, 2])
        with col_page1:
            if st.button("⬅️ Previous", disabled=st.session_state.history_page == 0):
                st.session_state.history_page = max(0, st.session_state.history_page - 1)
                st.rerun()
        
        with col_page3:
            if st.button("➡️ Next"):
                st.session_state.history_page += 1
                st.rerun()
        
        # Load history with pagination
        offset = st.session_state.history_page * st.session_state.history_per_page
        history = load_history(
            limit=st.session_state.history_per_page,
            offset=offset,
            search=search
        )
        
        if not history:
            st.info("No history found. Generate some music!")
            if st.session_state.history_page > 0:
                st.session_state.history_page = 0
                st.rerun()
        else:
            st.caption(f"Page {st.session_state.history_page + 1} | Showing {len(history)} items")
            
            for entry in history:
                with st.expander(f"🎵 {entry['title']} ({entry['timestamp'][:16]})"):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**Prompt:** {entry['prompt'][:150]}...")
                        
                        duration_display_str = safe_duration_display(entry['duration'])
                        audio_file = entry['audio_file']
                        format_display = Path(audio_file).suffix.upper()[1:] if audio_file and Path(audio_file).exists() else 'N/A'
                        st.caption(f"Model: {entry['model']} | Duration: {duration_display_str} | Format: {format_display}")
                    
                    with col2:
                        if entry['audio_file'] and Path(entry['audio_file']).exists():
                            render_audio_player(entry['audio_file'], "")
                            
                            with open(entry['audio_file'], 'rb') as f:
                                ext = Path(entry['audio_file']).suffix[1:]
                                st.download_button(
                                    f"Save {ext.upper()}",
                                    f,
                                    Path(entry['audio_file']).name,
                                    f"audio/{ext}",
                                    key=f"hist_dl_{entry['id']}"
                                )
                    
                    covers = safe_json_loads(entry['cover_images'])
                    if covers:
                        st.markdown("**Covers:**")
                        render_image_grid(covers)
    
    # ==================== STATS TAB ====================
    with tab3:
        st.subheader("📊 Usage Statistics")
        
        stats = get_stats()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Generations", stats['total_gens'])
        with col2:
            hours = stats['total_duration'] / 3600
            st.metric("Total Hours", f"{hours:.1f}h")
        with col3:
            st.metric("Avg Duration", f"{stats['avg_duration'] or 0:.1f}s")
        
        st.markdown("### Model Usage")
        if stats['model_stats']:
            models = list(stats['model_stats'].keys())
            counts = list(stats['model_stats'].values())
            st.bar_chart(dict(zip(models, counts)))
        else:
            st.info("No data yet")

if __name__ == "__main__":
    main()
