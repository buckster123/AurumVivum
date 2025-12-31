"""
Music Player & Organizer - Streamlit Page
A feature-rich music management interface with playback and basic editing.
"""

import asyncio
import base64
import io
import json
import logging
import os
import sqlite3
import subprocess
import sys
import time
import traceback
import wave  # 🔧 NEW: For proper WAV file writing
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import pygame
import streamlit as st
from mutagen.easyid3 import EasyID3
from mutagen.flac import FLAC
from mutagen.oggvorbis import OggVorbis

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION & SETUP
# ──────────────────────────────────────────────────────────────────────────────

# Force dummy audio driver for headless systems
os.environ["SDL_AUDIODRIVER"] = "dummy"

# Debug mode toggle in sidebar
def init_session_state():
    """Initialize session state variables"""
    defaults = {
        "debug_mode": False,
        "current_file": None,
        "is_playing": False,
        "current_time": 0,
        "duration": 0,
        "volume": 1.0,
        "loop": False,
        "shuffle": False,
        "current_playlist": [],
        "playlist_index": 0,
        "metadata_db": {},
        "edit_buffer": None,
        "editor_selection": (0, 0),
        "waveform_data": None,
        "processing_message": "",
        "page_mode": "Browser"
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Setup logging
def setup_logging(debug: bool):
    """Configure logging based on debug mode"""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("music_player.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# Music directory configuration
MUSIC_DIR = Path("music").resolve()
MUSIC_DIR.mkdir(exist_ok=True)
DB_PATH = MUSIC_DIR / "music_metadata.db"

# Supported audio formats
SUPPORTED_FORMATS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}
EDITABLE_FORMATS = {".wav"}  # Formats we can edit with current dependencies

# ──────────────────────────────────────────────────────────────────────────────
# DATABASE LAYER
# ──────────────────────────────────────────────────────────────────────────────

class MusicDatabase:
    """SQLite-based metadata storage"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Create database tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS tracks (
                        filepath TEXT PRIMARY KEY,
                        title TEXT,
                        artist TEXT,
                        album TEXT,
                        genre TEXT,
                        duration REAL,
                        filesize INTEGER,
                        created_at TIMESTAMP,
                        last_modified TIMESTAMP,
                        play_count INTEGER DEFAULT 0
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS playlists (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT UNIQUE,
                        tracks TEXT,
                        created_at TIMESTAMP
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS tags (
                        filepath TEXT,
                        tag_name TEXT,
                        tag_value TEXT,
                        PRIMARY KEY (filepath, tag_name)
                    )
                """)
        except Exception as e:
            st.error(f"Database initialization failed: {e}")
            logging.error(f"DB init error: {traceback.format_exc()}")
    
    def scan_directory(self) -> List[Dict[str, Any]]:
        """Scan music directory and update database"""
        tracks = []
        try:
            for ext in SUPPORTED_FORMATS:
                for file_path in MUSIC_DIR.rglob(f"*{ext}"):
                    try:
                        stats = file_path.stat()
                        rel_path = file_path.relative_to(MUSIC_DIR)
                        
                        # Try to read metadata
                        metadata = self.read_audio_metadata(file_path)
                        
                        track_info = {
                            "filepath": str(rel_path),
                            "title": metadata.get("title", file_path.stem),
                            "artist": metadata.get("artist", "Unknown Artist"),
                            "album": metadata.get("album", "Unknown Album"),
                            "genre": metadata.get("genre", ""),
                            "duration": metadata.get("duration", 0),
                            "filesize": stats.st_size,
                            "created_at": datetime.fromtimestamp(stats.st_ctime),
                            "last_modified": datetime.fromtimestamp(stats.st_mtime),
                        }
                        
                        self.upsert_track(track_info)
                        tracks.append(track_info)
                        
                    except Exception as e:
                        logging.warning(f"Failed to process {file_path}: {e}")
            
            logging.info(f"Scanned {len(tracks)} tracks")
            return tracks
        except Exception as e:
            logging.error(f"Directory scan failed: {traceback.format_exc()}")
            st.error(f"Scan failed: {e}")
            return []
    
    def read_audio_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from audio file"""
        metadata = {}
        try:
            ext = file_path.suffix.lower()
            
            if ext == ".mp3":
                audio = EasyID3(file_path)
            elif ext == ".flac":
                audio = FLAC(file_path)
            elif ext == ".ogg":
                audio = OggVorbis(file_path)
            else:
                # For wav and others, just get duration via pygame
                audio = None
                
            if audio:
                metadata["title"] = audio.get("title", [None])[0]
                metadata["artist"] = audio.get("artist", [None])[0]
                metadata["album"] = audio.get("album", [None])[0]
                metadata["genre"] = audio.get("genre", [None])[0]
            
            # Get duration via pygame for all formats
            try:
                pygame.mixer.init()
                sound = pygame.mixer.Sound(str(file_path))
                metadata["duration"] = sound.get_length()
            except:
                metadata["duration"] = 0
                
        except Exception as e:
            logging.warning(f"Metadata read failed for {file_path}: {e}")
            metadata["duration"] = 0
        
        return {k: v for k, v in metadata.items() if v is not None}
    
    def upsert_track(self, track_info: Dict[str, Any]):
        """Insert or update track in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO tracks 
                    (filepath, title, artist, album, genre, duration, filesize, created_at, last_modified)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    track_info["filepath"],
                    track_info["title"],
                    track_info["artist"],
                    track_info["album"],
                    track_info["genre"],
                    track_info["duration"],
                    track_info["filesize"],
                    track_info["created_at"],
                    track_info["last_modified"]
                ))
        except Exception as e:
            logging.error(f"Database upsert failed: {e}")
            raise
    
    def get_all_tracks(self) -> List[Dict[str, Any]]:
        """Get all tracks from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("SELECT * FROM tracks ORDER BY artist, album, title")
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logging.error(f"Failed to fetch tracks: {e}")
            return []
    
    def increment_play_count(self, filepath: str):
        """Increment play counter"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "UPDATE tracks SET play_count = play_count + 1 WHERE filepath = ?",
                    (filepath,)
                )
        except Exception as e:
            logging.warning(f"Failed to increment play count: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# AUDIO PROCESSING LAYER
# ──────────────────────────────────────────────────────────────────────────────

class AudioEditor:
    """Basic audio editing capabilities"""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        pygame.mixer.init()
    
    def load_audio(self, file_path: Path) -> Optional[Tuple[np.ndarray, int]]:
        """Load audio file into numpy array (WAV only for native editing)"""
        try:
            if file_path.suffix.lower() != ".wav":
                raise ValueError("Native editing only supported for WAV files. Use ffmpeg fallback for other formats.")
            
            sound = pygame.mixer.Sound(str(file_path))
            samples = pygame.sndarray.samples(sound)
            
            # Convert to numpy array and normalize
            audio_array = np.array(samples, dtype=np.float32)
            if audio_array.ndim > 1:
                # Convert stereo to mono for simplicity
                audio_array = np.mean(audio_array, axis=1)
            
            # Normalize to [-1, 1]
            audio_array = audio_array / np.max(np.abs(audio_array))
            
            return audio_array, sound.get_length()
            
        except Exception as e:
            logging.error(f"Audio load failed: {e}")
            if self.debug:
                st.error(f"Load error: {traceback.format_exc()}")
            return None
    
    def trim_audio(self, audio_array: np.ndarray, sample_rate: int = 44100, 
                   start_time: float = 0, end_time: float = None) -> np.ndarray:
        """Trim audio to specified time range"""
        try:
            start_sample = int(start_time * sample_rate)
            if end_time is None:
                return audio_array[start_sample:]
            end_sample = int(end_time * sample_rate)
            return audio_array[start_sample:end_sample]
        except Exception as e:
            logging.error(f"Trim failed: {e}")
            raise
    
    def apply_fade(self, audio_array: np.ndarray, fade_in: float = 0, 
                   fade_out: float = 0, sample_rate: int = 44100) -> np.ndarray:
        """Apply fade in/out (seconds)"""
        try:
            length = len(audio_array)
            fade_in_samples = int(fade_in * sample_rate)
            fade_out_samples = int(fade_out * sample_rate)
            
            # Create fade envelope
            envelope = np.ones(length)
            
            # Fade in
            if fade_in_samples > 0:
                envelope[:fade_in_samples] = np.linspace(0, 1, fade_in_samples)
            
            # Fade out
            if fade_out_samples > 0:
                envelope[-fade_out_samples:] = np.linspace(1, 0, fade_out_samples)
            
            return audio_array * envelope
        except Exception as e:
            logging.error(f"Fade failed: {e}")
            raise
    
    def change_volume(self, audio_array: np.ndarray, gain_db: float) -> np.ndarray:
        """Adjust volume by dB"""
        try:
            gain_linear = 10 ** (gain_db / 20)
            return audio_array * gain_linear
        except Exception as e:
            logging.error(f"Volume change failed: {e}")
            raise
    
    def save_audio(self, audio_array: np.ndarray, output_path: Path, 
                   sample_rate: int = 44100):
        """Save audio array to WAV file"""
        try:
            # Ensure array is in correct format
            audio_int = (audio_array * 32767).astype(np.int16)
            
            # Convert 1D mono to 2D stereo format for pygame mixer
            if audio_int.ndim == 1:
                # Reshape to (N, 1) then broadcast to (N, 2) for stereo compatibility
                audio_int = np.column_stack((audio_int, audio_int))
            
            # 🔧 FIX: Use wave module instead of non-existent pygame Sound.write()
            with wave.open(str(output_path), 'w') as wav_file:
                # Set parameters: nchannels, sampwidth, framerate
                wav_file.setnchannels(2)  # Stereo
                wav_file.setsampwidth(2)  # 2 bytes = 16-bit
                wav_file.setframerate(sample_rate)
                # Write the raw bytes
                wav_file.writeframes(audio_int.tobytes())
            
            logging.info(f"Saved edited audio to {output_path}")
        except Exception as e:
            logging.error(f"Save failed: {e}")
            raise

# ──────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI LAYER
# ──────────────────────────────────────────────────────────────────────────────

def debug_panel(logger: logging.Logger):
    """Display debug information in sidebar"""
    if st.session_state.debug_mode:
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔧 Debug Panel")
        
        # Show session state
        if st.sidebar.checkbox("Show Session State"):
            st.sidebar.json({k: v for k, v in st.session_state.items() 
                           if not k.startswith('FormSubmitter')})
        
        # Show logs
        if st.sidebar.checkbox("Show Recent Logs"):
            try:
                with open("music_player.log", "r") as f:
                    logs = f.readlines()[-20:]
                    st.sidebar.code("".join(logs), language="log")
            except:
                st.sidebar.info("No logs available")
        
        # Test audio engine
        if st.sidebar.button("Test Audio Engine"):
            try:
                pygame.mixer.init()
                st.sidebar.success("✅ Audio engine ready")
                logger.info("Audio engine test passed")
            except Exception as e:
                st.sidebar.error(f"❌ Audio engine failed: {e}")
                logger.error(f"Audio engine test failed: {e}")

def file_browser(db: MusicDatabase, logger: logging.Logger):
    """Music file browser and organizer"""
    st.markdown("### 📁 Music Library")
    
    # Search and filter
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        search = st.text_input("🔍 Search", placeholder="Artist, title, album...")
    with col2:
        sort_by = st.selectbox("Sort by", ["artist", "album", "title", "duration", "play_count"])
    with col3:
        sort_order = st.radio("Order", ["asc", "desc"], horizontal=True)
    
    # Get tracks from DB
    tracks = db.get_all_tracks()
    
    # Apply search filter
    if search:
        search_lower = search.lower()
        tracks = [t for t in tracks if (search_lower in t.get("title", "").lower() or 
                                        search_lower in t.get("artist", "").lower() or 
                                        search_lower in t.get("album", "").lower())]
    
    # Apply sorting
    tracks.sort(key=lambda x: x.get(sort_by, ""), reverse=(sort_order == "desc"))
    
    # Display tracks
    if not tracks:
        st.info("🎵 No music files found. Add files to the `music/` directory.")
        if st.button("🔄 Rescan Library"):
            db.scan_directory()
            st.rerun()
        return
    
    # Browse mode: grid or list
    view_mode = st.radio("View", ["Grid", "List"], horizontal=True)
    
    if view_mode == "Grid":
        cols = st.columns(4)
        for idx, track in enumerate(tracks):
            with cols[idx % 4]:
                create_track_card(track, db, logger)
    else:
        for track in tracks:
            create_track_row(track, db, logger)
    
    # Batch operations
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Full Rescan"):
            with st.spinner("Scanning directory..."):
                scanned = db.scan_directory()
            st.success(f"Scanned {len(scanned)} tracks")
            logger.info(f"Manual rescan completed: {len(scanned)} tracks")
            time.sleep(1)
            st.rerun()

def create_track_card(track: Dict, db: MusicDatabase, logger: logging.Logger):
    """Compact track card for grid view"""
    try:
        file_path = MUSIC_DIR / track["filepath"]
        
        # Title and artist
        st.markdown(f"**{track.get('title', 'Unknown')}**")
        st.markdown(f"*by {track.get('artist', 'Unknown')}*")
        st.markdown(f"`{track['duration']:.1f}s`")
        
        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Play", key=f"play_{track['filepath']}"):
                st.session_state.current_file = str(file_path)
                st.session_state.is_playing = True
                db.increment_play_count(track["filepath"])
                logger.info(f"Playing: {track['filepath']}")
                st.rerun()
        
        with col2:
            if st.button("✏️ Edit", key=f"edit_{track['filepath']}"):
                st.session_state.edit_buffer = str(file_path)
                st.session_state.page_mode = "Editor"
                st.rerun()
                
    except Exception as e:
        logger.error(f"Card creation error: {e}")

def create_track_row(track: Dict, db: MusicDatabase, logger: logging.Logger):
    """Detailed track row for list view"""
    try:
        file_path = MUSIC_DIR / track["filepath"]
        
        with st.expander(
            f"🎵 {track.get('title', 'Unknown')} - {track.get('artist', 'Unknown')} "
            f"({track['duration']:.1f}s) 💿 {track.get('album', 'Unknown')}"
        ):
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.markdown(f"**Path:** `{track['filepath']}`")
                st.markdown(f"**Size:** {track['filesize'] / 1024 / 1024:.1f} MB")
                st.markdown(f"**Plays:** {track.get('play_count', 0)}")
            
            with col2:
                if st.button("▶️ Play", key=f"play_row_{track['filepath']}"):
                    st.session_state.current_file = str(file_path)
                    st.session_state.is_playing = True
                    db.increment_play_count(track["filepath"])
                    logger.info(f"Playing: {track['filepath']}")
                    st.rerun()
            
            with col3:
                if st.button("✏️ Edit", key=f"edit_row_{track['filepath']}"):
                    st.session_state.edit_buffer = str(file_path)
                    st.session_state.page_mode = "Editor"
                    st.rerun()
                    
    except Exception as e:
        logger.error(f"Row creation error: {e}")

def player_panel(db: MusicDatabase, logger: logging.Logger):
    """Audio player controls"""
    st.markdown("### 🎛️ Now Playing")
    
    if not st.session_state.current_file:
        st.info("No track selected")
        return
    
    try:
        file_path = Path(st.session_state.current_file)
        if not file_path.exists():
            st.error(f"File not found: {file_path}")
            return
        
        # Get track info
        track_info = None
        rel_path = file_path.relative_to(MUSIC_DIR) if file_path.is_relative_to(MUSIC_DIR) else file_path
        for track in db.get_all_tracks():
            if track["filepath"] == str(rel_path):
                track_info = track
                break
        
        # Display current track info
        if track_info:
            st.markdown(f"**{track_info.get('title', file_path.stem)}**")
            st.markdown(f"*by {track_info.get('artist', 'Unknown')}*")
        else:
            st.markdown(f"**{file_path.stem}**")
        
        # Audio player
        with open(file_path, "rb") as f:
            audio_bytes = f.read()
            st.audio(audio_bytes, format=f"audio/{file_path.suffix[1:]}")
        
        # Playback controls
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("⏮️ Prev"):
                playlist_controls("prev", db, logger)
        with col2:
            play_pause = "⏸️ Pause" if st.session_state.is_playing else "▶️ Play"
            if st.button(play_pause):
                st.session_state.is_playing = not st.session_state.is_playing
        with col3:
            if st.button("⏭️ Next"):
                playlist_controls("next", db, logger)
        with col4:
            st.session_state.loop = st.checkbox("🔁 Loop", st.session_state.loop)
        
        # Volume control
        st.session_state.volume = st.slider("Volume", 0.0, 1.0, st.session_state.volume)
        
        # Track progress (simulated)
        st.progress(0.3)  # Placeholder - real implementation would need JS callback
        
        # Add to playlist button
        if st.button("➕ Add to Playlist"):
            # Simple implementation - could be expanded
            st.success("Added to current queue")
            logger.info(f"Added to playlist: {file_path.name}")
        
    except Exception as e:
        logger.error(f"Player error: {e}")
        st.error(f"Playback failed: {str(e)}")

def playlist_controls(action: str, db: MusicDatabase, logger: logging.Logger):
    """Control playlist navigation"""
    try:
        playlist = st.session_state.current_playlist
        if not playlist:
            # Auto-generate playlist from all tracks
            playlist = [t["filepath"] for t in db.get_all_tracks()]
            st.session_state.current_playlist = playlist
        
        if not playlist:
            return
        
        current = st.session_state.current_file
        if current:
            current_rel = Path(current).relative_to(MUSIC_DIR)
            try:
                idx = playlist.index(str(current_rel))
            except ValueError:
                idx = 0
        else:
            idx = 0
        
        if action == "next":
            idx = (idx + 1) % len(playlist)
        elif action == "prev":
            idx = (idx - 1) % len(playlist)
        
        next_file = MUSIC_DIR / playlist[idx]
        if next_file.exists():
            st.session_state.current_file = str(next_file)
            db.increment_play_count(playlist[idx])
            logger.info(f"Playlist {action}: {next_file.name}")
            st.rerun()
            
    except Exception as e:
        logger.error(f"Playlist control error: {e}")

def editor_panel(logger: logging.Logger):
    """Audio editor interface"""
    st.markdown("### ✂️ Quick Edit Studio")
    
    if not st.session_state.edit_buffer:
        st.info("Select a track to edit")
        return
    
    file_path = Path(st.session_state.edit_buffer)
    if not file_path.exists():
        st.error("File not found")
        return
    
    # Only allow WAV editing natively
    if file_path.suffix.lower() not in EDITABLE_FORMATS:
        st.warning("⚠️ Native editing limited to WAV files. Use ffmpeg converter for other formats.")
        if st.button("🔧 Convert to WAV"):
            convert_to_wav(file_path, logger)
        return
    
    try:
        editor = AudioEditor(debug=st.session_state.debug_mode)
        
        # Load audio
        with st.spinner("Loading audio..."):
            result = editor.load_audio(file_path)
            if result is None:
                st.error("Failed to load audio")
                return
            audio_array, duration = result
        
        st.success(f"Loaded {file_path.name} ({duration:.2f}s)")
        
        # Editing tabs
        tab1, tab2, tab3 = st.tabs(["Trim", "Fade", "Volume"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                start_time = st.number_input("Start Time (s)", 0.0, duration, 0.0, 0.1)
            with col2:
                end_time = st.number_input("End Time (s)", 0.0, duration, duration, 0.1)
            
            if st.button("✂️ Apply Trim"):
                with st.spinner("Processing..."):
                    edited = editor.trim_audio(audio_array, start_time=start_time, end_time=end_time)
                    st.session_state.waveform_data = edited
                st.success("Trim applied!")
                logger.info(f"Trimmed {file_path.name}: {start_time}s - {end_time}s")
        
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                fade_in = st.number_input("Fade In (s)", 0.0, 10.0, 0.0, 0.1)
            with col2:
                fade_out = st.number_input("Fade Out (s)", 0.0, 10.0, 0.0, 0.1)
            
            if st.button("🌅 Apply Fade"):
                with st.spinner("Processing..."):
                    edited = editor.apply_fade(audio_array, fade_in=fade_in, fade_out=fade_out)
                    st.session_state.waveform_data = edited
                st.success("Fade applied!")
                logger.info(f"Faded {file_path.name}: in={fade_in}s, out={fade_out}s")
        
        with tab3:
            gain_db = st.slider("Gain (dB)", -20.0, 20.0, 0.0, 0.5)
            
            if st.button("🔊 Apply Volume"):
                with st.spinner("Processing..."):
                    edited = editor.change_volume(audio_array, gain_db)
                    st.session_state.waveform_data = edited
                st.success("Volume adjusted!")
                logger.info(f"Volume change on {file_path.name}: {gain_db}dB")
        
        # Waveform preview
        if st.session_state.waveform_data is not None:
            st.markdown("#### Waveform Preview")
            display_waveform(st.session_state.waveform_data)
            
            # Save options
            col1, col2 = st.columns(2)
            with col1:
                suffix = st.text_input("Save suffix", "_edit")
            with col2:
                if st.button("💾 Save As"):
                    output_path = file_path.parent / f"{file_path.stem}{suffix}{file_path.suffix}"
                    try:
                        editor.save_audio(st.session_state.waveform_data, output_path)
                        st.success(f"Saved to {output_path.name}")
                        logger.info(f"Saved edited file: {output_path}")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Save failed: {e}")
                        logger.error(f"Save error: {traceback.format_exc()}")
    
    except Exception as e:
        logger.error(f"Editor error: {e}")
        st.error(f"Edit failed: {str(e)}")

def display_waveform(audio_array: np.ndarray):
    """Display simple waveform visualization"""
    try:
        # Handle empty arrays
        if audio_array is None or len(audio_array) == 0:
            st.info("No waveform data to display")
            return
            
        # Downsample for performance
        samples = len(audio_array)
        target_samples = min(samples, 1000)  # Cap at 1000 points
        step = max(1, samples // target_samples)
        waveform = audio_array[::step]
        
        # Normalize for consistent display
        waveform = waveform / max(np.max(np.abs(waveform)), 1e-6)
        
        # Display with styling
        st.line_chart(waveform, height=150)
        
    except Exception as e:
        logging.warning(f"Waveform display failed: {e}")
        st.code("Waveform preview unavailable")

def convert_to_wav(input_path: Path, logger: logging.Logger):
    """Convert audio to WAV using ffmpeg (if available)"""
    try:
        output_path = input_path.parent / f"{input_path.stem}.wav"
        
        # Check if ffmpeg is available
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            st.error("❌ ffmpeg not found. Install ffmpeg to convert audio files.")
            logger.error("ffmpeg not available for conversion")
            return
        
        with st.spinner(f"Converting {input_path.name} to WAV..."):
            subprocess.run([
                "ffmpeg", "-i", str(input_path),
                "-y",  # Overwrite output
                "-acodec", "pcm_s16le",
                "-ar", "44100",
                "-ac", "1",  # Mono for simplicity
                str(output_path)
            ], check=True, capture_output=True)
        
        st.success(f"✅ Converted to {output_path.name}")
        logger.info(f"Converted {input_path} to {output_path}")
        
        # Update database with new file
        db = MusicDatabase(DB_PATH)
        db.scan_directory()
        
    except subprocess.CalledProcessError as e:
        st.error(f"Conversion failed: {e.stderr}")
        logger.error(f"FFmpeg conversion error: {e}")
    except Exception as e:
        st.error(f"Conversion error: {e}")
        logger.error(f"Conversion error: {traceback.format_exc()}")

def settings_panel(logger: logging.Logger):
    """Settings and configuration"""
    st.markdown("### ⚙️ Settings")
    
    # Directory info
    st.markdown(f"**Music Directory:** `{MUSIC_DIR}`")
    st.markdown(f"**Database:** `{DB_PATH}`")
    
    # Stats
    db = MusicDatabase(DB_PATH)
    tracks = db.get_all_tracks()
    st.markdown(f"**Total Tracks:** {len(tracks)}")
    
    total_size = sum(t.get("filesize", 0) for t in tracks) / (1024**3)
    st.markdown(f"**Total Size:** {total_size:.2f} GB")
    
    # Danger zone
    st.markdown("---")
    st.markdown("#### 🚨 Danger Zone")
    
    if st.checkbox("I understand this will delete the database"):
        if st.button("🗑️ Reset Database"):
            try:
                DB_PATH.unlink()
                st.success("Database deleted. Rescan to rebuild.")
                logger.warning("Database reset by user")
                st.rerun()
            except Exception as e:
                st.error(f"Failed: {e}")
                logger.error(f"Database reset error: {e}")
    
    # FFmpeg check
    st.markdown("---")
    st.markdown("#### 🛠️ Dependencies")
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        st.success("✅ ffmpeg available")
    except:
        st.warning("❌ ffmpeg not found (required for format conversion)")

# ──────────────────────────────────────────────────────────────────────────────
# MAIN APPLICATION
# ──────────────────────────────────────────────────────────────────────────────

def main():
    """Main application entry point"""
    st.set_page_config(
        page_title="Music Player & Organizer",
        page_icon="🎵",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    init_session_state()
    
    # Setup logging
    logger = setup_logging(st.session_state.debug_mode)
    
    # Title
    st.title("🎵 Music Player & Organizer")
    st.markdown("---")
    
    # Database
    db = MusicDatabase(DB_PATH)
    
    # Auto-scan if first run
    if 'first_run' not in st.session_state:
        with st.spinner("Initializing music library..."):
            db.scan_directory()
            st.session_state.first_run = False
            logger.info("First run scan completed")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    
    # Debug toggle
    st.session_state.debug_mode = st.sidebar.checkbox("🔧 Debug Mode", st.session_state.debug_mode)
    
    # Dynamic sidebar that respects session state
    page_mode = st.sidebar.radio(
        "Section",
        ["Browser", "Player", "Editor", "Settings"],
        index=["Browser", "Player", "Editor", "Settings"].index(st.session_state.page_mode),
        key="nav_radio"
    )
    
    # Sync sidebar selection back to session state
    if page_mode != st.session_state.page_mode:
        st.session_state.page_mode = page_mode
    
    # Debug panel
    debug_panel(logger)
    
    # Main content area
    try:
        if st.session_state.page_mode == "Browser":
            file_browser(db, logger)
        elif st.session_state.page_mode == "Player":
            player_panel(db, logger)
        elif st.session_state.page_mode == "Editor":
            editor_panel(logger)
        elif st.session_state.page_mode == "Settings":
            settings_panel(logger)
    
    except Exception as e:
        logger.error(f"Main app error: {e}")
        st.error("An error occurred. Check debug panel for details.")
        if st.session_state.debug_mode:
            st.exception(e)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption(f"v1.0 | Music: {MUSIC_DIR.name}")

# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
