import pyaudio
import webrtcvad
import speech_recognition as sr
import queue
import time
import threading
import logging
import json
import asyncio
import os
import platform
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, List, Any, Optional
import requests
from config.settings import BRAVE_SEARCH_API_KEY
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/voice", tags=["voice"])

# Global settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK_DURATION_MS = 30  # Duration of each chunk in milliseconds
CHUNK_SIZE = int(RATE * CHUNK_DURATION_MS / 1000)  # Chunk size in samples
VAD_MODE = 3  # Aggressiveness mode (3 is the most aggressive)
PADDING_DURATION_MS = 300  # Amount of padding to add to each side of speech detection

# Detect if running in Docker
IN_DOCKER = os.path.exists('/.dockerenv')

# Active clients
active_connections: List[WebSocket] = []

class VoiceAssistant:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.vad = webrtcvad.Vad(VAD_MODE)
        self.audio_interface = None
        self.stream = None
        self.audio_buffer = queue.Queue()
        self.is_listening = False
        self.thread = None
        self.mock_mode = False
        
    def start(self):
        """Initialize and start the audio capture"""
        try:
            # Check if we're in a Docker container or environment without audio devices
            if IN_DOCKER:
                logger.info("Running in Docker, using mock audio mode")
                self.mock_mode = True
                self.is_listening = True
                # Start a mock audio thread for Docker environments
                self.thread = threading.Thread(target=self._mock_audio_loop)
                self.thread.daemon = True
                self.thread.start()
                logger.info("Voice assistant mock mode started successfully")
                return True
                
            # Normal audio device initialization for non-Docker environments
            self.audio_interface = pyaudio.PyAudio()
            self.stream = self.audio_interface.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE,
                stream_callback=self._audio_callback
            )
            self.is_listening = True
            self.thread = threading.Thread(target=self._process_audio_loop)
            self.thread.daemon = True
            self.thread.start()
            logger.info("Voice assistant started successfully")
            return True
        except Exception as e:
            logger.error(f"Error starting voice assistant: {str(e)}")
            # Try mock mode as fallback if audio devices aren't available
            try:
                logger.info("Falling back to mock audio mode due to audio device error")
                self.mock_mode = True
                self.is_listening = True
                self.thread = threading.Thread(target=self._mock_audio_loop)
                self.thread.daemon = True
                self.thread.start()
                logger.info("Voice assistant mock mode started successfully")
                return True
            except Exception as mock_err:
                logger.error(f"Failed to start mock mode: {str(mock_err)}")
                self.stop()
                return False
            
    def stop(self):
        """Stop audio capture and processing"""
        self.is_listening = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.audio_interface:
            self.audio_interface.terminate()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.stream = None
        self.audio_interface = None
        self.mock_mode = False
        logger.info("Voice assistant stopped")
    
    def _mock_audio_loop(self):
        """
        Mock audio processing loop for Docker environments without audio devices.
        This just listens for WebSocket commands and processes them.
        """
        logger.info("Mock audio loop started")
        self._broadcast_status("ready")
        
        # Keep the thread running
        while self.is_listening:
            time.sleep(1)
    
    def _process_text_input(self, text):
        """Process text input directly (used for WebSocket text commands in mock mode)"""
        if not text:
            return
            
        logger.info(f"Processing text input: {text}")
        self._broadcast_status("processing")
        self._broadcast_result(text)
        self._process_command(text)
            
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for audio stream"""
        try:
            self.audio_buffer.put(in_data)
            return (in_data, pyaudio.paContinue)
        except Exception as e:
            logger.error(f"Error in audio callback: {str(e)}")
            return (in_data, pyaudio.paAbort)
    
    def _process_audio_loop(self):
        """Main processing loop for audio data"""
        ring_buffer = []
        voiced_frames = []
        is_speech = False
        num_voiced = 0
        num_unvoiced = 0
        
        while self.is_listening:
            try:
                # Get audio chunk from buffer
                if self.audio_buffer.empty():
                    time.sleep(0.01)
                    continue
                    
                chunk = self.audio_buffer.get()
                ring_buffer.append(chunk)
                
                # Check if the chunk contains speech
                if len(chunk) == CHUNK_SIZE * 2:  # 16-bit audio = 2 bytes per sample
                    is_current_speech = self.vad.is_speech(chunk, RATE)
                    
                    if is_current_speech:
                        num_voiced += 1
                        num_unvoiced = 0
                    else:
                        num_unvoiced += 1
                        
                    # Voice activity detection logic
                    if not is_speech and num_voiced > 2:
                        is_speech = True
                        voiced_frames = ring_buffer.copy()
                        ring_buffer = []
                        logger.info("Speech detected, recording...")
                        self._broadcast_status("listening")
                        
                    elif is_speech:
                        voiced_frames.append(chunk)
                        
                        # If we detect a significant silence after speech, process the audio
                        if num_unvoiced > 20:  # About 600ms of silence
                            is_speech = False
                            num_voiced = 0
                            num_unvoiced = 0
                            
                            # Process the speech
                            if len(voiced_frames) > 10:  # Ensure we have at least 300ms of audio
                                self._process_speech(voiced_frames)
                            
                            voiced_frames = []
                            ring_buffer = []
                
                # Keep the ring buffer at a reasonable size
                if len(ring_buffer) > 30:  # About 900ms of audio
                    ring_buffer.pop(0)
                    
            except Exception as e:
                logger.error(f"Error processing audio: {str(e)}")
                time.sleep(0.1)
    
    def _process_speech(self, audio_frames):
        """Process detected speech and convert to text"""
        self._broadcast_status("processing")
        logger.info("Processing speech...")
        
        try:
            # Combine all audio frames into a single buffer
            audio_data = b''.join(audio_frames)
            
            # Create an AudioData object from raw audio
            audio = sr.AudioData(audio_data, RATE, 2)  # 2 bytes per sample for 16-bit audio
            
            # Attempt to recognize the speech
            try:
                text = self.recognizer.recognize_google(audio)
                logger.info(f"Recognized text: {text}")
                self._broadcast_result(text)
                self._process_command(text)
            except sr.UnknownValueError:
                logger.info("Speech Recognition could not understand audio")
                self._broadcast_status("not_understood")
            except sr.RequestError as e:
                logger.error(f"Could not request results from Speech Recognition service; {e}")
                self._broadcast_status("error")
        except Exception as e:
            logger.error(f"Error processing speech: {str(e)}")
            self._broadcast_status("error")
    
    def _process_command(self, text):
        """Process the recognized command text"""
        text_lower = text.lower()
        
        # Check for search commands
        if "search for" in text_lower or "search the web for" in text_lower or "look up" in text_lower:
            # Extract search query
            search_terms = [
                "search for", 
                "search the web for", 
                "look up"
            ]
            
            for term in search_terms:
                if term in text_lower:
                    query = text_lower.split(term, 1)[1].strip()
                    self._perform_search(query)
                    return
        
        # If no specific command is detected, treat as a general query for agents
        self._send_to_agent(text)
    
    def _perform_search(self, query):
        """Perform a web search using the Brave Search API"""
        logger.info(f"Performing web search for: {query}")
        self._broadcast_status("searching")
        
        try:
            # Check if Brave Search API key is available
            if not BRAVE_SEARCH_API_KEY:
                logger.warning("Brave Search API key not configured. Using mock search results.")
                # Return mock search results
                results = [
                    {
                        "title": "Mock Search Result 1",
                        "url": "https://example.com/result1",
                        "description": "This is a mock search result because Brave Search API key is not configured."
                    },
                    {
                        "title": "Mock Search Result 2",
                        "url": "https://example.com/result2",
                        "description": "Configure BRAVE_SEARCH_API_KEY in your environment to get real search results."
                    }
                ]
                self._broadcast_search_results(query, results)
                return
                
            # Call Brave search API
            url = "https://api.search.brave.com/res/v1/web/search"
            headers = {
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": BRAVE_SEARCH_API_KEY
            }
            params = {"q": query, "count": 5}
            
            response = requests.get(url, headers=headers, params=params)
            search_data = response.json()
            
            results = []
            if "web" in search_data and "results" in search_data["web"]:
                for result in search_data["web"]["results"][:5]:  # Limit to 5 results
                    results.append({
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "description": result.get("description", "")
                    })
            
            # Broadcast search results
            self._broadcast_search_results(query, results)
            
        except Exception as e:
            logger.error(f"Error performing web search: {str(e)}")
            self._broadcast_status("search_error")
    
    def _send_to_agent(self, text):
        """Send the text to the agent system for processing"""
        if not text:
            return
            
        self._broadcast_status("thinking")
        
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                # Make an HTTP request to the chat endpoint - using localhost inside the container
                url = "http://localhost:8000/api/chat"
                payload = {
                    "message": text,
                    "conversation_id": None  # Create a new conversation
                }
                
                response = requests.post(url, json=payload, timeout=10)  # Add timeout
                
                if response.status_code == 200:
                    agent_response = response.json().get("response", "")
                    self._broadcast_agent_response(agent_response)
                    return
                else:
                    logger.error(f"Error from agent API (attempt {attempt+1}/{max_retries}): {response.status_code} - {response.text}")
                    if attempt == max_retries - 1:
                        self._broadcast_agent_response(f"Sorry, I encountered an error processing your request. Please try again.")
                    else:
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
            except requests.exceptions.Timeout:
                logger.error(f"Timeout connecting to agent system (attempt {attempt+1}/{max_retries})")
                if attempt == max_retries - 1:
                    self._broadcast_agent_response(f"Sorry, the agent system is taking too long to respond. Please try again later.")
                else:
                    time.sleep(retry_delay)
                    retry_delay *= 2
            except Exception as e:
                logger.error(f"Error connecting to agent system (attempt {attempt+1}/{max_retries}): {str(e)}")
                if attempt == max_retries - 1:
                    self._broadcast_agent_response(f"Sorry, I encountered an error: {str(e)}")
                else:
                    time.sleep(retry_delay)
                    retry_delay *= 2
    
    def _broadcast_status(self, status):
        """Broadcast status updates to all connected clients"""
        asyncio.run(self._async_broadcast({"type": "status", "status": status}))
    
    def _broadcast_result(self, text):
        """Broadcast speech recognition results to all connected clients"""
        asyncio.run(self._async_broadcast({"type": "speech_result", "text": text}))
    
    def _broadcast_search_results(self, query, results):
        """Broadcast search results to all connected clients"""
        asyncio.run(self._async_broadcast({
            "type": "search_results", 
            "query": query, 
            "results": results
        }))
    
    def _broadcast_agent_response(self, response):
        """Broadcast agent response to all connected clients"""
        asyncio.run(self._async_broadcast({
            "type": "agent_response", 
            "response": response
        }))
    
    async def _async_broadcast(self, message):
        """Asynchronously broadcast a message to all connected WebSocket clients"""
        for connection in active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting message: {str(e)}")

# Create global voice assistant instance
voice_assistant = VoiceAssistant()

@router.on_event("startup")
async def startup_voice_assistant():
    """Start the voice assistant when the API starts"""
    voice_assistant.start()

@router.on_event("shutdown")
async def shutdown_voice_assistant():
    """Stop the voice assistant when the API shuts down"""
    voice_assistant.stop()

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for voice assistant communication"""
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        # Send initial status
        await websocket.send_json({"type": "status", "status": "ready"})
        
        # Keep the connection alive and handle any client messages
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                if message.get("action") == "start_listening":
                    if not voice_assistant.is_listening:
                        success = voice_assistant.start()
                        await websocket.send_json({
                            "type": "status", 
                            "status": "listening" if success else "error"
                        })
                elif message.get("action") == "stop_listening":
                    if voice_assistant.is_listening:
                        voice_assistant.stop()
                        await websocket.send_json({"type": "status", "status": "stopped"})
                elif message.get("action") == "process_text" and voice_assistant.mock_mode:
                    # Process text directly in mock mode
                    text = message.get("text", "")
                    if text:
                        voice_assistant._process_text_input(text)
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON received: {data}")
                
    except WebSocketDisconnect:
        active_connections.remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        if websocket in active_connections:
            active_connections.remove(websocket)

@router.get("/status")
async def get_status():
    """Get the current status of the voice assistant"""
    return {"is_listening": voice_assistant.is_listening, "mock_mode": voice_assistant.mock_mode}

@router.post("/start")
async def start_voice_assistant():
    """Start the voice assistant"""
    success = voice_assistant.start()
    return {"success": success, "is_listening": voice_assistant.is_listening, "mock_mode": voice_assistant.mock_mode}

@router.post("/stop")
async def stop_voice_assistant():
    """Stop the voice assistant"""
    voice_assistant.stop()
    return {"success": True, "is_listening": voice_assistant.is_listening, "mock_mode": voice_assistant.mock_mode}

@router.post("/text")
async def process_text_input(text: str):
    """Process text input directly (for when voice recognition is not available)"""
    if not voice_assistant.is_listening:
        success = voice_assistant.start()
        if not success:
            return {"success": False, "error": "Could not start voice assistant"}
    
    voice_assistant._process_text_input(text)
    return {"success": True} 