import os
from typing import TypedDict, List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv
import numpy as np
import sounddevice as sd
import threading
import whisper
from scipy.io import wavfile
import tempfile
from gtts import gTTS
import pygame
import queue
import requests  # For Qwen API calls
import time
import json
from langgraph.graph import StateGraph, END
# Qwen model integrated directly in LLMEvaluator class




MAX_FOLLOWUPS_PER_QUESTION = 2


load_dotenv()


# Llama 3 Configuration (no API key needed!)
LLAMA_BASE_URL = os.getenv("LLAMA_BASE_URL", "http://10.151.223.207")
LLAMA_MODEL = os.getenv("LLAMA_MODEL", "llama3:8b")

print(f"✓ Using Llama model: {LLAMA_MODEL}")
print(f"✓ Llama endpoint: {LLAMA_BASE_URL}")



@dataclass
class ConversationEntry:
    """Represents a single Q&A exchange in the interview"""
    question: str
    answer: str
    feedback: str
    is_followup: bool
    timestamp: datetime
    evaluation_score: Optional[float] = None


@dataclass
class EvaluationResult:
    """Result of answer evaluation by LLM"""
    feedback: str
    is_correct: bool
    is_partial: bool
    is_incomplete: bool
    needs_followup: bool
    score: float  # 0.0 to 1.0
    strengths: List[str]
    weaknesses: List[str]


@dataclass
class InterviewConfig:
    """Configuration for interview session"""
    mode: str  # "role" or "resume"
    role_or_resume_content: str
    max_questions: int = 10
    sample_rate: int = 16000
    audio_device_index: int = 8
    whisper_model: str = "small"
    gemini_model: str = "gemini-1.5-flash"


class InterviewState(TypedDict):
    
    mode: str  # "role" or "resume"
    role_or_resume: str  # Job role or resume summary
    current_question: str
    current_answer: str
    conversation_history: List[Dict[str, Any]]  # <-- fixed: Any instead of str
    question_count: int
    needs_followup: bool
    interview_active: bool
    consolidated_feedback: str


# Components

class AudioRecorder:
    """
    Captures audio input with manual start/stop control.
    Manages audio stream lifecycle and provides recording state tracking.
    """
    
    def __init__(self, sample_rate: int = 16000, device_index: Optional[int] = None):
        self.sample_rate = sample_rate
        self.device_index = device_index
        self._recording = False
        self._audio_data = []
        self._stream = None
        self._lock = threading.Lock()
    
    def start_recording(self) -> None:
        with self._lock:
            if self._recording:
                print("Warning: Already recording. Please stop current recording first.")
                return
            
            try:
                self._audio_data = []
                self._recording = True
                
                self._stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype='float32',
                    device=self.device_index,
                    callback=self._audio_callback
                )
                self._stream.start()
                print("🔴 Recording started...")
                
            except Exception as e:
                self._recording = False
                self._audio_data = []
                raise RuntimeError(f"Failed to start recording: {str(e)}")
    
    def stop_recording(self) -> np.ndarray:
        with self._lock:
            if not self._recording:
                print("Warning: Not currently recording.")
                return np.array([], dtype=np.int16)
            
            try:
                self._recording = False
                
                if self._stream:
                    self._stream.stop()
                    self._stream.close()
                    self._stream = None
                
                print("⏹️  Recording stopped.")
                
                if len(self._audio_data) == 0:
                    return np.array([], dtype=np.int16)
                
                audio_array = np.concatenate(self._audio_data, axis=0)
                audio_int16 = (audio_array * 32767).astype(np.int16)
                
                return audio_int16
                
            except Exception as e:
                print(f"Error stopping recording: {str(e)}")
                return np.array([], dtype=np.int16)
    
    def is_recording(self) -> bool:
        with self._lock:
            return self._recording
    
    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"Audio callback status: {status}")
        
        if self._recording:
            self._audio_data.append(indata.copy())


class SpeechTranscriber:
    """
    Converts audio to text using OpenAI's Whisper model.
    """
    
    def __init__(self, model_name: str = "small"):
        self.model_name = model_name
        self._model = None
        self._load_model()
    
    def _load_model(self) -> None:
        try:
            print(f"Loading Whisper {self.model_name} model...")
            self._model = whisper.load_model(self.model_name)
            print(f"✓ Whisper {self.model_name} model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load Whisper model '{self.model_name}': {str(e)}")
    
    def transcribe(self, audio_path: str) -> str:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            print("🎧 Transcribing audio...")
            result = self._model.transcribe(audio_path, fp16=False)
            transcribed_text = result["text"].strip()
            
            if not transcribed_text:
                print("⚠️ Transcription resulted in empty text (silent audio)")
                return ""
            
            print(f"✓ Transcription complete: {len(transcribed_text)} characters")
            return transcribed_text
            
        except Exception as e:
            raise RuntimeError(f"Transcription failed: {str(e)}")
    
    def transcribe_array(self, audio_data: np.ndarray, sample_rate: int) -> str:
        if audio_data is None or len(audio_data) == 0:
            raise ValueError("Audio data is empty")
        
        temp_file = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_file = f.name
            
            wavfile.write(temp_file, sample_rate, audio_data)
            transcribed_text = self.transcribe(temp_file)
            return transcribed_text
            
        except Exception as e:
            raise RuntimeError(f"Array transcription failed: {str(e)}")
        
        finally:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass


class TTSEngine:
    """
    Text-to-speech engine using gTTS (Google Text-to-Speech).
    Fixed timing issues for reliable playback.
    """
    def __init__(self, lang: str = 'en'):
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=4096)
            self.lang = lang
            self._temp_files = []  # Track temp files for cleanup
            print("✓ TTS Engine (gTTS) initialized successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize TTS engine: {str(e)}")

    def speak(self, text: str) -> None:
        if not text or not text.strip():
            return
        
        temp_file = None
        try:
            # Clean up old temp files first
            self._cleanup_old_files()
            
            # Create temporary file for audio
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
                temp_file = f.name
            
            # Generate speech using gTTS
            print("🔊 Generating speech...")
            tts = gTTS(text=text, lang=self.lang, slow=False)
            tts.save(temp_file)
            
            # Small delay to ensure file is fully written
            time.sleep(0.3)
            
            # Stop any currently playing audio
            pygame.mixer.music.stop()
            time.sleep(0.1)
            
            # Load and play audio
            pygame.mixer.music.load(temp_file)
            pygame.mixer.music.play()
            
            # Wait for audio to finish playing
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)  # More efficient waiting
            
            # Add delay after playback
            time.sleep(0.5)
            
            print(f"🔊 Spoke {len(text)} characters")
            
            # Track file for later cleanup
            self._temp_files.append(temp_file)
            
        except Exception as e:
            print(f"TTS Error: {str(e)}")
            # Try to clean up on error
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass

    def _cleanup_old_files(self) -> None:
        """Clean up old temporary files"""
        try:
            pygame.mixer.music.unload()
        except:
            pass
        
        time.sleep(0.2)
        
        for f in self._temp_files[:]:
            try:
                if os.path.exists(f):
                    os.remove(f)
                self._temp_files.remove(f)
            except:
                pass

    def stop(self) -> None:
        try:
            pygame.mixer.music.stop()
            self._cleanup_old_files()
            print("✓ TTS Engine stopped")
        except:
            pass



class LLMEvaluator:
    """
    Evaluates answers and generates questions using Llama 3:8b (Local Model).
    No API key needed - uses local Docker instance!
    """
    
    def __init__(self, base_url: str = "http://10.151.223.207", model: str = "llama3:8b"):
        self.base_url = base_url.rstrip('/')
        self.model_name = model
        self.generate_url = f"{self.base_url}/api/generate"
        print(f"✓ Llama {model} initialized successfully")
        print(f"✓ Using endpoint: {self.generate_url}")
    
    def _call_qwen(self, prompt: str, max_retries: int = 3) -> str:
        """Call Qwen model with retry logic"""
        for attempt in range(max_retries):
            try:
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,      # Balanced creativity
                        "top_p": 0.9,            # Diverse but focused
                        "top_k": 40,             # Limit token choices
                        "repeat_penalty": 1.15,  # Reduce repetition (slightly higher for Llama)
                        "num_predict": 512,      # Max tokens
                        "stop": ["<|eot_id|>", "<|end_of_text|>"]  # Llama 3 stop tokens
                    }
                }
                
                response = requests.post(
                    self.generate_url,
                    json=payload,
                    timeout=120
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get("response", "").strip()
                else:
                    raise Exception(f"API returned status {response.status_code}")
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Request failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"Failed after {max_retries} attempts: {str(e)}")
    
    def generate_first_question(self) -> str:
        return "Tell me about yourself"
    
    def generate_question(self, state: InterviewState) -> str:
        mode = state.get("mode", "role")
        role_or_resume = state.get("role_or_resume", "")
        question_count = state.get("question_count", 0)
        conversation_history = state.get("conversation_history", [])
        
        history_context = ""
        if conversation_history:
            history_context = "\n\nPrevious conversation:\n"
            for entry in conversation_history[-3:]:
                history_context += f"Q: {entry.get('question', '')}\n"
                history_context += f"A: {entry.get('answer', '')}\n"
        
        if mode == "role":
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert technical interviewer. Your task is to generate interview questions. Always respond with ONLY the question text, nothing else.<|eot_id|><|start_header_id|>user<|end_header_id|>
Generate ONE technical interview question for the role: {role_or_resume}

This is question #{question_count + 1}.
{history_context}

Requirements:
- Test practical knowledge and problem-solving
- Be specific to the role
- Different from previous questions
- Answerable in 2-3 minutes

Respond with ONLY the question text.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        
        else:
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert technical interviewer. Generate questions based on the candidate's resume. Always respond with ONLY the question text.<|eot_id|><|start_header_id|>user<|end_header_id|>
Candidate's Background:
{role_or_resume}

This is question #{question_count + 1}.
{history_context}

Generate ONE question that:
- Relates to their stated experience
- Tests depth of knowledge
- Different from previous questions
- Requires specific examples

Respond with ONLY the question text.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        
        return self._call_qwen(prompt)
    
    def evaluate_answer(self, question: str, answer: str, context: str = "") -> Dict[str, Any]:
        if not answer or not answer.strip():
            return {
                "feedback": "No answer provided",
                "is_correct": False,
                "is_partial": False,
                "is_incomplete": True,
                "needs_followup": True,
                "score": 0.0,
                "strengths": [],
                "weaknesses": ["No answer provided"]
            }
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert technical interviewer. Evaluate answers and respond ONLY with valid JSON. No markdown, no explanations.<|eot_id|><|start_header_id|>user<|end_header_id|>
Question: {question}
Answer: {answer}
Context: {context if context else "Technical interview"}

Evaluate and return ONLY this JSON format:
{{"feedback": "brief feedback", "is_correct": true/false, "is_partial": true/false, "is_incomplete": true/false, "needs_followup": true/false, "score": 0.0-1.0, "strengths": ["strength1"], "weaknesses": ["weakness1"]}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        
        try:
            response_text = self._call_qwen(prompt)
            
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            evaluation = json.loads(response_text)
            
            required_fields = ["feedback", "is_correct", "is_partial", "is_incomplete", 
                               "needs_followup", "score", "strengths", "weaknesses"]
            if all(field in evaluation for field in required_fields):
                return evaluation
            else:
                raise ValueError("Missing required fields in evaluation")
                
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Failed to parse evaluation JSON: {str(e)}")
            return {
                "feedback": "Unable to evaluate answer properly",
                "is_correct": False,
                "is_partial": True,
                "is_incomplete": True,
                "needs_followup": False,
                "score": 0.5,
                "strengths": [],
                "weaknesses": ["Evaluation error"]
            }
    
    def generate_followup(self, question: str, answer: str, evaluation: Dict[str, Any]) -> str:
        weaknesses = evaluation.get("weaknesses", [])
        weaknesses_text = ", ".join(weaknesses) if weaknesses else "incomplete information"
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert technical interviewer. Generate follow-up questions. Respond with ONLY the question text.<|eot_id|><|start_header_id|>user<|end_header_id|>
Original Question: {question}
Candidate's Answer: {answer}
Gaps Identified: {weaknesses_text}

Generate ONE follow-up question that:
- Addresses the gaps
- Helps demonstrate deeper understanding
- Is specific and focused
- Encouraging but probing

Respond with ONLY the follow-up question.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        
        return self._call_qwen(prompt)
    
    def generate_consolidated_feedback(self, conversation_history: List[Dict[str, Any]]) -> str:
        if not conversation_history:
            return "No conversation data available for feedback."
        
        conversation_summary = ""
        for i, entry in enumerate(conversation_history, 1):
            conversation_summary += f"\n--- Question {i} ---\n"
            conversation_summary += f"Q: {entry.get('question', 'N/A')}\n"
            conversation_summary += f"A: {entry.get('answer', 'N/A')}\n"
            if 'feedback' in entry:
                conversation_summary += f"Evaluation: {entry.get('feedback', 'N/A')}\n"
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert technical interviewer providing comprehensive feedback. Be specific, constructive, and encouraging.<|eot_id|><|start_header_id|>user<|end_header_id|>
Interview Transcript:
{conversation_summary}

Provide feedback covering:
1. Overall Performance (2-3 sentences)
2. Key Strengths (3-4 points)
3. Areas for Improvement (3-4 points)
4. Actionable Recommendations (3-4 items)
5. Encouraging Closing (1-2 sentences)

Be specific, balanced, and professional.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        
        return self._call_qwen(prompt)


# LangGraph Nodes

def initialize_node(state: InterviewState) -> InterviewState:
    print("\n" + "=" * 50)
    print("AI Interview Bot - Initialization")
    print("=" * 50)
    
    print("\nSelect interview mode:")
    print("1. Role-based interview")
    print("2. Resume-based interview")
    
    mode_choice = input("Enter choice (1 or 2): ").strip()
    
    if mode_choice == "1":
        state["mode"] = "role"
        role = input("\nEnter the job role for the interview: ").strip()
        state["role_or_resume"] = role
        print(f"\n✓ Role-based interview mode selected: {role}")
    elif mode_choice == "2":
        state["mode"] = "resume"
        print("\nEnter resume content or summary (press Enter twice when done):")
        resume_lines = []
        while True:
            line = input()
            if line == "" and resume_lines and resume_lines[-1] == "":
                break
            resume_lines.append(line)
        resume = "\n".join(resume_lines).strip()
        state["role_or_resume"] = resume
        print(f"\n✓ Resume-based interview mode selected")
    else:
        print("\nInvalid choice, defaulting to role-based mode")
        state["mode"] = "role"
        state["role_or_resume"] = "Software Engineer"
    
    state["current_question"] = ""
    state["current_answer"] = ""
    state["conversation_history"] = []
    state["question_count"] = 0
    state["needs_followup"] = False
    state["interview_active"] = True
    state["consolidated_feedback"] = ""
    
    print("\n✓ Interview initialized successfully")
    print("\nThe interview will now begin...")
    
    return state


def ask_question_node(state: InterviewState, llm_evaluator: LLMEvaluator, tts_engine: TTSEngine) -> InterviewState:
    print("\n" + "=" * 60)
    
    is_followup = state.get("needs_followup", False)
    
    total_main_questions = len([e for e in state.get("conversation_history", []) if not e.get("is_followup", False)])
    print(f"📊 Progress: Question {total_main_questions + 1} | Total Exchanges: {len(state.get('conversation_history', []))}")
    print("=" * 60)
    
    if state["question_count"] == 0 and len(state["conversation_history"]) == 0:
        question = llm_evaluator.generate_first_question()
        print(f"\n📝 Question 1: {question}")
    elif is_followup:
        last_entry = state["conversation_history"][-1] if state["conversation_history"] else {}
        question = llm_evaluator.generate_followup(
            last_entry.get("question", ""),
            last_entry.get("answer", ""),
            last_entry
        )
        print(f"\n🔄 Follow-up: {question}")
    else:
        question = llm_evaluator.generate_question(state)
        actual_question_num = len([e for e in state["conversation_history"] if not e.get("is_followup", False)]) + 1
        print(f"\n📝 Question {actual_question_num}: {question}")
    
    state["current_question"] = question
    
    parent_question_index = None
    if is_followup and len(state["conversation_history"]) > 0:
        for i in range(len(state["conversation_history"]) - 1, -1, -1):
            if not state["conversation_history"][i].get("is_followup", False):
                parent_question_index = i
                break
    
    question_entry: Dict[str, Any] = {
        "question": question,
        "answer": "",
        "feedback": "",
        "is_followup": is_followup,
        "parent_question_index": parent_question_index,
        "timestamp": datetime.now().isoformat(),
        "evaluation_score": None,
        "strengths": [],
        "weaknesses": [],
        "needs_followup": False
    }
    
    state["conversation_history"].append(question_entry)
    
    if is_followup:
        state["needs_followup"] = False
    
    try:
        tts_engine.speak(question)
        time.sleep(0.5)
    except Exception as e:
        print(f"⚠️ TTS Error: {e}")
    
    return state


def record_answer_node(state: InterviewState, audio_recorder: AudioRecorder, speech_transcriber: SpeechTranscriber, timeout: int = 120) -> InterviewState:
    print("\n[DEBUG] Entered record_answer_node")
    print("\n" + "-" * 60)
    print("💬 YOUR TURN TO ANSWER")
    print("-" * 60)
    print("\n📋 Commands:")
    print("   • 'start' - Begin recording your answer")
    print("   • 'stop'  - End recording and transcribe")
    print("   • 'quit'  - End interview early")
    print(f"\n⏱️  Response timeout: {timeout} seconds")
    print("-" * 60)
    
    answer_text = ""
    start_time = time.time()
    retry_count = 0
    max_retries = 3
    
    while True:
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout:
            print(f"\n⏰ Timeout: No response received within {timeout} seconds")
            print("⚠️ Please provide an answer to continue the interview.")
            
            continue_choice = input("Would you like to continue? (yes/no): ").strip().lower()
            if continue_choice == "yes":
                start_time = time.time()
                print("\n💬 Timer reset. Please answer the question:")
                continue
            else:
                state["interview_active"] = False
                state["current_answer"] = "[Timeout - No response]"
                return state
        
        command = input("\n> ").strip().lower()
        
        if command == "quit":
            state["interview_active"] = False
            state["current_answer"] = ""
            return state
        
        elif command == "start":
            if not audio_recorder.is_recording():
                try:
                    audio_recorder.start_recording()
                    print("\n" + "=" * 60)
                    print("🔴 RECORDING IN PROGRESS")
                    print("=" * 60)
                    print("Speak your answer clearly into the microphone...")
                    print("Type 'stop' when you're done speaking.")
                    print("=" * 60)
                except RuntimeError as e:
                    print(f"\n❌ Audio device error: {str(e)}")
                    print("\n⚠️ Unable to access microphone. Please check:")
                    print("   • Microphone is connected")
                    print("   • Microphone permissions are granted")
                    print("   • No other application is using the microphone")
                    
                    retry_device = input("\nWould you like to try again? (yes/no): ").strip().lower()
                    if retry_device == "yes":
                        continue
                    else:
                        state["interview_active"] = False
                        state["current_answer"] = "[Audio device error]"
                        return state
            else:
                print("\n⚠️ Already recording. Type 'stop' to end current recording.")
        
        elif command == "stop":
            if audio_recorder.is_recording():
                audio_data = audio_recorder.stop_recording()
                
                if len(audio_data) == 0:
                    print("⚠️ No audio captured. Please try again.")
                    continue
                
                try:
                    answer_text = speech_transcriber.transcribe_array(audio_data, audio_recorder.sample_rate)
                    
                    if not answer_text or not answer_text.strip():
                        retry_count += 1
                        print(f"\n⚠️ Transcription resulted in empty text (silent audio or unintelligible)")
                        
                        if retry_count < max_retries:
                            print(f"   Retry {retry_count}/{max_retries}")
                            retry = input("Would you like to record again? (yes/no): ").strip().lower()
                            if retry == "yes":
                                continue
                            else:
                                answer_text = "[No answer provided]"
                                break
                        else:
                            print(f"   Maximum retries ({max_retries}) reached")
                            answer_text = "[No answer provided after retries]"
                            break
                    
                    print("\n" + "=" * 60)
                    print("✓ TRANSCRIPTION COMPLETE")
                    print("=" * 60)
                    print(f"\n📝 Your answer:\n\"{answer_text}\"")
                    print("\n" + "=" * 60)
                    break
                    
                except Exception as e:
                    retry_count += 1
                    print(f"❌ Transcription failed: {str(e)}")
                    
                    if retry_count < max_retries:
                        print(f"   Retry {retry_count}/{max_retries}")
                        retry = input("Would you like to try again? (yes/no): ").strip().lower()
                        if retry == "yes":
                            continue
                        else:
                            answer_text = "[Transcription failed]"
                            break
                    else:
                        print(f"   Maximum retries ({max_retries}) reached")
                        answer_text = "[Transcription failed after retries]"
                        break
            else:
                print("Not currently recording. Type 'start' first.")
        
        else:
            print("Invalid command. Use 'start', 'stop', or 'quit'")
    
    state["current_answer"] = answer_text
    
    if len(state["conversation_history"]) > 0:
        state["conversation_history"][-1]["answer"] = answer_text
    
    return state


def evaluate_answer_node(state: InterviewState, llm_evaluator: LLMEvaluator) -> InterviewState:
    question = state["current_question"]
    answer = state["current_answer"]
    context = f"Mode: {state['mode']}, Context: {state['role_or_resume']}"
    
    print("\n🤔 Evaluating your answer...")
    
    try:
        evaluation = llm_evaluator.evaluate_answer(question, answer, context)
        
        if len(state["conversation_history"]) > 0:
            entry = state["conversation_history"][-1]
            entry["feedback"] = evaluation.get("feedback", "")
            entry["evaluation_score"] = evaluation.get("score", 0.0)
            entry["strengths"] = evaluation.get("strengths", [])
            entry["weaknesses"] = evaluation.get("weaknesses", [])
            entry["needs_followup"] = evaluation.get("needs_followup", False)
            
            # -------------------------------d
            # FOLLOW-UP LIMIT HANDLING (MAX 4)
            # -------------------------------
            idx = len(state["conversation_history"]) - 1
            if entry.get("is_followup"):
                main_idx = entry.get("parent_question_index")
                if main_idx is None:
                    main_idx = idx
            else:
                main_idx = idx
            
            followup_count = sum(
                1
                for e in state["conversation_history"]
                if e.get("is_followup") and e.get("parent_question_index") == main_idx
            )
            
            # Only allow follow-up if:
            #  - LLM says needs_followup
            #  - current followup_count < MAX_FOLLOWUPS_PER_QUESTION
            needs_followup_flag = evaluation.get("needs_followup", False) and followup_count < MAX_FOLLOWUPS_PER_QUESTION
            
            state["needs_followup"] = needs_followup_flag
            entry["needs_followup"] = needs_followup_flag
        
        print("✓ Evaluation complete (stored internally)")
        
    except Exception as e:
        print(f"⚠️ Evaluation error: {str(e)}")
        if len(state["conversation_history"]) > 0:
            state["conversation_history"][-1]["feedback"] = "Evaluation unavailable"
            state["conversation_history"][-1]["evaluation_score"] = 0.5
            state["conversation_history"][-1]["needs_followup"] = False
        
        state["needs_followup"] = False
    
    return state


def decide_next_node(state: InterviewState, llm_evaluator: LLMEvaluator, tts_engine: TTSEngine, max_questions: int = 10) -> InterviewState:
    if not state["interview_active"]:
        print("\n📊 Interview ended early by user")
        return state
    
    if state["needs_followup"]:
        print("\n🔄 Follow-up question will be generated...")
        # Do NOT increment main question count for follow-ups
    else:
        if state["question_count"] == 0 or len(state["conversation_history"]) > 0:
            state["question_count"] += 1
        
        if state["question_count"] >= max_questions:
            print(f"\n✓ Completed {state['question_count']} main questions")
            state["interview_active"] = False
        else:
            print(f"\n✓ Moving to question {state['question_count'] + 1}")
    
    return state


def consolidate_node(state: InterviewState, llm_evaluator: LLMEvaluator, tts_engine: TTSEngine, max_questions: int = 10) -> InterviewState:
    print("\n" + "=" * 60)
    
    is_early_termination = state.get("question_count", 0) < max_questions and len(state.get("conversation_history", [])) > 0
    
    if is_early_termination:
        print("           INTERVIEW ENDED EARLY")
        print("=" * 60)
        print(f"📊 Questions completed: {state.get('question_count', 0)}/{max_questions}")
        print(f"📊 Total exchanges: {len(state.get('conversation_history', []))}")
    else:
        print("           INTERVIEW COMPLETE")
        print("=" * 60)
        print(f"📊 Questions completed: {state.get('question_count', 0)}/{max_questions}")
        print(f"📊 Total exchanges: {len(state.get('conversation_history', []))}")
    
    print("=" * 60)
    
    try:
        if not state.get("conversation_history") or len(state["conversation_history"]) == 0:
            feedback = "No conversation data available. The interview ended before any questions were answered."
        else:
            print("\n⏳ Generating comprehensive feedback...")
            feedback = llm_evaluator.generate_consolidated_feedback(state["conversation_history"])
        
        state["consolidated_feedback"] = feedback
        
        print("\n" + "=" * 60)
        if is_early_termination:
            print("           PARTIAL FEEDBACK")
            print("     (Based on available responses)")
        else:
            print("           CONSOLIDATED FEEDBACK")
        print("=" * 60)
        print()
        
        feedback_lines = feedback.split('\n')
        for line in feedback_lines:
            if line.strip():
                print(f"  {line}")
            else:
                print()
        
        print("\n" + "=" * 60)
        
        print("\n🔊 Speaking feedback aloud...")
        try:
            tts_engine.speak(feedback)
            print("(Audio playing in background)")
        except Exception as e:
            print(f"⚠️ TTS Error: {e}")
        
        print("\n✓ Feedback delivery complete")
        
    except Exception as e:
        print(f"\n❌ Failed to generate consolidated feedback: {str(e)}")
        state["consolidated_feedback"] = "Unable to generate feedback due to an error."
    
    return state


def build_interview_graph(
    audio_recorder: AudioRecorder,
    speech_transcriber: SpeechTranscriber,
    tts_engine: TTSEngine,
    llm_evaluator: LLMEvaluator,
    max_questions: int = 10
):
    """
    Build and compile the LangGraph state machine for the interview flow.
    """
    graph = StateGraph(InterviewState)
    
    graph.add_node("initialize", initialize_node)
    graph.add_node("ask_question", lambda state: ask_question_node(state, llm_evaluator, tts_engine))
    graph.add_node("record_answer", lambda state: record_answer_node(state, audio_recorder, speech_transcriber, timeout=120))
    graph.add_node("evaluate_answer", lambda state: evaluate_answer_node(state, llm_evaluator))
    graph.add_node("decide_next", lambda state: decide_next_node(state, llm_evaluator, tts_engine, max_questions))
    graph.add_node("consolidate", lambda state: consolidate_node(state, llm_evaluator, tts_engine))
    
    graph.set_entry_point("initialize")
    
    graph.add_edge("initialize", "ask_question")
    graph.add_edge("ask_question", "record_answer")
    graph.add_edge("record_answer", "evaluate_answer")
    graph.add_edge("evaluate_answer", "decide_next")
    
    def route_after_decision(state: InterviewState) -> str:
        if not state.get("interview_active", True):
            return "consolidate"
        
        if state.get("needs_followup", False) or state.get("interview_active", True):
            return "ask_question"
        
        return "consolidate"
    
    graph.add_conditional_edges(
        "decide_next",
        route_after_decision,
        {
            "ask_question": "ask_question",
            "consolidate": "consolidate"
        }
    )
    
    graph.add_edge("consolidate", END)
    
    compiled_graph = graph.compile()
    
    print("✓ LangGraph state machine compiled successfully")
    
    return compiled_graph


def main():
    print("\n" + "=" * 60)
    print("           AI INTERVIEW BOT V2")
    print("=" * 60)
    print("\nWelcome to the AI-powered interview system!")
    print("This bot will conduct a technical interview with you.")
    print("\n" + "=" * 60)
    
    # No API key check needed for Qwen!
    
    try:
        config = InterviewConfig(
            mode="role",
            role_or_resume_content="",
            max_questions=10,
            sample_rate=16000,
            audio_device_index=8,
            whisper_model="small",
            gemini_model="gemini-2.0-flash"
        )
        
        print("\n🔧 Initializing components...")
        
        audio_recorder = AudioRecorder(
            sample_rate=config.sample_rate,
            device_index=config.audio_device_index
        )
        
        speech_transcriber = SpeechTranscriber(
            model_name=config.whisper_model
        )
        
        tts_engine = TTSEngine(
            lang='en'
        )
        
        llm_evaluator = LLMEvaluator(
            base_url=LLAMA_BASE_URL,
            model=LLAMA_MODEL
        )
        
        print("✓ All components initialized successfully")
        
        print("\n🔧 Building interview state machine...")
        interview_graph = build_interview_graph(
            audio_recorder=audio_recorder,
            speech_transcriber=speech_transcriber,
            tts_engine=tts_engine,
            llm_evaluator=llm_evaluator,
            max_questions=config.max_questions
        )
        
        print("✓ State machine ready")
        
        initial_state: InterviewState = {
            "mode": "",
            "role_or_resume": "",
            "current_question": "",
            "current_answer": "",
            "conversation_history": [],
            "question_count": 0,
            "needs_followup": False,
            "interview_active": True,
            "consolidated_feedback": ""
        }
        
        print("\n" + "=" * 60)
        print("Starting interview...")
        print("=" * 60)
        
        final_state = interview_graph.invoke(initial_state)
        
        print("\n" + "=" * 60)
        print("Interview session completed!")
        print("=" * 60)
        
        if final_state:
            print(f"\n📊 Interview Summary:")
            print(f"   Mode: {final_state.get('mode', 'N/A')}")
            print(f"   Questions answered: {len(final_state.get('conversation_history', []))}")
            print(f"   Main questions: {final_state.get('question_count', 0)}")
        
        print("\n✓ Thank you for using AI Interview Bot!")
        print("=" * 60)
        
        try:
            tts_engine.stop()
        except:
            pass
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Interview interrupted by user (Ctrl+C)")
        print("Exiting...")
        
    except Exception as e:
        print(f"\n\n❌ An error occurred: {str(e)}")
        print("Please check your configuration and try again.")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()