"""
captioner.py

Vision-Language Model (VLM) wrapper for generating rich scene captions.
Supports multiple backends: OpenAI GPT-4o, Google Gemini, and local models.
"""

import json
import base64
from io import BytesIO
from pathlib import Path
from PIL import Image
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import torch


@dataclass
class CaptionResult:
    """Structured caption result from VLM."""
    visual_description: str
    subjects: str
    action: str
    narrative_context: str
    emotional_tone: str
    thematic_connection: str
    search_phrases: List[str]
    raw_response: str  # Original model response
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "visual_description": self.visual_description,
            "subjects": self.subjects,
            "action": self.action,
            "narrative_context": self.narrative_context,
            "emotional_tone": self.emotional_tone,
            "thematic_connection": self.thematic_connection,
            "search_phrases": self.search_phrases
        }
    
    def to_embedding_text(self, movie_info: Optional[Dict] = None) -> str:
        """
        Convert caption to text suitable for embedding.
        
        This creates a rich text representation that will be embedded
        by the text encoder for semantic search.
        
        Args:
            movie_info: Optional movie metadata to include.
            
        Returns:
            Formatted text for embedding.
        """
        parts = []
        
        # Add movie context if available
        if movie_info:
            if movie_info.get("title"):
                parts.append(f"Movie: {movie_info['title']}")
            if movie_info.get("genre"):
                genres = movie_info["genre"]
                if isinstance(genres, list):
                    genres = ", ".join(genres)
                parts.append(f"Genre: {genres}")
        
        # Add caption components
        parts.extend([
            f"Visual: {self.visual_description}",
            f"Subjects: {self.subjects}",
            f"Action: {self.action}",
            f"Narrative: {self.narrative_context}",
            f"Emotion: {self.emotional_tone}",
            f"Themes: {self.thematic_connection}",
        ])
        
        # Add search phrases for better recall
        if self.search_phrases:
            parts.append(f"Related: {', '.join(self.search_phrases)}")
        
        return " ".join(parts)


class VLMCaptioner:
    """
    Vision-Language Model captioner for generating rich scene descriptions.
    
    This class generates structured captions that capture:
    - What visually appears (objects, setting, style)
    - What is happening (actions, events)
    - Why it matters (narrative context, themes, emotions)
    
    The plot context is crucial for understanding WHY things happen,
    which enables queries like "wave destroying city" to match 2012
    but not Interstellar.
    """
    
    # Prompt template for caption generation
    CAPTION_PROMPT_TEMPLATE = """Analyze this scene from the movie "{title}".

MOVIE CONTEXT:
- Title: {title} ({year})
- Genre: {genre}
- Plot Summary: {plot_summary}
- Key Themes: {themes}

SCENE AUDIO/DIALOGUE:
{transcript}

Based on the provided visual content from this scene, generate a detailed structured analysis. 

IMPORTANT: Your analysis should connect what you SEE in the frames with the MOVIE CONTEXT provided. This helps understand not just what appears, but WHY it's happening in the story.

Respond with a JSON object containing these fields:

{{
    "visual_description": "Describe the visual elements: setting, lighting, color palette, cinematography style, and any notable visual details. Be specific about what you see.",
    
    "subjects": "Identify who/what is in the scene. Describe characters (appearance, state, position) and significant objects.",
    
    "action": "Describe what is physically happening in this scene. What actions are being performed? What events are unfolding?",
    
    "narrative_context": "Based on the plot summary, explain what part of the story this scene likely represents. Why is this scene happening? What is its purpose in the narrative?",
    
    "emotional_tone": "What emotions does this scene convey? What should the audience feel? Consider both the visual mood and the narrative weight.",
    
    "thematic_connection": "How does this scene connect to the movie's themes of {themes}? What does it represent or symbolize?",
    
    "search_phrases": [
        "List 10-15 different ways someone might describe this scene when searching for it.",
        "Include both literal descriptions (what appears) and contextual descriptions (what happens/means).",
        "Vary the specificity: some general, some specific.",
        "Include emotional and thematic descriptions too."
    ]
}}

Respond ONLY with the JSON object, no additional text."""

    def __init__(
        self,
        provider: str = "openai",
        model_name: str = "gpt-4o",
        max_tokens: int = 1500,
        api_key: Optional[str] = None
    ):
        """
        Initialize VLM captioner.
        
        Args:
            provider: API provider - "openai", "google", or "local".
            model_name: Model identifier.
                OpenAI: "gpt-4o", "gpt-4-turbo"
                Google: "gemini-1.5-pro", "gemini-1.5-flash"
                Local: "llava-v1.6-34b", "Qwen/Qwen2-VL-7B-Instruct"
            max_tokens: Maximum tokens for response.
            api_key: API key (reads from env if not provided).
        """
        self.provider = provider
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.api_key = api_key
        
        self._setup_client()
    
    def _setup_client(self):
        """Set up the appropriate API client."""
        if self.provider == "openai":
            self._setup_openai()
        elif self.provider == "google":
            self._setup_google()
        elif self.provider == "local":
            self._setup_local()
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def _setup_openai(self):
        """Set up OpenAI client."""
        try:
            from openai import OpenAI
            import os
            
            api_key = self.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY env var.")
            
            self.client = OpenAI(api_key=api_key)
            
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
    
    def _setup_google(self):
        """Set up Google Generative AI client."""
        try:
            import google.generativeai as genai
            import os
            
            api_key = self.api_key or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("Google API key not found. Set GOOGLE_API_KEY env var.")
            
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(self.model_name)
            
        except ImportError:
            raise ImportError("Please install google-generativeai: pip install google-generativeai")
    
    def _setup_local(self):
        """Set up a local Qwen2.5-VL model via Transformers."""
        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            from qwen_vl_utils import process_vision_info

            print(f"Loading local VLM: {self.model_name}")

            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self._process_vision_info = process_vision_info
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map="auto",
            )
            self.model.eval()

            print("Local VLM loaded successfully.")

        except ImportError as e:
            raise ImportError(
                "Please install/upgrade transformers, accelerate, and qwen-vl-utils for local Qwen2.5-VL support."
            ) from e
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    def _build_prompt(
        self,
        movie_info: Dict,
        transcript: Optional[str]
    ) -> str:
        """
        Build the prompt for caption generation.
        
        Args:
            movie_info: Movie metadata from plot_summaries.json
            transcript: Transcribed dialogue/narration
            
        Returns:
            Formatted prompt string.
        """
        # Extract movie info with defaults
        title = movie_info.get("title", "Unknown")
        year = movie_info.get("year", "Unknown")
        
        genre = movie_info.get("genre", ["Unknown"])
        if isinstance(genre, list):
            genre = ", ".join(genre)
        
        themes = movie_info.get("themes", ["Unknown"])
        if isinstance(themes, list):
            themes = ", ".join(themes)
        
        plot_summary = movie_info.get("plot_summary", "Plot information not available.")
        
        # Handle transcript
        if transcript:
            transcript_text = f'"{transcript}"'
        else:
            transcript_text = "No dialogue or narration detected in this scene."
        
        # Build prompt
        prompt = self.CAPTION_PROMPT_TEMPLATE.format(
            title=title,
            year=year,
            genre=genre,
            themes=themes,
            plot_summary=plot_summary,
            transcript=transcript_text
        )
        
        return prompt
    
    def _parse_response(self, response_text: str) -> CaptionResult:
        """
        Parse VLM response into structured CaptionResult.
        
        Args:
            response_text: Raw response from VLM.
            
        Returns:
            Parsed CaptionResult.
        """
        # Try to extract JSON from response
        try:
            # Handle potential markdown code blocks
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                json_str = response_text[start:end].strip()
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                json_str = response_text[start:end].strip()
            else:
                json_str = response_text.strip()
            
            data = json.loads(json_str)
            
            return CaptionResult(
                visual_description=data.get("visual_description", ""),
                subjects=data.get("subjects", ""),
                action=data.get("action", ""),
                narrative_context=data.get("narrative_context", ""),
                emotional_tone=data.get("emotional_tone", ""),
                thematic_connection=data.get("thematic_connection", ""),
                search_phrases=data.get("search_phrases", []),
                raw_response=response_text
            )
            
        except json.JSONDecodeError:
            # If JSON parsing fails, create a basic caption from the raw text
            print(f"Warning: Failed to parse JSON response, using raw text")
            return CaptionResult(
                visual_description=response_text,
                subjects="",
                action="",
                narrative_context="",
                emotional_tone="",
                thematic_connection="",
                search_phrases=[],
                raw_response=response_text
            )
    
    def generate_caption(
        self,
        frames: List[Image.Image],
        movie_info: Dict,
        transcript: Optional[str] = None,
        video_path: Optional[str] = None,
    ) -> CaptionResult:
        """
        Generate a rich, structured caption for a scene.
        
        This is the main method for caption generation. It sends frames
        and context to the VLM and parses the structured response.
        
        Args:
            frames: List of PIL Images from the scene.
            movie_info: Movie metadata dict with title, plot, themes, etc.
            transcript: Optional transcribed dialogue.
            video_path: Optional local video path for models that support native video input.
            
        Returns:
            CaptionResult with structured caption.
        """
        # Build prompt
        prompt = self._build_prompt(movie_info, transcript)
        
        # Call appropriate API
        if self.provider == "openai":
            response = self._call_openai(frames, prompt)
        elif self.provider == "google":
            response = self._call_google(frames, prompt)
        elif self.provider == "local":
            response = self._call_local(frames, prompt, video_path=video_path)
        else:
            raise ValueError(f"Provider {self.provider} not implemented")
        
        # Parse response
        return self._parse_response(response)
    
    def _call_openai(
        self, 
        frames: List[Image.Image], 
        prompt: str
    ) -> str:
        """Call OpenAI API with images."""
        # Prepare image content
        image_content = []
        for frame in frames:
            base64_image = self._image_to_base64(frame)
            image_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "low"  # Use "high" for better quality but more tokens
                }
            })
        
        # Build messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    *image_content
                ]
            }
        ]
        
        # Call API
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=0.3  # Lower temperature for more consistent output
        )
        
        return response.choices[0].message.content
    
    def _call_google(
        self, 
        frames: List[Image.Image], 
        prompt: str
    ) -> str:
        """Call Google Generative AI API with images."""
        # Prepare content
        content = [prompt]
        for frame in frames:
            content.append(frame)
        
        # Call API
        response = self.client.generate_content(
            content,
            generation_config={
                "max_output_tokens": self.max_tokens,
                "temperature": 0.3
            }
        )
        
        return response.text

    def _call_local(
        self,
        frames: List[Image.Image],
        prompt: str,
        video_path: Optional[str] = None,
    ) -> str:
        """Call a local Qwen2.5-VL model using video mode when available."""
        content = []
        if video_path:
            video_uri = Path(video_path).expanduser().resolve().as_uri()
            content.append({
                "type": "video",
                "video": video_uri,
                "fps": 1.0,
            })
        else:
            if not frames:
                raise ValueError("At least one frame is required for local captioning.")
            content.extend(
                {"type": "image", "image": frame.convert("RGB")}
                for frame in frames
            )
        content.append({"type": "text", "text": prompt})

        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        try:
            image_inputs, video_inputs, video_kwargs = self._process_vision_info(
                messages,
                return_video_kwargs=True,
            )
        except TypeError:
            image_inputs, video_inputs = self._process_vision_info(messages)
            video_kwargs = {}

        # Some qwen-vl-utils / transformers combinations return fps as a
        # single-item list (or an empty list) for single-video inference,
        # while the processor expects a scalar.
        fps_value = video_kwargs.get("fps")
        if isinstance(fps_value, list):
            if len(fps_value) == 0:
                video_kwargs.pop("fps", None)
            elif len(fps_value) == 1:
                video_kwargs["fps"] = fps_value[0]

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = inputs.to(next(self.model.parameters()).device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0]
    
    def __repr__(self) -> str:
        return f"VLMCaptioner(provider={self.provider}, model={self.model_name})"
