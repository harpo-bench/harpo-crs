#!/usr/bin/env python3
"""
HARPO: ReDial Dataset Converter

Converts ReDial conversational recommendation dataset to HARPO-MT format.
ReDial has ACTUAL movie recommendations - essential for proper ranking evaluation.

Uses GPT-4o-mini for ALL processing (no heuristics):
- Utterance classification
- VTO assignment  
- Preference extraction
- Rejected response generation

Downloads from GitHub (not HuggingFace).

Usage:
    python convert_redial.py --output ./redial_harpo_data --use-llm --api-key YOUR_KEY
"""

import json
import zipfile
import io
import os
import re
import random
import time
import urllib.request
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from tqdm import tqdm

# All available VTOs
ALL_VTOS = [
    "extract_context", "extract_entities", "retrieve_preferences", 
    "identify_constraints", "search_candidates", "filter_results",
    "rank_options", "compare_options", "explain_choice", "refine_query"
]

# ============================================================================
# LLM PROMPTS
# ============================================================================

CLASSIFY_UTTERANCE_PROMPT = """Classify this utterance in a movie recommendation conversation.

Context: {context}
Current utterance: {utterance}
Speaker: {speaker}
Contains movie mention: {has_movie}

Classify into ONE of these categories:
- greeting: Initial greeting (hi, hello)
- ask_preference: Recommender asking what user wants
- provide_preference: User stating their preferences
- recommend: Recommender suggesting specific movies
- explain: Providing information about movies
- accept: User accepting/liking a recommendation
- reject: User rejecting/disliking a recommendation
- ask_info: User asking for more details
- provide_info: Giving details about a movie
- thank: Thanking
- goodbye: Ending conversation

Return ONLY the category name, nothing else."""

VTO_ASSIGNMENT_PROMPT = """Assign Virtual Tool Operations (VTOs) for this recommendation system response.

Context: {context}
User input: {user_input}
System response: {response}
Utterance type: {utterance_type}

Available VTOs:
- extract_context: Extract situation/occasion from conversation
- extract_entities: Identify movies, genres, actors mentioned
- retrieve_preferences: Get user's stated preferences
- identify_constraints: Find requirements (genre, year, mood)
- search_candidates: Search for matching movies
- filter_results: Apply filters to narrow down
- rank_options: Order movies by relevance
- compare_options: Compare multiple movies
- explain_choice: Explain why recommending
- refine_query: Ask clarifying questions

Return a comma-separated list of 1-4 most relevant VTOs for this response.
Example: search_candidates, rank_options, explain_choice"""

EXTRACT_PREFERENCES_PROMPT = """Extract user preferences from this movie recommendation conversation.

Conversation:
{conversation}

Extract these preferences if mentioned (return JSON):
{{
    "genres": ["list of genres mentioned"],
    "mood": "funny/scary/romantic/exciting/thoughtful/null",
    "actors": ["actors mentioned"],
    "directors": ["directors mentioned"],
    "similar_to": ["movies they liked"],
    "avoid": ["things they don't want"],
    "era": "recent/classic/80s/90s/null"
}}

Return ONLY valid JSON."""

CHARM_PREFERENCE_PROMPT = """Generate a {quality} quality response for preference learning.

Context: {context}
User: {user_input}
Domain: {domain}

If "high": Natural, helpful, relevant recommendations with specific movie names
If "low": Misses context, wrong approach, generic/vague, or recommends wrong movies

Response:"""


# ============================================================================
# LLM CLIENT (matches data_generation.py exactly)
# ============================================================================

class LLMClient:
    """OpenAI client for data generation with parallel processing support"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini", 
                 max_workers: int = 10, rate_limit_per_min: int = 500):
        self.model = model
        self.api_key = api_key
        self._client = None
        self.max_workers = max_workers
        self.rate_limit_per_min = rate_limit_per_min
        self._request_times = []
    
    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                
                api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OpenAI API key required. Set OPENAI_API_KEY or use --api-key")
                
                self._client = OpenAI(api_key=api_key)
            except ImportError:
                raise ImportError("Install openai: pip install openai")
        return self._client
    
    def _rate_limit_wait(self):
        """Simple rate limiting"""
        now = time.time()
        self._request_times = [t for t in self._request_times if now - t < 60]
        
        if len(self._request_times) >= self.rate_limit_per_min:
            sleep_time = 60 - (now - self._request_times[0]) + 0.1
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        self._request_times.append(time.time())
    
    def generate(self, prompt: str, temperature: float = 0.7,
                 max_tokens: int = 1024, json_mode: bool = False) -> str:
        """Generate text from prompt (single request)"""
        self._rate_limit_wait()
        client = self._get_client()
        
        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        
        try:
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM API error: {e}")
            return ""
    
    def generate_batch(self, prompts: List[str], temperature: float = 0.7,
                       max_tokens: int = 256, json_mode: bool = False,
                       show_progress: bool = True) -> List[str]:
        """Generate responses for multiple prompts in parallel."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        results = [""] * len(prompts)
        
        def process_single(idx_prompt):
            idx, prompt = idx_prompt
            try:
                self._rate_limit_wait()
                response = self.generate(prompt, temperature, max_tokens, json_mode)
                return idx, response
            except Exception as e:
                print(f"Error processing prompt {idx}: {e}")
                return idx, ""
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(process_single, (i, p)): i 
                      for i, p in enumerate(prompts)}
            
            iterator = as_completed(futures)
            if show_progress:
                iterator = tqdm(iterator, total=len(prompts), desc="LLM batch processing")
            
            for future in iterator:
                try:
                    idx, response = future.result(timeout=60)
                    results[idx] = response
                except Exception as e:
                    print(f"Future error: {e}")
        
        return results
    
    def generate_batch_with_retry(self, prompts: List[str], 
                                   max_retries: int = 3,
                                   **kwargs) -> List[str]:
        """Generate with automatic retry for failed requests."""
        results = self.generate_batch(prompts, **kwargs)
        
        failed_indices = [i for i, r in enumerate(results) if not r]
        
        for retry in range(max_retries):
            if not failed_indices:
                break
            
            print(f"Retry {retry + 1}: {len(failed_indices)} failed requests")
            failed_prompts = [prompts[i] for i in failed_indices]
            retry_results = self.generate_batch(failed_prompts, show_progress=False, **kwargs)
            
            for i, result in zip(failed_indices, retry_results):
                if result:
                    results[i] = result
            
            failed_indices = [i for i, r in enumerate(results) if not r]
        
        return results


# ============================================================================
# GITHUB DOWNLOAD
# ============================================================================

def download_redial_from_github(output_dir: str = "./redial_raw") -> Tuple[str, str]:
    """Download ReDial dataset zip directly from the data branch"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Target paths
    train_path = os.path.join(output_dir, "train_data.jsonl")
    test_path = os.path.join(output_dir, "test_data.jsonl")
    
    if os.path.exists(train_path) and os.path.exists(test_path):
        print(f"  Files already exist in {output_dir}, skipping download")
        return train_path, test_path

    # CORRECT RAW URL for the specific zip file in the repo
    # using 'raw=true' ensures we get binary data, not the GitHub HTML wrapper
    zip_url = "https://github.com/ReDialData/website/blob/data/redial_dataset.zip?raw=true"
    print(f"  Downloading ReDial dataset from {zip_url}...")
    
    try:
        # 1. Download the zip file into memory
        with urllib.request.urlopen(zip_url) as response:
            zip_content = response.read()
            
        # 2. Extract specific files from the zip
        with zipfile.ZipFile(io.BytesIO(zip_content)) as z:
            print("    Zip content:", z.namelist())
            
            # Helper to extract a file by approximate name (ignores folders)
            def extract_file(target_name, dest_path):
                # Find the file in the zip (ignoring directory structure)
                source_name = next((n for n in z.namelist() if n.endswith(target_name)), None)
                if not source_name:
                    raise ValueError(f"Could not find {target_name} inside the downloaded zip.")
                
                print(f"    Extracting {source_name} -> {dest_path}...")
                with z.open(source_name) as source, open(dest_path, "wb") as target:
                    target.write(source.read())

            extract_file("train_data.jsonl", train_path)
            extract_file("test_data.jsonl", test_path)

        print(f"    ✓ Successfully saved to {output_dir}")
        
    except Exception as e:
        print(f"    ✗ Failed to download/extract: {e}")
        raise
    
    return train_path, test_path


def load_redial_from_github(raw_dir: str = "./redial_raw") -> Tuple[List[Dict], List[Dict]]:
    """Load ReDial dataset from GitHub downloads"""
    print("Loading ReDial from GitHub...")
    
    train_path, test_path = download_redial_from_github(raw_dir)
    
    train_data = []
    with open(train_path, 'r') as f:
        for line in f:
            if line.strip():
                train_data.append(json.loads(line))
    
    test_data = []
    with open(test_path, 'r') as f:
        for line in f:
            if line.strip():
                test_data.append(json.loads(line))
    
    print(f"  Loaded {len(train_data)} train, {len(test_data)} test conversations")
    return train_data, test_data


# ============================================================================
# LLM-BASED CLASSIFICATION (replaces all heuristics)
# ============================================================================

def classify_utterances_batch(llm: LLMClient, utterances: List[Dict]) -> List[str]:
    """Batch classify utterances using LLM"""
    prompts = []
    for u in utterances:
        speaker = "Recommender" if u["is_recommender"] else "User"
        prompt = CLASSIFY_UTTERANCE_PROMPT.format(
            context=u["context"][-500:] if u["context"] else "Start of conversation",
            utterance=u["text"][:200],
            speaker=speaker,
            has_movie="Yes" if u["has_movie"] else "No"
        )
        prompts.append(prompt)
    
    print(f"  Classifying {len(prompts)} utterances...")
    responses = llm.generate_batch_with_retry(prompts, temperature=0.3, max_tokens=50)
    
    valid_types = ["greeting", "ask_preference", "provide_preference", "recommend",
                   "explain", "accept", "reject", "ask_info", "provide_info", 
                   "thank", "goodbye"]
    
    results = []
    for i, response in enumerate(responses):
        response_lower = response.lower().strip()
        found = False
        for vtype in valid_types:
            if vtype in response_lower:
                results.append(vtype)
                found = True
                break
        if not found:
            results.append("provide_preference" if not utterances[i]["is_recommender"] else "explain")
    
    return results


def assign_vtos_batch(llm: LLMClient, items: List[Dict]) -> List[List[str]]:
    """Batch assign VTOs using LLM"""
    prompts = []
    for item in items:
        prompt = VTO_ASSIGNMENT_PROMPT.format(
            context=item["context"][-500:] if item["context"] else "",
            user_input=item["user_input"][:200],
            response=item["response"][:300],
            utterance_type=item["utterance_type"]
        )
        prompts.append(prompt)
    
    print(f"  Assigning VTOs for {len(prompts)} responses...")
    responses = llm.generate_batch_with_retry(prompts, temperature=0.3, max_tokens=100)
    
    results = []
    for response in responses:
        vtos = []
        for vto in ALL_VTOS:
            if vto in response.lower():
                vtos.append(vto)
        if not vtos:
            vtos = ["extract_context"]
        results.append(vtos[:4])
    
    return results


# ============================================================================
# HEURISTIC FALLBACKS (when LLM not available)
# ============================================================================

def classify_utterance_heuristic(text: str, is_recommender: bool, has_movie: bool) -> str:
    """Fallback heuristic classification"""
    text_lower = text.lower()
    
    if any(w in text_lower for w in ["hi", "hello", "hey"]) and len(text) < 50:
        return "greeting"
    if any(w in text_lower for w in ["thank", "thanks"]):
        return "thank"
    if any(w in text_lower for w in ["bye", "goodbye", "see you"]):
        return "goodbye"
    
    if is_recommender:
        if has_movie:
            if any(w in text_lower for w in ["recommend", "suggest", "try", "watch", "check out", "should"]):
                return "recommend"
            return "provide_info"
        return "ask_preference" if "?" in text else "explain"
    else:
        if has_movie:
            if any(w in text_lower for w in ["love", "like", "enjoy", "great", "good"]):
                return "accept"
            elif any(w in text_lower for w in ["don't", "not", "didn't", "hate"]):
                return "reject"
            return "provide_preference"
        return "ask_info" if "?" in text else "provide_preference"


VTO_MAPPING_FALLBACK = {
    "greeting": ["extract_context"],
    "ask_preference": ["extract_context", "refine_query"],
    "provide_preference": ["extract_entities", "identify_constraints", "retrieve_preferences"],
    "recommend": ["search_candidates", "rank_options", "explain_choice"],
    "explain": ["compare_options", "explain_choice"],
    "accept": ["filter_results"],
    "reject": ["refine_query", "search_candidates"],
    "ask_info": ["extract_entities", "compare_options"],
    "provide_info": ["explain_choice"],
    "thank": [],
    "goodbye": [],
}

def assign_vtos_heuristic(utterance_type: str) -> List[str]:
    """Fallback heuristic VTO assignment"""
    return VTO_MAPPING_FALLBACK.get(utterance_type, ["extract_context"])


# ============================================================================
# MOVIE EXTRACTION
# ============================================================================

def extract_movie_mentions(text: str, movie_map: Dict[str, str]) -> List[str]:
    """Extract movie names from text using @movieId pattern"""
    movie_ids = re.findall(r'@(\d+)', text)
    movies = []
    for mid in movie_ids:
        if mid in movie_map:
            movies.append(movie_map[mid])
    return movies


YEAR_RE = re.compile(r"^(?P<title>.+?)\s*\((?P<year>(?:19|20)\d{2})\)\s*$")


def item_metadata_for(title: str) -> Dict[str, str]:
    """Structured metadata for an item title.

    ReDial encodes the release year in the title itself ("Inception (2010)").
    Pulling it out gives the item tower a real field and lets two films sharing
    a title be told apart -- otherwise near-duplicates become each other's
    negatives.
    """
    if not title:
        return {}
    m = YEAR_RE.match(title.strip())
    return {"year": m.group("year")} if m else {}


def build_movie_database(train_data: List[Dict], test_data: List[Dict]) -> Dict[str, str]:
    """Build movie ID to name mapping from all conversations"""
    movie_db = {}
    
    for conv in train_data + test_data:
        mentions = conv.get("movieMentions", {})
        if isinstance(mentions, dict):
            for mid, mname in mentions.items():
                movie_db[str(mid)] = mname
        elif isinstance(mentions, list):
            for item in mentions:
                if isinstance(item, dict):
                    mid = str(item.get("movieId", ""))
                    mname = item.get("movieName", "")
                    if mid and mname:
                        movie_db[mid] = mname
    
    print(f"  Built movie database with {len(movie_db)} movies")
    return movie_db


# ============================================================================
# TOOL CALL GENERATION
# ============================================================================

def create_tool_call(utterance_type: str, movies: List[str], preferences: Dict) -> Optional[Dict]:
    """Create tool call based on utterance type"""
    if utterance_type == "recommend" and movies:
        return {
            "tool": "recommend",
            "args": {
                "movies": movies[:3],
                "criteria": preferences.get("genres", ["rating"])[0] if preferences.get("genres") else "rating"
            }
        }
    elif utterance_type == "provide_preference":
        genre = preferences.get("genres", ["movie"])[0] if preferences.get("genres") else "movie"
        return {
            "tool": "search",
            "args": {
                "query": genre,
                "category": "movies"
            }
        }
    elif utterance_type in ["explain", "provide_info"] and movies:
        return {
            "tool": "get_info",
            "args": {
                "movie": movies[0],
                "fields": ["plot", "cast", "rating"]
            }
        }
    return None


# ============================================================================
# MAIN CONVERSION
# ============================================================================

def extract_utterances_from_conversation(conv: Dict, movie_db: Dict[str, str]) -> List[Dict]:
    """Extract all utterances from a conversation with metadata"""
    utterance_infos = []
    
    movie_mentions = {}
    if isinstance(conv.get("movieMentions"), dict):
        movie_mentions = {str(k): v for k, v in conv["movieMentions"].items()}
    
    all_movies = {**movie_db, **movie_mentions}
    
    messages = conv.get("messages", [])
    if not messages:
        return []
    
    initiator_id = conv.get("initiatorWorkerId", 0)
    
    history = []
    conversation_movies = []
    
    for i, msg in enumerate(messages):
        if isinstance(msg, dict):
            text = msg.get("text", "")
            sender_id = msg.get("senderWorkerId", 0)
        elif isinstance(msg, str):
            text = msg
            sender_id = initiator_id if i % 2 == 0 else initiator_id + 1
        else:
            continue
        
        if not text.strip():
            continue
        
        is_recommender = (sender_id != initiator_id)
        
        # Replace @movieId with actual movie names
        movies_in_msg = []
        for mid, mname in all_movies.items():
            if f"@{mid}" in text:
                text = text.replace(f"@{mid}", f'"{mname}"')
                movies_in_msg.append(mname)
                conversation_movies.append(mname)
        
        context_str = "\n".join(history[-6:])
        utterance_infos.append({
            "text": text,
            "context": context_str,
            "is_recommender": is_recommender,
            "has_movie": bool(movies_in_msg),
            "movies_in_msg": movies_in_msg,
            "conversation_movies": list(set(conversation_movies)),
            "history": list(history),
            "conv_id": str(conv.get("conversationId", "")),
            "turn_idx": len(history)
        })
        
        role = "User" if not is_recommender else "Assistant"
        history.append(f"{role}: {text}")
    
    return utterance_infos


def process_utterances_with_llm(utterance_infos: List[Dict], llm: LLMClient) -> List[Dict]:
    """Process all utterances with LLM for classification and VTO assignment"""
    
    # Step 1: Classify all utterances
    print("\n  Step 1/2: Classifying utterances with LLM...")
    classifications = classify_utterances_batch(llm, utterance_infos)
    
    for info, utype in zip(utterance_infos, classifications):
        info["utterance_type"] = utype
    
    # Step 2: Assign VTOs for recommender responses
    recommender_infos = [info for info in utterance_infos if info["is_recommender"] and info["history"]]
    
    vto_items = []
    for info in recommender_infos:
        user_input = ""
        for h in reversed(info["history"]):
            if h.startswith("User:"):
                user_input = h[5:].strip()
                break
        
        vto_items.append({
            "context": info["context"],
            "user_input": user_input,
            "response": info["text"],
            "utterance_type": info["utterance_type"]
        })
    
    if vto_items:
        print("\n  Step 2/2: Assigning VTOs with LLM...")
        vto_results = assign_vtos_batch(llm, vto_items)
        
        for info, vtos in zip(recommender_infos, vto_results):
            info["vtos"] = vtos
    
    return utterance_infos


def process_utterances_heuristic(utterance_infos: List[Dict]) -> List[Dict]:
    """Process all utterances with heuristics (fallback)"""
    
    for info in tqdm(utterance_infos, desc="Processing utterances (heuristic)"):
        info["utterance_type"] = classify_utterance_heuristic(
            info["text"], info["is_recommender"], info["has_movie"]
        )
        
        if info["is_recommender"]:
            info["vtos"] = assign_vtos_heuristic(info["utterance_type"])
    
    return utterance_infos


def create_sft_examples(utterance_infos: List[Dict]) -> List[Dict]:
    """Create SFT examples from processed utterance infos"""
    sft_examples = []
    
    for info in utterance_infos:
        if not info["is_recommender"] or not info["history"]:
            continue
        
        input_text = "<|domain:movies|>\n\n"
        input_text += "\n".join(info["history"])
        input_text = input_text.strip()
        
        vtos = info.get("vtos", ["extract_context"])
        vto_str = ", ".join(vtos)
        output_parts = [f"<|think|>{vto_str}<|/think|>"]
        
        tool_call = create_tool_call(info["utterance_type"], info["movies_in_msg"], {})
        if tool_call:
            output_parts.append(f"<|tool_start|>[{json.dumps(tool_call)}]<|tool_end|>")
        
        output_parts.append(info["text"])
        output_text = "\n".join(output_parts)
        
        sft_examples.append({
            "input": input_text,
            "output": output_text,
            "conversation_id": info["conv_id"],
            "turn_id": info["turn_idx"],
            "domain": "movies",
            "vtos": vtos,
            "tools": [tool_call["tool"]] if tool_call else [],
            "movies_mentioned": info["movies_in_msg"],
            "all_conversation_movies": info["conversation_movies"],
            "ground_truth_item": info["movies_in_msg"][0] if info["movies_in_msg"] else None,
            "item_metadata": item_metadata_for(
                info["movies_in_msg"][0] if info["movies_in_msg"] else ""),
            "satisfaction_score": 4.0 if info["movies_in_msg"] else 3.5
        })
    
    return sft_examples


def generate_preference_pairs(sft_data: List[Dict], all_movies: List[str], 
                              llm: Optional[LLMClient] = None) -> List[Dict]:
    """Generate preference pairs for CHARM training using LLM"""
    preference_pairs = []
    
    examples_with_gt = [ex for ex in sft_data if ex.get("ground_truth_item")]
    print(f"  Examples with ground truth items: {len(examples_with_gt)}")
    
    if not examples_with_gt:
        return []
    
    if llm:
        print("  Generating rejected responses with LLM...")
        
        prompts = []
        for ex in examples_with_gt:
            context = ex["input"][-500:]
            user_input_match = re.search(r'User:\s*([^\n]+)(?:\n|$)', context)
            user_input = user_input_match.group(1) if user_input_match else "recommendation"
            
            prompt = CHARM_PREFERENCE_PROMPT.format(
                quality="low",
                context=context,
                user_input=user_input,
                domain="movies"
            )
            prompts.append(prompt)
        
        rejected_responses = llm.generate_batch_with_retry(
            prompts, temperature=0.9, max_tokens=256, show_progress=True
        )
        
        for ex, rejected in zip(examples_with_gt, rejected_responses):
            chosen = ex["output"]
            gt_movie = ex["ground_truth_item"]
            
            if rejected and len(rejected) > 20:
                rejected_text = rejected
            else:
                rejected_movies = [m for m in all_movies if m != gt_movie]
                if rejected_movies:
                    wrong_movie = random.choice(rejected_movies)
                    rejected_text = chosen.replace(f'"{gt_movie}"', f'"{wrong_movie}"')
                    if rejected_text == chosen:
                        rejected_text = "I'm not sure what to recommend. Let me think about it."
                else:
                    rejected_text = "I'm not sure what to recommend."
            
            preference_pairs.append({
                "conversation_id": ex["conversation_id"],
                "context": ex["input"],
                "chosen": chosen,
                "rejected": rejected_text,
                "chosen_vtos": ex["vtos"],
                "rejected_vtos": ["search_candidates"],
                "reward_margin": 0.3 + random.random() * 0.4,
                "ground_truth_movie": gt_movie,
                "hierarchical_rewards": {
                    "relevance": {"chosen": 0.9, "rejected": 0.3},
                    "diversity": {"chosen": 0.8, "rejected": 0.5},
                    "user_satisfaction": {"chosen": 0.85, "rejected": 0.4},
                    "engagement": {"chosen": 0.9, "rejected": 0.6}
                }
            })
    else:
        print("  Generating rejected responses (heuristic)...")
        
        generic_bad = [
            "I can help you with that. Let me know if you need more information.",
            "That's a good choice. Would you like to see more options?",
            "Have you considered trying a different category entirely?",
            "I'm not sure about that. Maybe try something else.",
            "There are many options available. It's hard to recommend one.",
        ]
        
        for ex in tqdm(examples_with_gt, desc="Creating preference pairs"):
            chosen = ex["output"]
            gt_movie = ex["ground_truth_item"]
            
            if random.random() < 0.5:
                rejected_movies = [m for m in all_movies if m != gt_movie]
                if rejected_movies:
                    wrong_movie = random.choice(rejected_movies)
                    rejected = chosen.replace(f'"{gt_movie}"', f'"{wrong_movie}"')
                    if rejected == chosen:
                        rejected = random.choice(generic_bad)
                else:
                    rejected = random.choice(generic_bad)
            else:
                rejected = random.choice(generic_bad)
            
            preference_pairs.append({
                "conversation_id": ex["conversation_id"],
                "context": ex["input"],
                "chosen": chosen,
                "rejected": rejected,
                "chosen_vtos": ex["vtos"],
                "rejected_vtos": ["search_candidates"],
                "reward_margin": 0.3 + random.random() * 0.4,
                "ground_truth_movie": gt_movie,
                "hierarchical_rewards": {
                    "relevance": {"chosen": 0.9, "rejected": 0.3},
                    "diversity": {"chosen": 0.8, "rejected": 0.5},
                    "user_satisfaction": {"chosen": 0.85, "rejected": 0.4},
                    "engagement": {"chosen": 0.9, "rejected": 0.6}
                }
            })
    
    return preference_pairs


def convert_redial_to_harpo(output_dir: str, max_train: int = None, max_test: int = None,
                           llm: Optional[LLMClient] = None):
    """Main conversion function"""
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("ReDial to HARPO-MT Conversion")
    print("=" * 60)
    print(f"Using LLM: {llm is not None}")
    print(f"Output: {output_dir}")
    print("")
    
    # Load data from GitHub
    train_raw, test_raw = load_redial_from_github()
    
    if max_train:
        train_raw = train_raw[:max_train]
    if max_test:
        test_raw = test_raw[:max_test]
    
    # Build movie database
    print("\nBuilding movie database...")
    movie_db = build_movie_database(train_raw, test_raw)
    
    # Process training data
    print("\nProcessing training conversations...")
    all_train_utterances = []
    for conv in tqdm(train_raw, desc="Extracting utterances"):
        utterances = extract_utterances_from_conversation(conv, movie_db)
        all_train_utterances.extend(utterances)
    
    print(f"  Total utterances: {len(all_train_utterances)}")
    
    # Classify and assign VTOs
    if llm:
        all_train_utterances = process_utterances_with_llm(all_train_utterances, llm)
    else:
        all_train_utterances = process_utterances_heuristic(all_train_utterances)
    
    # Create SFT examples
    print("\nCreating SFT examples...")
    train_sft = create_sft_examples(all_train_utterances)
    print(f"  Training examples: {len(train_sft)}")
    
    # Process test data
    print("\nProcessing test conversations...")
    all_test_utterances = []
    for conv in tqdm(test_raw, desc="Extracting test utterances"):
        utterances = extract_utterances_from_conversation(conv, movie_db)
        all_test_utterances.extend(utterances)
    
    if llm:
        all_test_utterances = process_utterances_with_llm(all_test_utterances, llm)
    else:
        all_test_utterances = process_utterances_heuristic(all_test_utterances)
    
    test_sft = create_sft_examples(all_test_utterances)
    print(f"  Test examples: {len(test_sft)}")
    
    # Collect all movies
    all_movies = list(set(movie_db.values()))
    print(f"\nTotal unique movies: {len(all_movies)}")
    
    # Generate preference pairs
    print("\nGenerating preference pairs...")
    train_pref = generate_preference_pairs(train_sft, all_movies, llm)
    print(f"  Preference pairs: {len(train_pref)}")
    
    # Save files
    print("\nSaving files...")
    
    with open(os.path.join(output_dir, "sft_data.json"), "w") as f:
        json.dump(train_sft, f, indent=2)
    
    with open(os.path.join(output_dir, "test_sft.json"), "w") as f:
        json.dump(test_sft, f, indent=2)
    
    with open(os.path.join(output_dir, "preference_data.json"), "w") as f:
        json.dump(train_pref, f, indent=2)
    
    with open(os.path.join(output_dir, "movie_list.json"), "w") as f:
        json.dump(all_movies, f, indent=2)
    
    stats = {
        "num_train_sft": len(train_sft),
        "num_test_sft": len(test_sft),
        "num_preference_pairs": len(train_pref),
        "num_movies": len(all_movies),
        "examples_with_movies": sum(1 for ex in train_sft if ex.get("ground_truth_item")),
        "used_llm": llm is not None
    }
    
    with open(os.path.join(output_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Conversion Complete!")
    print("=" * 60)
    print(f"Training examples: {stats['num_train_sft']}")
    print(f"Test examples: {stats['num_test_sft']}")
    print(f"Preference pairs: {stats['num_preference_pairs']}")
    print(f"Total movies: {stats['num_movies']}")
    print(f"Examples with recommendations: {stats['examples_with_movies']}")
    print(f"Used LLM: {stats['used_llm']}")
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert ReDial to HARPO-MT format")
    parser.add_argument("--output", default="./redial_harpo_data", help="Output directory")
    parser.add_argument("--max-train", type=int, default=None, help="Max training conversations")
    parser.add_argument("--max-test", type=int, default=None, help="Max test conversations")
    parser.add_argument("--use-llm", action="store_true", 
                        help="Use GPT-4o-mini for classification, VTO assignment, and preference pairs")
    parser.add_argument("--api-key", type=str, default=None,
                        help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="OpenAI model to use (default: gpt-4o-mini)")
    
    args = parser.parse_args()
    
    llm = None
    if args.use_llm:
        print("Initializing LLM client...")
        try:
            llm = LLMClient(api_key=args.api_key, model=args.model)
            llm._get_client()
            print(f"  ✓ Connected to OpenAI ({args.model})")
        except Exception as e:
            print(f"  ✗ Failed to initialize LLM: {e}")
            print("  Falling back to heuristic processing")
            llm = None
    
    convert_redial_to_harpo(args.output, args.max_train, args.max_test, llm)
