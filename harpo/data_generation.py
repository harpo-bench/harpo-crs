"""
HARPO

Handles:
1. Loading and converting your conversation data
2. VTO annotation (heuristic + LLM-based)
3. Preference pair generation for CHARM
4. Cross-domain data preparation

Focus: Generate data that teaches RECOMMENDATION QUALITY
"""

import json
import random
import re
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import asdict
from tqdm import tqdm

from .config import (
    VTO, VTO_DESCRIPTIONS, Domain, DOMAIN_CONFIGS,
    Conversation, ConversationTurn, ToolCall, PreferencePair,
    SPECIAL_TOKENS, VTO_ANNOTATION_PROMPT, CHARM_PREFERENCE_PROMPT
)


# ============================================================================
# LLM CLIENT
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
                import os as _os
                
                api_key = self.api_key or _os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OpenAI API key required")
                
                self._client = OpenAI(api_key=api_key)
            except ImportError:
                raise ImportError("Install openai: pip install openai")
        return self._client
    
    def _rate_limit_wait(self):
        """Simple rate limiting"""
        import time
        now = time.time()
        # Remove requests older than 60 seconds
        self._request_times = [t for t in self._request_times if now - t < 60]
        
        if len(self._request_times) >= self.rate_limit_per_min:
            # Wait until oldest request is 60+ seconds old
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
        """
        Generate responses for multiple prompts in parallel.
        
        Uses ThreadPoolExecutor for concurrent API calls.
        Includes rate limiting and error handling.
        
        Args:
            prompts: List of prompts to process
            temperature: Sampling temperature
            max_tokens: Max tokens per response
            json_mode: Whether to request JSON output
            show_progress: Show tqdm progress bar
            
        Returns:
            List of responses (empty string for failures)
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import time
        
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
        
        # Process in parallel with ThreadPoolExecutor
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
        """
        Generate with automatic retry for failed requests.
        """
        results = self.generate_batch(prompts, **kwargs)
        
        # Find failed indices
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
# VTO ANNOTATOR
# ============================================================================

class VTOAnnotator:
    """Annotate conversations with VTOs (heuristic + optional LLM)"""
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client
        self.vto_desc_str = "\n".join([
            f"- {vto.value}: {desc}"
            for vto, desc in VTO_DESCRIPTIONS.items()
        ])
    
    def annotate_turn(self, context: str, user_input: str,
                      system_response: str, use_llm: bool = False) -> List[VTO]:
        """Annotate a single turn with VTOs"""
        if use_llm and self.llm:
            return self._llm_annotate(context, user_input, system_response)
        return self._heuristic_annotate(user_input, system_response)
    
    def _heuristic_annotate(self, user_input: str, system_response: str) -> List[VTO]:
        """Fast heuristic-based VTO annotation"""
        vtos = []
        user_lower = user_input.lower()
        response_lower = system_response.lower()
        
        # Context/Entity extraction
        context_words = ["looking", "need", "want", "find", "search", "show"]
        if any(w in user_lower for w in context_words):
            vtos.append(VTO.EXTRACT_CONTEXT)
            vtos.append(VTO.EXTRACT_ENTITIES)
        
        # Sentiment
        sentiment_words = ["feel", "nervous", "excited", "worried", "happy", "love", "hate"]
        if any(w in user_lower for w in sentiment_words):
            vtos.append(VTO.ANALYZE_SENTIMENT)
        
        # Preferences
        pref_words = ["like", "prefer", "want", "love", "favorite"]
        if any(w in user_lower for w in pref_words):
            vtos.append(VTO.RETRIEVE_PREFERENCES)
        
        # Constraints
        constraint_words = ["budget", "price", "under", "max", "size", "color", "not", "without"]
        if any(w in user_lower for w in constraint_words):
            vtos.append(VTO.IDENTIFY_CONSTRAINTS)
        
        # Search/Recommendation
        if any(w in response_lower for w in ["recommend", "suggest", "try", "check out", "consider"]):
            vtos.append(VTO.SEARCH_CANDIDATES)
            vtos.append(VTO.RANK_OPTIONS)
        
        # Comparison
        if any(w in user_lower for w in ["compare", "difference", "versus", "vs", "or"]):
            vtos.append(VTO.COMPARE_OPTIONS)
        
        # Explanation
        if any(w in response_lower for w in ["because", "since", "this would", "perfect for", "great for"]):
            vtos.append(VTO.EXPLAIN_CHOICE)
        
        # Clarification
        if "?" in system_response:
            vtos.append(VTO.REFINE_QUERY)
        
        # Filter/Match
        if any(w in response_lower for w in ["filter", "narrow", "match", "suitable"]):
            vtos.append(VTO.FILTER_RESULTS)
            vtos.append(VTO.MATCH_ATTRIBUTES)
        
        # Defaults
        if not vtos:
            vtos = [VTO.EXTRACT_CONTEXT, VTO.SEARCH_CANDIDATES]
        
        return list(set(vtos))  # Deduplicate
    
    def _llm_annotate(self, context: str, user_input: str, 
                      system_response: str) -> List[VTO]:
        """LLM-based VTO annotation"""
        prompt = VTO_ANNOTATION_PROMPT.format(
            vto_descriptions=self.vto_desc_str,
            context=context[:500],  # Truncate
            user_input=user_input,
            system_response=system_response[:500]
        )
        
        try:
            response = self.llm.generate(prompt, temperature=0.3, json_mode=True)
            result = json.loads(response)
            vto_names = result.get("vtos", [])
            
            vtos = []
            for name in vto_names:
                try:
                    vtos.append(VTO(name.lower()))
                except ValueError:
                    continue
            
            return vtos if vtos else self._heuristic_annotate(user_input, system_response)
        except:
            return self._heuristic_annotate(user_input, system_response)


# ============================================================================
# DATASET LOADER
# ============================================================================

class DatasetLoader:
    """Load and convert various data formats"""
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.annotator = VTOAnnotator(llm_client)
    
    def load_your_format(self, data: List[Dict]) -> List[Conversation]:
        """
        Load your conversation format:
        {
            "conversation_id": "...",
            "turns": [
                {
                    "turn_id": 1,
                    "user_input": "...",
                    "system_response": "...",
                    "intent": "...",
                    "tool_calls": [{"tool_name": "...", "arguments": {...}}],
                    "entities": {...}
                }
            ],
            "conversation_metadata": {...}
        }
        """
        conversations = []
        
        for conv_data in tqdm(data, desc="Loading conversations"):
            conv_id = conv_data.get("conversation_id", f"conv_{len(conversations)}")
            
            # Detect domain from entities or metadata
            domain = self._detect_domain(conv_data)
            
            turns = []
            context = ""
            
            for turn_data in conv_data.get("turns", []):
                user_input = turn_data.get("user_input", "")
                system_response = turn_data.get("system_response", "")
                
                # Convert tool calls
                tool_calls = []
                for tc in turn_data.get("tool_calls", []):
                    tool_name = tc.get("tool_name", "")
                    arguments = tc.get("arguments", {})
                    result = tc.get("execution_result", "")
                    
                    # Map tool to VTO
                    vto_type = self._tool_to_vto(tool_name)
                    
                    tool_calls.append(ToolCall(
                        tool_name=tool_name,
                        arguments=arguments,
                        vto_type=vto_type,
                        result=result,
                        is_explicit=True
                    ))
                
                # Annotate VTOs
                vtos = self.annotator.annotate_turn(context, user_input, system_response)
                
                # Add VTOs from explicit tools
                for tc in tool_calls:
                    if tc.vto_type not in vtos:
                        vtos.append(tc.vto_type)
                
                turns.append(ConversationTurn(
                    turn_id=turn_data.get("turn_id", len(turns)),
                    user_input=user_input,
                    system_response=system_response,
                    tool_calls=tool_calls,
                    vto_sequence=vtos,
                    entities=turn_data.get("entities", {}),
                    intent=turn_data.get("intent", "")
                ))
                
                context += f"\nUser: {user_input}\nAssistant: {system_response}"
            
            if turns:
                conversations.append(Conversation(
                    conversation_id=conv_id,
                    domain=domain,
                    turns=turns,
                    metadata=conv_data.get("conversation_metadata", {})
                ))
        
        return conversations
    
    def _detect_domain(self, conv_data: Dict) -> Domain:
        """Detect domain from conversation data"""
        # Check entities
        entities = {}
        for turn in conv_data.get("turns", []):
            entities.update(turn.get("entities", {}))
        
        category = entities.get("category", "").lower()
        
        if category in ["electronics", "phone", "laptop", "headphones", "smartwatch"]:
            return Domain.ELECTRONICS
        elif category in ["fashion", "clothing", "apparel", "shoes", "accessories"]:
            return Domain.FASHION
        elif category in ["movies", "film", "tv", "entertainment"]:
            return Domain.MOVIES
        elif category in ["food", "restaurant", "dining"]:
            return Domain.FOOD
        elif category in ["books", "reading"]:
            return Domain.BOOKS
        
        # Check tools used
        tools = conv_data.get("conversation_metadata", {}).get("tools_used", [])
        if tools:
            return Domain.GENERAL  # Has explicit tools
        
        return Domain.GENERAL
    
    def _tool_to_vto(self, tool_name: str) -> VTO:
        """Map tool name to VTO"""
        mapping = {
            "search": VTO.SEARCH_CANDIDATES,
            "search_products": VTO.SEARCH_CANDIDATES,
            "filter": VTO.FILTER_RESULTS,
            "compare": VTO.COMPARE_OPTIONS,
            "recommend": VTO.RANK_OPTIONS,
            "explain": VTO.EXPLAIN_CHOICE,
            "visual_match": VTO.MATCH_ATTRIBUTES,
            "style_check": VTO.MATCH_ATTRIBUTES,
        }
        return mapping.get(tool_name.lower(), VTO.SEARCH_CANDIDATES)


# ============================================================================
# PREFERENCE PAIR GENERATOR
# ============================================================================

class PreferencePairGenerator:
    """Generate preference pairs for CHARM training with batch processing"""
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client
    
    def generate_pairs_from_conversation(self, conv: Conversation,
                                         pairs_per_turn: int = 1) -> List[PreferencePair]:
        """Generate preference pairs from a conversation (single conversation)"""
        pairs = []
        context = ""
        
        for turn in conv.turns:
            # Original response is "chosen"
            chosen = turn.system_response
            chosen_vtos = turn.vto_sequence
            
            # Generate "rejected" alternatives
            for _ in range(pairs_per_turn):
                rejected = self._generate_rejected(
                    context, turn.user_input, chosen, conv.domain
                )
                
                if rejected and len(rejected) > 10:
                    # Compute hierarchical rewards
                    hier_rewards = self._compute_rewards(
                        context, turn.user_input, chosen, rejected
                    )
                    
                    pairs.append(PreferencePair(
                        conversation_id=conv.conversation_id,
                        context=context + f"\nUser: {turn.user_input}",
                        chosen_response=chosen,
                        rejected_response=rejected,
                        chosen_vtos=chosen_vtos,
                        rejected_vtos=[],  # Bad response likely wrong VTOs
                        chosen_tools=turn.tool_calls,
                        rejected_tools=[],
                        reward_margin=0.3,
                        hierarchical_rewards=hier_rewards,
                        domain=conv.domain
                    ))
            
            context += f"\nUser: {turn.user_input}\nAssistant: {turn.system_response}"
        
        return pairs
    
    def generate_pairs_batch(self, conversations: List[Conversation],
                             pairs_per_turn: int = 1,
                             show_progress: bool = True) -> List[PreferencePair]:
        """
        Generate preference pairs for multiple conversations using batch LLM processing.
        
        This is MUCH faster than processing one at a time.
        
        Args:
            conversations: List of conversations to process
            pairs_per_turn: Number of preference pairs per turn
            show_progress: Show progress bar
            
        Returns:
            List of PreferencePair objects
        """
        if not self.llm:
            # Fall back to heuristic generation (no batching needed)
            all_pairs = []
            iterator = tqdm(conversations, desc="Generating pairs (heuristic)") if show_progress else conversations
            for conv in iterator:
                all_pairs.extend(self.generate_pairs_from_conversation(conv, pairs_per_turn))
            return all_pairs
        
        # Step 1: Collect all prompts needed
        print("Step 1/3: Preparing prompts...")
        prompt_infos = []  # List of (conv_idx, turn_idx, context, user_input, chosen, domain)
        
        for conv_idx, conv in enumerate(conversations):
            context = ""
            for turn_idx, turn in enumerate(conv.turns):
                for _ in range(pairs_per_turn):
                    prompt_infos.append({
                        "conv_idx": conv_idx,
                        "turn_idx": turn_idx,
                        "context": context,
                        "user_input": turn.user_input,
                        "chosen": turn.system_response,
                        "chosen_vtos": turn.vto_sequence,
                        "chosen_tools": turn.tool_calls,
                        "domain": conv.domain,
                        "conv_id": conv.conversation_id
                    })
                context += f"\nUser: {turn.user_input}\nAssistant: {turn.system_response}"
        
        print(f"  Total prompts to process: {len(prompt_infos)}")
        
        # Step 2: Generate all prompts
        prompts = []
        for info in prompt_infos:
            prompt = CHARM_PREFERENCE_PROMPT.format(
                quality="low",
                context=info["context"][-500:],
                user_input=info["user_input"],
                domain=info["domain"].value
            )
            prompts.append(prompt)
        
        # Step 3: Batch LLM call
        print("Step 2/3: Generating rejected responses (batch)...")
        rejected_responses = self.llm.generate_batch_with_retry(
            prompts, 
            temperature=0.9, 
            max_tokens=256,
            show_progress=show_progress
        )
        
        # Step 4: Create PreferencePairs
        print("Step 3/3: Computing rewards and creating pairs...")
        all_pairs = []
        
        iterator = zip(prompt_infos, rejected_responses)
        if show_progress:
            iterator = tqdm(list(iterator), desc="Creating pairs")
        
        for info, rejected in iterator:
            if rejected and len(rejected) > 10:
                # Compute hierarchical rewards
                hier_rewards = self._compute_rewards(
                    info["context"], 
                    info["user_input"], 
                    info["chosen"], 
                    rejected
                )
                
                all_pairs.append(PreferencePair(
                    conversation_id=info["conv_id"],
                    context=info["context"] + f"\nUser: {info['user_input']}",
                    chosen_response=info["chosen"],
                    rejected_response=rejected,
                    chosen_vtos=info["chosen_vtos"],
                    rejected_vtos=[],
                    chosen_tools=info["chosen_tools"],
                    rejected_tools=[],
                    reward_margin=0.3,
                    hierarchical_rewards=hier_rewards,
                    domain=info["domain"]
                ))
        
        print(f"  Generated {len(all_pairs)} preference pairs")
        return all_pairs
    
    def _generate_rejected(self, context: str, user_input: str,
                           chosen: str, domain: Domain) -> str:
        """Generate a worse alternative response"""
        if self.llm:
            return self._llm_generate_rejected(context, user_input, domain)
        return self._heuristic_rejected(chosen)
    
    def _llm_generate_rejected(self, context: str, user_input: str,
                               domain: Domain) -> str:
        """Use LLM to generate rejected response"""
        prompt = CHARM_PREFERENCE_PROMPT.format(
            quality="low",
            context=context[-500:],  # Truncate
            user_input=user_input,
            domain=domain.value
        )
        
        try:
            return self.llm.generate(prompt, temperature=0.9, max_tokens=256)
        except:
            return ""
    
    def _heuristic_rejected(self, chosen: str) -> str:
        """Generate rejected response heuristically"""
        # Strategies for creating bad responses:
        strategies = [
            # Too generic
            "I can help you with that. Let me know if you need more information.",
            "That's a good choice. Would you like to see more options?",
            "I understand. Is there anything else I can help with?",
            # Off-topic
            "Have you considered trying a different category entirely?",
            "I'm not sure about that. Maybe try something else.",
            # Unhelpful
            "Sorry, I don't have much information on that.",
            "There are many options available. It's hard to recommend one.",
        ]
        
        return random.choice(strategies)
    
    def _compute_rewards(self, context: str, user_input: str,
                         chosen: str, rejected: str) -> Dict[str, Tuple[float, float]]:
        """Compute hierarchical reward scores
        
        IMPROVED: More nuanced scoring that better differentiates chosen from rejected.
        This is critical for CHARM training to learn meaningful preferences.
        """
        def score_response(response: str, user_input: str, context: str) -> Dict[str, float]:
            scores = {}
            response_lower = response.lower()
            user_lower = user_input.lower()
            context_lower = context.lower()
            
            # ===== RELEVANCE =====
            # Check how well response addresses the query
            user_words = set(user_lower.split()) - {'i', 'a', 'the', 'is', 'are', 'for', 'to', 'and'}
            response_words = set(response_lower.split())
            context_words = set(context_lower.split()) - {'i', 'a', 'the', 'is', 'are', 'for', 'to', 'and'}
            
            # Word overlap with user query
            user_overlap = len(user_words & response_words) / max(len(user_words), 1)
            # Context awareness
            context_overlap = len(context_words & response_words) / max(len(context_words), 1)
            
            # Specific item mentions boost relevance
            has_specific_items = bool(re.findall(r'\d+\.|\"|\'|specific|exactly|particular', response_lower))
            
            relevance = 0.2 + 0.3 * user_overlap + 0.2 * context_overlap + 0.2 * has_specific_items
            scores["relevance"] = min(relevance, 1.0)
            
            # ===== DIVERSITY =====
            # Penalize generic/vague responses
            generic_phrases = [
                "let me know", "anything else", "help you", "good choice",
                "many options", "lots of", "various", "different options",
                "you could try", "just", "maybe", "perhaps", "some"
            ]
            generic_count = sum(1 for p in generic_phrases if p in response_lower)
            
            # Reward specific language
            specific_phrases = [
                "specifically", "in particular", "notably", "features",
                "compared to", "unlike", "better than", "top rated",
                "best seller", "highly rated", "premium", "budget-friendly"
            ]
            specific_count = sum(1 for p in specific_phrases if p in response_lower)
            
            diversity = 0.8 - 0.1 * generic_count + 0.1 * specific_count
            scores["diversity"] = max(0.1, min(diversity, 1.0))
            
            # ===== USER SATISFACTION =====
            # Multi-factor assessment
            satisfaction = 0.3  # Base
            
            # Length factor (not just > 50)
            if len(response) < 20:
                satisfaction += 0.0
            elif len(response) < 50:
                satisfaction += 0.1
            elif len(response) < 100:
                satisfaction += 0.2
            else:
                satisfaction += 0.25
            
            # Helpfulness indicators
            helpful_indicators = [
                "recommend", "suggest", "try", "consider", "would be great",
                "perfect for", "ideal for", "here are", "top picks", "best options"
            ]
            helpful_count = sum(1 for h in helpful_indicators if h in response_lower)
            satisfaction += min(0.25, 0.05 * helpful_count)
            
            # Unhelpful indicators (reduces satisfaction)
            unhelpful_indicators = [
                "don't know", "not sure", "can't help", "unable to",
                "sorry", "unfortunately", "go somewhere else", "good luck"
            ]
            unhelpful_count = sum(1 for u in unhelpful_indicators if u in response_lower)
            satisfaction -= 0.15 * unhelpful_count
            
            # Actionable advice
            has_action = bool(re.findall(r'\d\.|first|then|next|finally|step', response_lower))
            satisfaction += 0.1 if has_action else 0
            
            scores["user_satisfaction"] = max(0.1, min(satisfaction, 1.0))
            
            # ===== ENGAGEMENT =====
            engagement = 0.3  # Base
            
            # Questions encourage engagement
            question_count = response.count("?")
            engagement += min(0.2, 0.1 * question_count)
            
            # Recommendations encourage engagement
            has_rec = any(w in response_lower for w in ["recommend", "suggest", "try", "consider"])
            engagement += 0.2 if has_rec else 0
            
            # Comparison encourages engagement
            has_comparison = any(w in response_lower for w in ["vs", "versus", "compared", "difference", "or"])
            engagement += 0.15 if has_comparison else 0
            
            # Call to action
            cta_phrases = ["let me", "shall i", "would you like", "want me to", "show you"]
            has_cta = any(p in response_lower for p in cta_phrases)
            engagement += 0.15 if has_cta else 0
            
            scores["engagement"] = max(0.1, min(engagement, 1.0))
            
            return scores
        
        chosen_scores = score_response(chosen, user_input, context)
        rejected_scores = score_response(rejected, user_input, context)
        
        # Ensure chosen is actually better (add minimum margin)
        for key in chosen_scores:
            if chosen_scores[key] <= rejected_scores[key]:
                # Chosen should be better - adjust slightly
                margin = 0.1 + random.random() * 0.1  # 0.1-0.2 margin
                chosen_scores[key] = min(1.0, rejected_scores[key] + margin)
        
        return {
            "relevance": (chosen_scores["relevance"], rejected_scores["relevance"]),
            "diversity": (chosen_scores["diversity"], rejected_scores["diversity"]),
            "user_satisfaction": (chosen_scores["user_satisfaction"], 
                                  rejected_scores["user_satisfaction"]),
            "engagement": (chosen_scores["engagement"], rejected_scores["engagement"])
        }


# ============================================================================
# DATA PREPARATION
# ============================================================================

def convert_to_sft_format(conversations: List[Conversation],
                          include_tools: bool = True) -> List[Dict]:
    """Convert conversations to SFT training format"""
    sft_data = []
    
    for conv in conversations:
        history = ""
        domain_token = f"<|domain:{conv.domain.value}|>"
        
        for turn in conv.turns:
            # Input
            input_text = f"{domain_token}\n{history}\nUser: {turn.user_input}"
            
            # Output with VTOs
            vto_str = ", ".join([v.value for v in turn.vto_sequence])
            output_parts = [f"<|think|>{vto_str}<|/think|>"]
            
            # Add tool calls if present
            if include_tools and turn.tool_calls:
                tool_str = json.dumps([{
                    "tool": tc.tool_name,
                    "args": tc.arguments
                } for tc in turn.tool_calls])
                output_parts.append(f"<|tool_start|>{tool_str}<|tool_end|>")
            
            output_parts.append(turn.system_response)
            output_text = "\n".join(output_parts)
            
            sft_data.append({
                "input": input_text.strip(),
                "output": output_text.strip(),
                "conversation_id": conv.conversation_id,
                "turn_id": turn.turn_id,
                "domain": conv.domain.value,
                "vtos": [v.value for v in turn.vto_sequence],
                "tools": [tc.tool_name for tc in turn.tool_calls],
                "satisfaction_score": conv.metadata.get("satisfaction_score", 0.7)
            })
            
            history += f"\nUser: {turn.user_input}\nAssistant: {turn.system_response}"
    
    return sft_data


def prepare_training_data(conversations: List[Conversation],
                          llm_client: Optional[LLMClient],
                          output_path: str,
                          generate_preferences: bool = True,
                          pairs_per_turn: int = 1,
                          use_batch: bool = True) -> Dict[str, Any]:
    """
    Prepare complete training data.
    
    Args:
        conversations: List of Conversation objects
        llm_client: Optional LLM client for generating rejected responses
        output_path: Directory to save output files
        generate_preferences: Whether to generate preference pairs
        pairs_per_turn: Number of preference pairs per conversation turn
        use_batch: Use batch processing for LLM calls (MUCH faster)
    
    Returns:
        Dict with sft_data, preference_data, and stats
    """
    os.makedirs(output_path, exist_ok=True)
    
    # Convert to SFT format
    print("Converting to SFT format...")
    sft_data = convert_to_sft_format(conversations)
    
    sft_path = os.path.join(output_path, "sft_data.json")
    with open(sft_path, "w") as f:
        json.dump(sft_data, f, indent=2)
    print(f"  Saved {len(sft_data)} SFT examples to {sft_path}")
    
    # Generate preference pairs
    preference_data = []
    if generate_preferences:
        print("\nGenerating preference pairs...")
        pair_generator = PreferencePairGenerator(llm_client)
        
        if use_batch and llm_client:
            # Use batch processing (MUCH faster with LLM)
            print("  Using batch processing (parallel LLM calls)...")
            pairs = pair_generator.generate_pairs_batch(
                conversations, 
                pairs_per_turn=pairs_per_turn,
                show_progress=True
            )
        else:
            # Fall back to sequential processing
            print("  Using sequential processing...")
            pairs = []
            for conv in tqdm(conversations, desc="Generating pairs"):
                pairs.extend(
                    pair_generator.generate_pairs_from_conversation(conv, pairs_per_turn)
                )
        
        # Convert to dict format
        for pair in pairs:
            preference_data.append({
                "conversation_id": pair.conversation_id,
                "context": pair.context,
                "chosen": pair.chosen_response,
                "rejected": pair.rejected_response,
                "chosen_vtos": [v.value for v in pair.chosen_vtos],
                "rejected_vtos": [v.value for v in pair.rejected_vtos],
                "reward_margin": pair.reward_margin,
                "hierarchical_rewards": {
                    k: {"chosen": v[0], "rejected": v[1]}
                    for k, v in pair.hierarchical_rewards.items()
                }
            })
        
        pref_path = os.path.join(output_path, "preference_data.json")
        with open(pref_path, "w") as f:
            json.dump(preference_data, f, indent=2)
        print(f"  Saved {len(preference_data)} preference pairs to {pref_path}")
    else:
        # Create empty file
        pref_path = os.path.join(output_path, "preference_data.json")
        with open(pref_path, "w") as f:
            json.dump([], f)
    
    # Statistics
    total_turns = sum(len(c.turns) for c in conversations)
    total_tools = sum(len(t.tool_calls) for c in conversations for t in c.turns)
    
    stats = {
        "num_conversations": len(conversations),
        "num_sft_examples": len(sft_data),
        "num_preference_pairs": len(preference_data),
        "total_turns": total_turns,
        "total_tool_calls": total_tools,
        "domains": {d.value: sum(1 for c in conversations if c.domain == d) for d in Domain},
        "tools_used": list(set(tc.tool_name for c in conversations 
                              for t in c.turns for tc in t.tool_calls))
    }
    
    stats_path = os.path.join(output_path, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nData preparation complete!")
    print(f"  Conversations: {stats['num_conversations']}")
    print(f"  SFT examples: {stats['num_sft_examples']}")
    print(f"  Preference pairs: {stats['num_preference_pairs']}")
    
    return {
        "sft_data": sft_data,
        "preference_data": preference_data,
        "stats": stats
    }