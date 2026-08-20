import re

def preprocess_choice_data(example, task_name):
    """
    Standardize incoming data profiles.
    
    Args:
        example (dict): A single data example from the dataset.
        task_name (str): The dataset task name, either 'commonsense_qa' or 'social_iqa'.
        
    Returns:
        dict: A standardized dictionary containing:
            - 'context': The prompt/context string.
            - 'choices': A list of string choices.
            - 'label': The 0-indexed integer label of the correct choice.
    """
    # Normalize task_name to support variations (e.g. 'CommonsenseQA', 'commonsense_qa', 'SocialIQA', etc.)
    task_name_clean = str(task_name).lower().replace("_", "").replace(" ", "").replace("-", "")
    
    if task_name_clean == "commonsenseqa":
        # CommonsenseQA format:
        # question: str
        # choices: {"text": [str, ...], "label": [str, ...]}
        # answerKey: str (e.g., 'A', 'B', 'C', 'D', 'E')
        question = example.get("question", "")
        choices_dict = example.get("choices", {})
        choices_text = choices_dict.get("text", [])
        answer_key = example.get("answerKey", "")
        
        # Map 'A', 'B', ... to 0, 1, ...
        # Standardize inputs: handle uppercase and potential list formats gracefully
        label = -1
        if isinstance(answer_key, str) and answer_key:
            label = ord(answer_key.upper()) - ord("A")
            
        return {
            "context": question,
            "choices": choices_text,
            "label": label
        }
        
    elif task_name_clean == "socialiqa":
        # SocialIQA format:
        # context: str
        # question: str
        # answerA: str
        # answerB: str
        # answerC: str
        # label: str (e.g., '1', '2', '3')
        context = example.get("context", "")
        question = example.get("question", "")
        prompt = f"{context} {question}".strip()
        
        choices = [
            example.get("answerA", ""),
            example.get("answerB", ""),
            example.get("answerC", "")
        ]
        
        raw_label = example.get("label", "")
        label = -1
        if isinstance(raw_label, str) and raw_label.isdigit():
            label = int(raw_label) - 1
        elif isinstance(raw_label, int):
            label = raw_label - 1
            
        return {
            "context": prompt,
            "choices": choices,
            "label": label
        }
    else:
        raise ValueError(f"Unknown task name: {task_name}")


class SemanticFilters:
    @staticmethod
    def filter_negation(text):
        """
        Flags the presence of negation words.
        
        Args:
            text (str): Input text string.
            
        Returns:
            bool: True if negation indicators are present, False otherwise.
        """
        if not isinstance(text, str):
            return False
        # Match word boundaries for 'not', 'never', 'no', 'cannot', 'neither', 'nor', 'none', 'without'
        pattern = re.compile(r"\b(not|never|no|cannot|neither|nor|none|without)\b", re.IGNORECASE)
        return bool(pattern.search(text))

    @staticmethod
    def filter_spatial(text):
        """
        Flags the presence of spatial indicators.
        
        Args:
            text (str): Input text string.
            
        Returns:
            bool: True if spatial indicators are present, False otherwise.
        """
        if not isinstance(text, str):
            return False
        # Match word boundaries for spatial keywords
        pattern = re.compile(
            r"\b(inside|top|under|left|right|near|above|below|behind|front|outside|between|beside|across|through)\b",
            re.IGNORECASE
        )
        return bool(pattern.search(text))

    @staticmethod
    def filter_causal(text):
        """
        Flags the presence of causal indicator words.
        
        Args:
            text (str): Input text string.
            
        Returns:
            bool: True if causal indicators are present, False otherwise.
        """
        if not isinstance(text, str):
            return False
        # Match word boundaries for causal keywords
        pattern = re.compile(
            r"\b(because|causes?|why|reasons?|results?|consequences?|effects?|happen|since|due to)\b",
            re.IGNORECASE
        )
        return bool(pattern.search(text))

    @staticmethod
    def filter_temporal(text):
        """
        Flags the presence of temporal sequences or timing indicator words.
        
        Args:
            text (str): Input text string.
            
        Returns:
            bool: True if temporal indicators are present, False otherwise.
        """
        if not isinstance(text, str):
            return False
        # Match word boundaries for temporal keywords
        pattern = re.compile(
            r"\b(before|after|during|meanwhile|first|next|last|until|while|when|then|yesterday|tomorrow|initially|finally|soon|suddenly)\b",
            re.IGNORECASE
        )
        return bool(pattern.search(text))

    @staticmethod
    def filter_social_emotional(text):
        """
        Flags the presence of social interactions, emotional states, or motivations.
        
        Args:
            text (str): Input text string.
            
        Returns:
            bool: True if social/emotional indicators are present, False otherwise.
        """
        if not isinstance(text, str):
            return False
        # Match word boundaries for emotions, motivation, and social roles
        pattern = re.compile(
            r"\b(feel|want|need|describe|intent|motivated|reaction|implication|empathy|love|hate|angry|sad|happy|upset|proud|ashamed|scared|afraid|excited|nervous|jealous|friend|relationship|others)\b",
            re.IGNORECASE
        )
        return bool(pattern.search(text))

    @staticmethod
    def filter_comparative(text):
        """
        Flags the presence of relative comparisons, superlatives, and comparative limits.
        
        Args:
            text (str): Input text string.
            
        Returns:
            bool: True if comparative indicators are present, False otherwise.
        """
        if not isinstance(text, str):
            return False
        # Matches comparative/superlative keywords or words ending in 'er'/'est' with word boundaries
        pattern = re.compile(
            r"\b(more|less|than|most|least|better|worse|best|worst|taller|larger|smaller|greater|faster|slower|heavier|lighter|older|newer)\b|\b\w+est\b", 
            re.IGNORECASE
        )
        return bool(pattern.search(text))

    @staticmethod
    def filter_quantifier(text):
        """
        Flags the presence of quantitative, numeric scope, or quantificational keywords.
        
        Args:
            text (str): Input text string.
            
        Returns:
            bool: True if quantificational indicators are present, False otherwise.
        """
        if not isinstance(text, str):
            return False
        # Matches quantifiers and numeric indicators
        pattern = re.compile(
            r"\b(all|every|each|some|many|few|several|both|either|neither|none|any|most|one|two|three|double|triple|twice|once)\b",
            re.IGNORECASE
        )
        return bool(pattern.search(text))

    @staticmethod
    def profile_subcategories(text):
        """
        Profiles the text across 17 fine-grained sub-categories.
        
        Args:
            text (str): Input text string.
            
        Returns:
            dict: Dictionary of category/sub-category matches.
        """
        if not isinstance(text, str):
            return {}
            
        matches = {}
        
        # 1. Negation Sub-categories
        if re.search(r"\b(not|cannot)\b|n't", text, re.IGNORECASE):
            matches["Negation_Direct"] = True
        if re.search(r"\b(never|none|nothing|nobody|nowhere|impossible|unlike)\b", text, re.IGNORECASE):
            matches["Negation_Lexical"] = True
        if re.search(r"\b(without|except|unless)\b", text, re.IGNORECASE):
            matches["Negation_Prepositional"] = True
        if re.search(r"\b(neither|nor|but not|instead)\b", text, re.IGNORECASE):
            matches["Negation_Contrastive"] = True
            
        # 2. Spatial Sub-categories
        if re.search(r"\b(left|right|front|behind|east|west|north|south)\b", text, re.IGNORECASE):
            matches["Spatial_Directional"] = True
        if re.search(r"\b(top|above|under|below|bottom|beneath)\b", text, re.IGNORECASE):
            matches["Spatial_Vertical"] = True
        if re.search(r"\b(inside|outside|near|between|beside|across|through|next)\b", text, re.IGNORECASE):
            matches["Spatial_Proximity"] = True
            
        # 3. Causal Sub-categories
        if re.search(r"\b(why|because|reason|due to)\b", text, re.IGNORECASE):
            matches["Causal_Attribution"] = True
        if re.search(r"\b(result|consequence|effect|leads to|causes?)\b", text, re.IGNORECASE):
            matches["Causal_Outcome"] = True
            
        # 4. Temporal Sub-categories
        if re.search(r"\b(before|after|first|next|last|then|initially|finally|soon|suddenly)\b", text, re.IGNORECASE):
            matches["Temporal_Sequence"] = True
        if re.search(r"\b(during|while|meanwhile|until)\b", text, re.IGNORECASE):
            matches["Temporal_Simultaneity"] = True
            
        # 5. Social/Emotional Sub-categories
        if re.search(r"\b(feel|happy|sad|angry|afraid|anxious|upset|proud|ashamed|scared|excited|nervous|jealous|love|hate)\b", text, re.IGNORECASE):
            matches["Social_Emotion"] = True
        if re.search(r"\b(want|need|intent|motivation|goal)\b", text, re.IGNORECASE):
            matches["Social_Motivation"] = True
        if re.search(r"\b(friend|relationship|others|family|partner)\b", text, re.IGNORECASE):
            matches["Social_Interpersonal"] = True
            
        # 6. Comparative Sub-categories
        if re.search(r"\b(most|least|best|worst)\b|\b\w+est\b", text, re.IGNORECASE):
            matches["Comparative_Superlative"] = True
        if re.search(r"\b(more|less|than|better|worse|taller|larger|smaller|greater|faster|slower|heavier|lighter|older|newer)\b", text, re.IGNORECASE):
            matches["Comparative_Relative"] = True
            
        # 7. Quantifier Sub-categories
        if re.search(r"\b(all|every|each)\b", text, re.IGNORECASE):
            matches["Quantifier_Universal"] = True
        if re.search(r"\b(some|many|few|several|any)\b", text, re.IGNORECASE):
            matches["Quantifier_Partial"] = True
        if re.search(r"\b(one|two|three|double|triple|twice|once)\b", text, re.IGNORECASE):
            matches["Quantifier_Numeric"] = True
            
        return matches

