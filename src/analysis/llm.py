"""
LLM integration module for generating insights from restaurant reviews.
"""
import os
import json
from typing import Dict, List, Any, Optional, Union, Tuple
import time
import logging

from dotenv import load_dotenv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from anthropic import Anthropic

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv("config.env")

# LLM settings
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "local")
LLM_MODEL = os.getenv("LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
DEVICE = os.getenv("DEVICE", "cpu")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")


class LLMFactory:
    """Factory for creating and managing LLM instances."""
    
    _instance = None
    _llm = None
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        """Initialize LLM."""
        self._llm = None
    
    def get_llm(self):
        """Get or create LLM instance."""
        if self._llm is None:
            if LLM_PROVIDER == "anthropic":
                self._llm = AnthropicLLM()
            elif LLM_PROVIDER == "huggingface":
                self._llm = HuggingFaceLLM()
            else:  # local
                self._llm = LocalLLM()
                
        return self._llm


class BaseLLM:
    """Base class for LLM implementations."""
    
    def generate(self, prompt: str) -> str:
        """Generate text from prompt."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def generate_restaurant_analysis(
        self, 
        restaurant_info: Dict[str, Any],
        reviews: List[Dict[str, Any]],
        sentiment_summary: Dict[str, Any],
        sentiment_trends: Dict[str, Any]
    ) -> Dict[str, str]:
        """Generate a comprehensive analysis of restaurant reviews."""
        # Create prompt with restaurant data and reviews
        prompt = self._create_restaurant_analysis_prompt(
            restaurant_info, reviews, sentiment_summary, sentiment_trends
        )
        
        # Generate response
        response = self.generate(prompt)
        
        # Parse and structure the response
        analysis = self._parse_restaurant_analysis(response)
        
        return analysis
    
    def generate_sentiment_insights(
        self,
        reviews: List[Dict[str, Any]],
        sentiment_summary: Dict[str, Any]
    ) -> str:
        """Generate insights about sentiment patterns."""
        # Create prompt for sentiment insights
        prompt = self._create_sentiment_insights_prompt(reviews, sentiment_summary)
        
        # Generate response
        response = self.generate(prompt)
        
        return response
    
    def generate_competitive_analysis(
        self,
        restaurant_info: Dict[str, Any],
        sentiment_summary: Dict[str, Any],
        cuisine_type: str
    ) -> str:
        """Generate competitive analysis insights."""
        # Create prompt for competitive analysis
        prompt = self._create_competitive_analysis_prompt(
            restaurant_info, sentiment_summary, cuisine_type
        )
        
        # Generate response
        response = self.generate(prompt)
        
        return response
    
    def _create_restaurant_analysis_prompt(
        self,
        restaurant_info: Dict[str, Any],
        reviews: List[Dict[str, Any]],
        sentiment_summary: Dict[str, Any],
        sentiment_trends: Dict[str, Any]
    ) -> str:
        """Create a prompt for restaurant analysis."""
        # Format restaurant information
        restaurant_str = f"""
Restaurant: {restaurant_info.get('name')}
Address: {restaurant_info.get('address')}, {restaurant_info.get('city')}, {restaurant_info.get('state')} {restaurant_info.get('postal_code')}
Cuisine Type: {restaurant_info.get('cuisine_type')}
Price Range: {restaurant_info.get('price_range')}
Average Rating: {restaurant_info.get('average_rating')}
"""

        # Format sentiment summary
        sentiment_str = f"""
Total Reviews: {sentiment_summary.get('total_reviews')}
Positive Reviews: {sentiment_summary.get('positive_count')} ({sentiment_summary.get('positive_percentage'):.1f}%)
Negative Reviews: {sentiment_summary.get('negative_count')} ({sentiment_summary.get('negative_percentage'):.1f}%)
Neutral Reviews: {sentiment_summary.get('neutral_count')} ({sentiment_summary.get('neutral_percentage'):.1f}%)
Average Sentiment Score: {sentiment_summary.get('average_compound', 0):.2f}
Overall Sentiment Trend: {sentiment_trends.get('overall_trend', 'stable')}
"""

        # Format review samples 
        review_samples = []
        positive_reviews = [r for r in reviews if r.get('sentiment', {}).get('combined_label') == 'positive']
        negative_reviews = [r for r in reviews if r.get('sentiment', {}).get('combined_label') == 'negative']
        
        # Add top positive reviews
        positive_reviews.sort(key=lambda x: x.get('sentiment', {}).get('vader', {}).get('compound', 0), reverse=True)
        for i, review in enumerate(positive_reviews[:5]):
            review_samples.append(f"Positive Review #{i+1} [Rating: {review.get('rating')}]:\n{review.get('review_text')}\n")
            
        # Add top negative reviews
        negative_reviews.sort(key=lambda x: x.get('sentiment', {}).get('vader', {}).get('compound', 0))
        for i, review in enumerate(negative_reviews[:5]):
            review_samples.append(f"Negative Review #{i+1} [Rating: {review.get('rating')}]:\n{review.get('review_text')}\n")
        
        review_str = "\n".join(review_samples)

        # Create the final prompt
        prompt = f"""
You are a restaurant business analyst tasked with creating a detailed report for the management of {restaurant_info.get('name')}. 
Please analyze the following restaurant data, review samples, and sentiment analysis to provide insights.

### RESTAURANT INFORMATION ###
{restaurant_str}

### SENTIMENT SUMMARY ###
{sentiment_str}

### REVIEW SAMPLES ###
{review_str}

Based on the data above, please provide a comprehensive analysis of the restaurant with the following sections:
1. Executive Summary: A brief overview of the restaurant's performance and key insights.
2. Strengths: What customers love about the restaurant (specific food items, service aspects, ambiance, etc.).
3. Areas for Improvement: What customers complain about or suggest could be improved.
4. Customer Experience: Analysis of the overall customer experience, including service quality, ambiance, and value perception.
5. Food Quality: Insights about menu items, taste, presentation, and consistency.
6. Recommendations: Actionable suggestions for improving the restaurant's performance and customer satisfaction.

For each section, please be specific and reference actual data from the reviews. Format your response as JSON with the following structure:
{{
  "executive_summary": "Your analysis here",
  "strengths": "Your analysis here",
  "areas_for_improvement": "Your analysis here",
  "customer_experience": "Your analysis here",
  "food_quality": "Your analysis here",
  "recommendations": "Your analysis here"
}}
"""
        return prompt

    def _parse_restaurant_analysis(self, response: str) -> Dict[str, str]:
        """Parse the restaurant analysis response into structured data."""
        try:
            # Extract JSON from response if needed
            response = response.strip()
            
            # Find JSON in the response (it might be embedded in other text)
            start_idx = response.find('{')
            end_idx = response.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx+1]
                analysis = json.loads(json_str)
                
                # Ensure all expected keys are present
                expected_keys = [
                    "executive_summary", "strengths", "areas_for_improvement",
                    "customer_experience", "food_quality", "recommendations"
                ]
                
                for key in expected_keys:
                    if key not in analysis:
                        analysis[key] = "Not available"
                
                return analysis
            else:
                # If JSON parsing fails, structure the response manually
                return {
                    "executive_summary": "Unable to parse analysis in JSON format.",
                    "strengths": response[:500] if len(response) > 500 else response,
                    "areas_for_improvement": "Not available",
                    "customer_experience": "Not available",
                    "food_quality": "Not available",
                    "recommendations": "Not available"
                }
        except json.JSONDecodeError:
            # If JSON decoding fails, return a structured error response
            return {
                "executive_summary": "Unable to parse analysis in JSON format.",
                "strengths": response[:500] if len(response) > 500 else response,
                "areas_for_improvement": "Not available",
                "customer_experience": "Not available",
                "food_quality": "Not available",
                "recommendations": "Not available"
            }
    
    def _create_sentiment_insights_prompt(
        self,
        reviews: List[Dict[str, Any]],
        sentiment_summary: Dict[str, Any]
    ) -> str:
        """Create a prompt for sentiment insights."""
        summary_str = f"""
Total Reviews: {sentiment_summary.get('total_reviews')}
Positive Reviews: {sentiment_summary.get('positive_count')} ({sentiment_summary.get('positive_percentage'):.1f}%)
Negative Reviews: {sentiment_summary.get('negative_count')} ({sentiment_summary.get('negative_percentage'):.1f}%)
Neutral Reviews: {sentiment_summary.get('neutral_count')} ({sentiment_summary.get('neutral_percentage'):.1f}%)
Average Sentiment Score: {sentiment_summary.get('average_compound', 0):.2f}
"""

        # Extract common phrases and words from positive and negative reviews
        positive_reviews = [r.get('review_text', '') for r in reviews 
                          if r.get('sentiment', {}).get('combined_label') == 'positive']
        negative_reviews = [r.get('review_text', '') for r in reviews 
                          if r.get('sentiment', {}).get('combined_label') == 'negative']
        
        # Sample reviews
        positive_samples = positive_reviews[:5]
        negative_samples = negative_reviews[:5]
        
        positive_str = "\n".join([f"- {review[:200]}..." for review in positive_samples])
        negative_str = "\n".join([f"- {review[:200]}..." for review in negative_samples])
        
        prompt = f"""
You are a sentiment analysis expert. Based on the following sentiment summary and sample reviews, please provide insights about customer sentiment patterns, common themes in positive and negative reviews, and the emotional triggers that drive customer satisfaction or dissatisfaction.

### SENTIMENT SUMMARY ###
{summary_str}

### POSITIVE REVIEW SAMPLES ###
{positive_str}

### NEGATIVE REVIEW SAMPLES ###
{negative_str}

Please provide a detailed analysis that covers:
1. Key sentiment patterns and trends
2. Common positive themes and emotional triggers
3. Common negative themes and emotional triggers
4. Suggestions for leveraging positive sentiment
5. Recommendations for addressing negative sentiment

Keep your analysis concise but insightful, focusing on practical observations that could help improve the business.
"""
        return prompt
    
    def _create_competitive_analysis_prompt(
        self,
        restaurant_info: Dict[str, Any],
        sentiment_summary: Dict[str, Any],
        cuisine_type: str
    ) -> str:
        """Create a prompt for competitive analysis."""
        restaurant_str = f"""
Restaurant: {restaurant_info.get('name')}
Cuisine Type: {cuisine_type}
Price Range: {restaurant_info.get('price_range')}
Average Rating: {restaurant_info.get('average_rating')}
"""

        sentiment_str = f"""
Total Reviews: {sentiment_summary.get('total_reviews')}
Positive Percentage: {sentiment_summary.get('positive_percentage'):.1f}%
Negative Percentage: {sentiment_summary.get('negative_percentage'):.1f}%
Average Sentiment Score: {sentiment_summary.get('average_compound', 0):.2f}
"""

        prompt = f"""
You are a restaurant industry consultant specializing in competitive analysis. Based on the following restaurant data and sentiment summary, please provide a competitive analysis for this {cuisine_type} restaurant.

### RESTAURANT INFORMATION ###
{restaurant_str}

### SENTIMENT SUMMARY ###
{sentiment_str}

Please provide a detailed competitive analysis that includes:
1. Typical strengths and weaknesses of {cuisine_type} restaurants in this price range
2. Key differentiators that successful {cuisine_type} restaurants employ
3. Common challenges faced by {cuisine_type} restaurants and how to address them
4. Recommendations for competitive positioning in the {cuisine_type} restaurant market
5. Emerging trends in the {cuisine_type} restaurant industry that could be leveraged

Keep your analysis focused on practical insights that could help this restaurant gain a competitive advantage.
"""
        return prompt


class LocalLLM(BaseLLM):
    """Implementation using local models with transformers."""
    
    def __init__(self):
        """Initialize local LLM."""
        self.model = None
        self.tokenizer = None
    
    def _load_model(self):
        """Load the model and tokenizer if not already loaded."""
        if self.model is None or self.tokenizer is None:
            try:
                logger.info(f"Loading model: {LLM_MODEL}")
                self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
                self.model = AutoModelForCausalLM.from_pretrained(
                    LLM_MODEL,
                    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                    low_cpu_mem_usage=True if DEVICE == "cpu" else False,
                    trust_remote_code=True
                )
                
                if DEVICE == "cuda" and torch.cuda.is_available():
                    self.model = self.model.to("cuda")
                
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                raise
    
    def generate(self, prompt: str) -> str:
        """Generate text from prompt."""
        self._load_model()
        
        try:
            # Create the text generation pipeline
            pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=4096,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                device=0 if DEVICE == "cuda" and torch.cuda.is_available() else -1
            )
            
            # Generate text
            result = pipe(prompt)[0]["generated_text"]
            
            # Remove the prompt from the result
            response = result[len(prompt):].strip()
            
            return response
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            return f"Error generating analysis: {str(e)}"


class HuggingFaceLLM(BaseLLM):
    """Implementation using Hugging Face models with proper handling of instruction formats."""
    
    def __init__(self):
        """Initialize Hugging Face LLM."""
        self.model = None
        self.tokenizer = None
        
        # Define instruction templates for different models
        self.templates = {
            "mistralai/Mistral-7B-Instruct-v0.2": {
                "prefix": "<s>[INST] ",
                "suffix": " [/INST]",
                "response_prefix": ""
            },
            "tiiuae/falcon-7b-instruct": {
                "prefix": "User: ",
                "suffix": "\nAssistant: ",
                "response_prefix": "\nAssistant: "
            },
            "google/gemma-7b-it": {
                "prefix": "<start_of_turn>user\n",
                "suffix": "<end_of_turn>\n<start_of_turn>model\n",
                "response_prefix": "<start_of_turn>model\n"
            },
            # Add more templates as needed for other models
            "default": {
                "prefix": "### Instruction:\n",
                "suffix": "\n### Response:\n",
                "response_prefix": "\n### Response:\n"
            }
        }
    
    def _get_template(self) -> Dict[str, str]:
        """Get the appropriate template for the current model."""
        for model_name, template in self.templates.items():
            if model_name in LLM_MODEL:
                return template
        return self.templates["default"]
    
    def _load_model(self):
        """Load the model and tokenizer if not already loaded."""
        if self.model is None or self.tokenizer is None:
            try:
                logger.info(f"Loading model: {LLM_MODEL}")
                
                # Use token if provided
                token = HF_TOKEN if HF_TOKEN else None
                
                self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, token=token)
                self.model = AutoModelForCausalLM.from_pretrained(
                    LLM_MODEL,
                    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                    low_cpu_mem_usage=True if DEVICE == "cpu" else False,
                    trust_remote_code=True,
                    token=token
                )
                
                if DEVICE == "cuda" and torch.cuda.is_available():
                    self.model = self.model.to("cuda")
                
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                raise
    
    def generate(self, prompt: str) -> str:
        """Generate text from prompt."""
        self._load_model()
        
        try:
            # Get the appropriate template
            template = self._get_template()
            
            # Format the prompt according to the template
            formatted_prompt = f"{template['prefix']}{prompt}{template['suffix']}"
            
            # Generate text
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
            
            if DEVICE == "cuda" and torch.cuda.is_available():
                inputs = inputs.to("cuda")
                
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_length=4096,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the response part
            if template['response_prefix'] in full_response:
                response = full_response.split(template['response_prefix'], 1)[1].strip()
            else:
                # If we can't find the response prefix, return everything after the prompt
                response = full_response[len(formatted_prompt):].strip()
            
            return response
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            return f"Error generating analysis: {str(e)}"


class AnthropicLLM(BaseLLM):
    """Implementation using Anthropic's Claude API."""
    
    def __init__(self):
        """Initialize Anthropic client."""
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
            
        self.client = Anthropic(api_key=ANTHROPIC_API_KEY)
        self.model = LLM_MODEL or "claude-3-sonnet-20240229"
    
    def generate(self, prompt: str) -> str:
        """Generate text from prompt."""
        try:
            # Call Anthropic API
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                temperature=0.7,
                system="You are an expert restaurant business analyst specializing in sentiment analysis and customer feedback. Provide detailed, insightful analysis based on the data presented.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract and return the response text
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error calling Anthropic API: {str(e)}")
            return f"Error generating analysis: {str(e)}"