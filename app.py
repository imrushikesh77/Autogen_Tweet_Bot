from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager, Agent
import google.generativeai as genai
import tweepy
from dotenv import load_dotenv
import os

load_dotenv()

# Twitter API Configuration
def post_to_twitter(tweet_text):
    client = tweepy.Client(
        consumer_key=os.getenv("TWITTER_API_KEY"),
        consumer_secret=os.getenv("TWITTER_API_SECRET"),
        access_token=os.getenv("TWITTER_ACCESS_TOKEN"),
        access_token_secret=os.getenv("TWITTER_ACCESS_TOKEN_SECRET"),
        wait_on_rate_limit=True,
    )
    try:
        response = client.create_tweet(text=tweet_text)
        print(f"Successfully tweeted: {tweet_text}")
    except Exception as e:
        print(f"Error posting tweet: {e}")


# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

config_list = [{
    "model": "gemini-2.0-flash",
    "api_key": os.getenv("GEMINI_API_KEY"),
    "base_url": "https://generativelanguage.googleapis.com/v1beta"
}]

llm_config = {
    "config_list": config_list,
    "temperature": 0.2,
    "timeout": 60,
}

class TweetingAgent(AssistantAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def generate_reply(self, messages=None, sender=None, **kwargs):
        # First generate the response normally
        reply = super().generate_reply(messages, sender, **kwargs)
        
        # Only post the actual generated content (not the whole message history)
        if reply and isinstance(reply, str):
            post_to_twitter(reply)
        
        return reply

tweetingAgent = TweetingAgent(
    name="tweeting_agent",
    system_message="""You are an expert C programmer creating technical tweets for experienced developers.

        Guidelines:
        1. Cover intermediate to advanced C concepts – skip basic syntax, but don’t go ultra-obscure (no compiler internals or arcane UB unless it's practical).
        2. Focus on clever uses of the language, performance tips, or overlooked features.
        3. Use a confident, meme-driven format like “Interviewer said / Chad me” or other short-form humor that conveys mastery.
        4. Use 1-2 relevant technical hashtags like #SystemsProgramming or #LowLevel.
        5. Keep it under 280 characters.
        6. Return ONLY the tweet text.
        7. Style should feel like a senior dev casually flexing knowledge – funny, sharp, and useful without going too deep into niche academia.

        Examples:
        - "Interviewer: 'You can't return a local array in C.'  
        Me: `return (int[]){1, 2, 3};`  
        Compound literals exist. Stay mad.  
        #CProgramming #LowLevel"

        - "Interviewer: 'C can't do RAII.'  
        Me: `_cleanup_` attribute with GCC.  
        Who needs destructors when you have GNU extensions?  
        #Ctips #SystemsProgramming"

        Avoid:
        - Overly simple content (e.g., “Pointers are variables that store addresses.”)
        - Overly complex stuff (e.g., “manually manipulating the ELF symbol table for loader trickery”)
        - Anything that requires extensive setup or is too niche for a tweet.""",
    llm_config=llm_config,
)


userProxyAgent = UserProxyAgent(
    name="user",
    system_message="You are a user who wants to tweet.",
    code_execution_config=False,
    human_input_mode="NEVER",
)

def state_transition(last_speaker, groupchat):
    return tweetingAgent if last_speaker.name == "user" else None

groupchat = GroupChat(
    agents=[userProxyAgent, tweetingAgent],
    max_round=2,
    speaker_selection_method=state_transition,
)

manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)

userProxyAgent.initiate_chat(
    manager,
    message="Can you write a tweet about C?",
)
