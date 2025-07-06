from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
import google.generativeai as genai
import tweepy
from dotenv import load_dotenv
import os
from typing import Optional

# Load environment variables
load_dotenv()

# === Configure Gemini ===
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

config_list = [{
    "model": "gemini-2.5-pro",
    "api_key": os.getenv("GEMINI_API_KEY"),
    "base_url": "https://generativelanguage.googleapis.com/v1beta"
}]

llm_config = {
    "config_list": config_list,
    "temperature": 0.3,
    "timeout": 60,
}

# === Twitter API Function ===
def post_to_twitter(tweet_text):
    """Manual tweet posting function with verbose logging"""
    print("\n=== EXECUTING post_to_twitter ===")
    print(f"Tweet content: {tweet_text}")
    try:
        # Check if we have all required API keys
        required_keys = [
            "TWITTER_API_KEY", "TWITTER_API_SECRET",
            "TWITTER_ACCESS_TOKEN", "TWITTER_ACCESS_TOKEN_SECRET"
        ]
        print("TWITTER_API_KEY:", os.getenv("TWITTER_API_KEY"), "...", len(os.getenv("TWITTER_API_KEY")))
        print("TWITTER_API_SECRET:", os.getenv("TWITTER_API_SECRET"), "...", len(os.getenv("TWITTER_API_SECRET")))
        print("TWITTER_ACCESS_TOKEN:", os.getenv("TWITTER_ACCESS_TOKEN"), "...", len(os.getenv("TWITTER_ACCESS_TOKEN")))
        print("TWITTER_ACCESS_TOKEN_SECRET:", os.getenv("TWITTER_ACCESS_TOKEN_SECRET"), "...", len(os.getenv("TWITTER_ACCESS_TOKEN_SECRET")))
        print("TWITTER_BEARER_TOKEN:", os.getenv("TWITTER_BEARER_TOKEN"), "...", len(os.getenv("TWITTER_BEARER_TOKEN")))

        if not all(os.getenv(k) for k in required_keys):
            print("⚠️ Twitter API keys not configured. Simulating tweet post.")
            print(f"Simulated tweet: {tweet_text}")
            return {"id": "simulated", "text": tweet_text}
        
        print("🔑 Twitter API keys found. Attempting real post...")
        client = tweepy.Client(
            consumer_key=os.getenv("TWITTER_API_KEY"),
            consumer_secret=os.getenv("TWITTER_API_SECRET"),
            access_token=os.getenv("TWITTER_ACCESS_TOKEN"),
            access_token_secret=os.getenv("TWITTER_ACCESS_TOKEN_SECRET")
        )
        response = client.create_tweet(text=tweet_text, user_auth=True)
        if not response.data:
            raise Exception("No data returned from Twitter API")
        print(f"✅ Successfully tweeted: {tweet_text}")
        return {"id": response.data["id"], "text": response.data["text"]}
    except Exception as e:
        print(f"❌ Error posting tweet: {e}")
        return None

# === Agents ===
def is_termination_msg(msg):
    """Check if message contains termination signal"""
    content = msg.get("content", "").upper()
    return any(word in content for word in ["TERMINATE", "TWEET POSTED", "ERROR"])

user_agent = UserProxyAgent(
    name="user",
    system_message="A user who wants to post technical tweets",
    code_execution_config=False,
    human_input_mode="NEVER",
    is_termination_msg=is_termination_msg,
)

tweet_generator = AssistantAgent(
    name="tweet_generator",
    llm_config=llm_config,
    system_message="""
You craft technical tweets about system design. Rules:
1. Focus on system design concepts
2. Use humor and senior-dev tone
3. Strictly 280 chars max
4. Include 1-2 relevant hashtags
5. Respond ONLY with the tweet text
""",
    is_termination_msg=is_termination_msg,
)

tweet_critic = AssistantAgent(
    name="tweet_critic",
    llm_config=llm_config,
    system_message="""
You evaluate technical tweets. Respond in this format:
---
VERDICT: [APPROVED/REJECTED]
FEEDBACK: [Specific suggestions]
---
Criteria:
1. Technical accuracy
2. Engagement potential
3. Clarity
4. Length (280 chars max)
""",
    is_termination_msg=is_termination_msg,
)

# === Manual Posting Logic ===
class ManualPosterAgent(AssistantAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_tweet = None
        self.posted_successfully = False
    
    def receive(self, message, sender, request_reply=True, silent=False):
        # Store the last tweet from generator
        if message.get("name") == "tweet_generator":
            self.last_tweet = message.get("content", "").strip()
        
        # Check for approval from critic
        if (message.get("name") == "tweet_critic" and 
            "VERDICT: APPROVED" in message.get("content", "") and
            self.last_tweet and not self.posted_successfully):
            
            print("\n=== MANUAL POSTING TRIGGERED ===")
            result = post_to_twitter(self.last_tweet)
            
            if result and result.get("id"):
                self.posted_successfully = True
                # Return both status and termination
                return f"TWEET POSTED: {self.last_tweet}\nTERMINATE"
            return f"ERROR: Failed to post tweet\nTERMINATE"
        
        return super().receive(message, sender, request_reply, silent)

tweet_poster = ManualPosterAgent(
    name="tweet_poster",
    llm_config=llm_config,
    system_message="""
You are the final gatekeeper. When the critic approves a tweet:
1. Wait for the system to automatically post it
2. Confirm the posting result
3. Terminate the conversation
""",
    human_input_mode="NEVER",
    is_termination_msg=is_termination_msg,
)

# === Group Chat Setup ===
groupchat = GroupChat(
    agents=[user_agent, tweet_generator, tweet_critic, tweet_poster],
    messages=[],
    max_round=4,
    speaker_selection_method="auto",
)

manager = GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config,
    is_termination_msg=is_termination_msg,
)

# === Main Execution ===
def run_workflow(topic: str = "system design") -> Optional[str]:
    """Execute the tweet generation and manual posting workflow"""
    print(f"\n=== Starting workflow for: {topic} ===")
    
    # Reset all agents
    manager.reset()
    for agent in groupchat.agents:
        agent.reset()
    
    # Initiate chat
    user_agent.initiate_chat(
        manager,
        message=f"Create a humorous yet insightful tweet about {topic}",
        clear_history=True
    )
    
    # Check results by scanning messages for posting confirmation
    for msg in reversed(manager.groupchat.messages):
        content = msg.get("content", "")
        if "TWEET POSTED:" in content:
            # Extract just the tweet content (before any newlines)
            tweet_content = content.split("TWEET POSTED:")[1].split("\n")[0].strip()
            print(f"\n=== WORKFLOW COMPLETE ===")
            print(f"Result: Tweet successfully processed")
            return tweet_content
    
    print("\n❌ Workflow did not complete successfully")
    return None

if __name__ == "__main__":
    successful_tweet = run_workflow("microservices architecture")
    
    if successful_tweet:
        twitter_configured = all(os.getenv(k) for k in [
            "TWITTER_API_KEY", "TWITTER_API_SECRET",
            "TWITTER_ACCESS_TOKEN", "TWITTER_ACCESS_TOKEN_SECRET"
        ])
        
        print(f"\n{'🎉 REAL TWEET POSTED' if twitter_configured else '🔄 SIMULATED TWEET'}:")
        print("-"*50)
        print(successful_tweet)
        print("-"*50)
    else:
        print("\n❌ Failed to generate and post an approved tweet")