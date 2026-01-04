"""
ToonDB Agent Memory System - Customer Support Scenario
Simulates a multi-turn customer support conversation
"""
from typing import List


class CustomerSupportScenario:
    """
    Customer support scenario with temporal continuity
    
    Simulates a customer contacting support multiple times over 24 hours.
    Agent must remember:
    - Previous issues discussed
    - Account details mentioned
    - Solutions already attempted
    - Follow-up commitments made
    """
    
    def __init__(self):
        self.messages = self._create_scenario()
    
    def _create_scenario(self) -> List[str]:
        """Create realistic customer support conversation"""
        return [
            # Initial contact - Morning
            "Hi, I'm having trouble logging into my account. It keeps saying 'invalid credentials' but I'm sure my password is correct.",
            
            "I've tried resetting it twice already. The reset emails arrive fine, but when I use the new password, same error.",
            
            "My email is john.doe@example.com and my account ID is ACC-7482. I've been a customer for 3 years.",
            
            "Yes, I'm using Chrome on Windows 11. Version 120.0.6099.109.",
            
            "Okay, I'll try clearing my browser cache and cookies. Give me a minute.",
            
            "That didn't work either. Still getting the same error. This is really frustrating - I need to access my account urgently.",
            
            "What do you mean check for browser extensions? I have LastPass and uBlock Origin installed.",
            
            "Alright, I disabled LastPass and it worked! I can log in now. Thanks so much!",
            
            "Just to confirm - is it safe to re-enable LastPass after I'm logged in, or will that cause the same issue?",
            
            "Got it, thank you for your help! I'll keep LastPass disabled for your site.",
            
            # Follow-up conversation - Afternoon (few hours later)
            "Hi, it's me again. Remember we spoke earlier about the login issue?",
            
            "Well, now I'm facing a different problem. I can log in fine (with LastPass disabled like you suggested), but when I try to download my invoice history, nothing happens.",
            
            "I click the 'Download All Invoices' button and it just spins indefinitely. I've tried multiple times.",
            
            "I need the invoices from the last 6 months for my tax filing. Is there another way to get them?",
            
            "Tried that, still the same spinner issue. Could you email them to john.doe@example.com instead?",
            
            "Perfect! How long will that take? I need to submit my taxes by end of day.",
            
            "Thank you! That would be very helpful. I really appreciate your patience with all these issues today.",
            
            # Another follow-up - Evening
            "Hey, quick question about my account (ACC-7482). I received the invoices you emailed earlier - thank you!",
            
            "I noticed I was charged twice for the same service in March. Invoice #INV-3847 and #INV-3891 both show $49.99 for 'Premium Plan'.",
            
            "Yes, I can see that INV-3847 is dated March 3rd and INV-3891 is March 17th. But I should only be charged once per month, right?",
            
            "Oh! I see. So the March 17th charge was for upgrading from Basic to Premium mid-cycle?",
            
            "That makes sense now. I forgot I upgraded halfway through the month. Sorry for the confusion!",
            
            "No other issues. You've been incredibly helpful today with the login problem, the invoices, and now explaining the billing.",
            
            "One last question - when does my annual renewal happen? Want to make sure there are no surprises.",
            
            "Got it, September 15th. I'll make a note. Thanks again for all your help today!",
            
            # Next day follow-up
            "Good morning! Remember helping me yesterday with login and billing issues?",
            
            "Everything's working fine now, but I wanted to follow up on the LastPass issue. Have you heard from other customers with similar problems?",
            
            "Interesting. Do you know if your engineering team is working on a fix? I really prefer using LastPass for security.",
            
            "That's great to hear! Can I sign up to be notified when the fix is released?",
            
            "Perfect. My email is john.doe@example.com - same as my account. When should I expect an update?",
            
            "Sounds good! Thanks for the excellent support. You've really gone above and beyond.",
            
            # Final message
            "One more thing - is there a satisfaction survey I can fill out? Your support has been exceptional and I'd like to give you a positive review.",
        ]
    
    def get_messages(self) -> List[str]:
        """Get all scenario messages"""
        return self.messages
    
    def get_num_turns(self) -> int:
        """Get number of turns in scenario"""
        return len(self.messages)


def get_scenario(name: str = "customer_support"):
    """Factory function to get scenarios"""
    if name == "customer_support":
        return CustomerSupportScenario()
    elif name == "research_assistant":
        from scenarios.research_assistant import ResearchAssistantScenario
        return ResearchAssistantScenario()
    else:
        raise ValueError(f"Unknown scenario: {name}. Available: ['customer_support', 'research_assistant']")
