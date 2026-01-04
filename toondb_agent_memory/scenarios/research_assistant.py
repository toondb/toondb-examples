"""
ToonDB Agent Memory System - Research Assistant Scenario
A more complex scenario with multi-topic research, data aggregation, and cross-references
"""
from typing import List


class ResearchAssistantScenario:
    """
    Research assistant helping with a complex multi-day research project
    
    This scenario is more demanding than customer support because:
    - Multiple distinct research topics
    - Cross-references between conversations
    - Data aggregation from multiple sources
    - Long-term memory requirements (references from days ago)
    - Higher context complexity
    """
    
    def __init__(self):
        self.messages = self._create_scenario()
    
    def _create_scenario(self) -> List[str]:
        """Create complex research assistant conversation"""
        return [
            # Day 1 Morning - Initial research request
            "I'm working on a research paper about the environmental impact of AI data centers. Can you help me organize my research?",
            
            "Let's start with energy consumption. What are the key metrics I should be tracking?",
            
            "Good points. I found a Nature paper from 2023 that says data centers consume 1-2% of global electricity. Can you note that down?",
            
            "Also, there's a study showing GPT-3 training consumed 1,287 MWh. That's equivalent to 522 tons of CO2. Add that to our notes.",
            
            "What about water usage? I heard data centers use a lot of water for cooling.",
            
            "OK, found it - Microsoft's data centers used 1.7 billion gallons in 2021. That's a 30% increase from 2020.",
            
            "Let me switch topics. What about the carbon footprint of inference vs training? Which is bigger over time?",
            
            "Interesting. So if a model like ChatGPT serves 100M users daily, the cumulative inference emissions could exceed training emissions within months?",
            
            # Day 1 Afternoon - Different research angle
            "I want to look at renewable energy adoption by major AI companies now. What's the current state?",
            
            "I found that Google claims 100% renewable energy matching for their data centers. But is that actually net-zero?",
            
            "Right, there's a difference between 'matched' and 'powered by'. They buy renewable credits but still use grid power with fossil fuels.",
            
            "What about Meta? What's their renewable energy percentage?",
            
            "I see Meta is at 75% renewable. AWS is only at 50%. Why such a big difference?",
            
            # Day 1 Evening - Synthesis question
            "Can you summarize what we've covered today? I need to write an outline for my paper.",
            
            "Perfect. Now, based on all the data we discussed - the 1-2% global electricity, the GPT-3 training emissions, the water usage - what's the single biggest environmental concern?",
            
            # Day 2 Morning - Follow-up on previous data
            "Hi! Remember we talked about data center water usage yesterday? I found more recent data.",
            
            "Meta's data centers in Arizona used 662 million gallons in 2022. That's during a drought. This seems particularly problematic, right?",
            
            "Can you compare that to Microsoft's 1.7 billion gallons we discussed yesterday? Which company is more water-efficient per user?",
            
            "Good analysis. Now, I want to add a section on future projections. What happens if AI usage grows 10x in the next 5 years?",
            
            "So if data centers are already at 1-2% of global electricity, and AI grows 10x, we could hit 10-20% of global electricity just for AI?",
            
            # Day 2 Afternoon - Policy implications
            "Let's shift to policy. What regulations exist for data center environmental impact?",
            
            "Are there any countries requiring data centers to use renewable energy?",
            
            "Denmark requires data centers to reuse waste heat. Can you explain how that works?",
            
            "That's clever. Has anyone else adopted this? What about Ireland, since they have a lot of data centers?",
            
            # Day 2 Evening - Counter-arguments
            "I need to address counter-arguments. What do tech companies say about their environmental impact?",
            
            "They argue AI helps climate research and optimization. Do we have examples of that?",
            
            "OK, so DeepMind reduced Google's cooling costs by 40% using AI. But does that offset the environmental cost of training and running the AI itself?",
            
            "This is the rebound effect, right? Efficiency gains lead to more usage?",
            
            # Day 3 - Final synthesis
            "I'm writing my conclusion. Can you remind me of the key statistics we gathered over the last two days?",
            
            "Perfect! Now, we discussed renewable energy adoption - Google at 100%, Meta at 75%, AWS at 50%. What was my concern about Google's claim?",
            
            "Right, the 'matching' vs 'powered by' distinction. For my conclusion, what's the most important policy recommendation based on everything we discussed?",
            
            "Excellent. Can you draft a brief conclusion paragraph incorporating the 1-2% electricity stat, the water usage concerns, and the renewable energy matching issue?",
            
            # Testing cross-session memory
            "Wait, I just realized - what was that specific number for GPT-3 training emissions? I need to cite it properly.",
            
            "And the Microsoft water usage figure - was it 1.7 billion or 1.7 million? I want to make sure I have it right.",
            
            "Last question - can you list all the companies we discussed and their renewable energy percentages? I need this for a comparison table.",
        ]
    
    def get_messages(self) -> List[str]:
        """Get all scenario messages"""
        return self.messages
    
    def get_num_turns(self) -> int:
        """Get number of turns in scenario"""
        return len(self.messages)


def get_scenario(name: str = "customer_support"):
    """Factory function to get scenarios"""
    scenarios = {
        "customer_support": "scenarios.customer_support.CustomerSupportScenario",
        "research_assistant": ResearchAssistantScenario,
    }
    
    if name not in scenarios:
        raise ValueError(f"Unknown scenario: {name}. Available: {list(scenarios.keys())}")
    
    if name == "customer_support":
        # Import dynamically to avoid circular dependency
        from scenarios.customer_support import CustomerSupportScenario
        return CustomerSupportScenario()
    
    return scenarios[name]()
