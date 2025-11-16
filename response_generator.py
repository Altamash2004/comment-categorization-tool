import random

class ResponseGenerator:
    """
    Generate appropriate response templates for each comment category
    Helps teams respond efficiently and empathetically
    """
    
    def __init__(self):
        self.templates = {
            'Praise': {
                'action': '✅ Engage Positively',
                'priority': 'High',
                'suggested_responses': [
                    "Thank you so much! Your support means the world to us! 🙏",
                    "We're thrilled you loved it! More content coming soon! ❤️",
                    "Your kind words made our day! Thank you for the encouragement! ✨",
                    "So glad you enjoyed it! Stay tuned for more! 🎉",
                    "Thank you for the amazing feedback! We appreciate you! 💫"
                ],
                'tips': 'Respond quickly to build community. Use emojis to show warmth.'
            },
            
            'Support': {
                'action': '✅ Acknowledge & Thank',
                'priority': 'High',
                'suggested_responses': [
                    "Your support keeps us going! Thank you for believing in us! 💪",
                    "Comments like these motivate us to keep creating. Thank you! 🙏",
                    "We appreciate you being part of our journey! Thank you! ❤️",
                    "Your encouragement means everything! Thank you for sticking with us! 🌟",
                    "Thank you for the motivation! We won't let you down! 🚀"
                ],
                'tips': 'Show gratitude. These are your loyal supporters - nurture them.'
            },
            
            'Constructive Criticism': {
                'action': '✅ Address Thoughtfully',
                'priority': 'Very High',
                'suggested_responses': [
                    "Thank you for the honest feedback! We'll definitely work on improving that. 🙏",
                    "Great point! We appreciate constructive feedback and will consider this for future content. ✨",
                    "Thanks for taking the time to share your thoughts! We're always looking to improve. 💪",
                    "We hear you! This feedback helps us grow. Thanks for being specific! 🎯",
                    "Appreciate the critique! We'll keep this in mind for our next project. 📝"
                ],
                'tips': 'PRIORITY: These comments are valuable! Respond professionally. Show you value improvement. Never be defensive.'
            },
            
            'Hate': {
                'action': '🚫 Ignore or Delete',
                'priority': 'Low (Monitor)',
                'suggested_responses': [
                    "[Do not engage. Delete if violates community guidelines.]",
                    "[Block user if harassment continues.]",
                    "[Report to platform if contains threats or extreme abuse.]"
                ],
                'tips': 'Do not feed trolls. Delete if offensive. Report severe cases. Never respond emotionally.'
            },
            
            'Threat': {
                'action': '⚠️ Report & Escalate',
                'priority': 'Critical',
                'suggested_responses': [
                    "[REPORT IMMEDIATELY to platform.]",
                    "[Document and save evidence.]",
                    "[Contact platform support team.]",
                    "[If serious, contact legal counsel or authorities.]"
                ],
                'tips': 'CRITICAL: Take all threats seriously. Document everything. Report to authorities if needed.'
            },
            
            'Emotional': {
                'action': '✅ Respond with Empathy',
                'priority': 'High',
                'suggested_responses': [
                    "Thank you for sharing this with us. We're so glad this resonated with you. ❤️",
                    "Your comment touched our hearts. Thank you for being vulnerable. 🙏",
                    "We're honored that this connected with you on a deep level. 💫",
                    "Thank you for letting us be part of your story. Sending positive vibes! ✨",
                    "Comments like these remind us why we create. Thank you for sharing. 💙"
                ],
                'tips': 'Be empathetic and genuine. These are deep connections - handle with care.'
            },
            
            'Spam': {
                'action': '🚫 Delete or Ignore',
                'priority': 'Very Low',
                'suggested_responses': [
                    "[Delete immediately.]",
                    "[Mark as spam.]",
                    "[Do not engage or respond.]"
                ],
                'tips': 'Never engage. Delete to keep comment section clean. Report repeat offenders.'
            },
            
            'Question': {
                'action': '✅ Provide Helpful Answer',
                'priority': 'High',
                'suggested_responses': [
                    "Great question! [Provide specific answer]. Hope this helps! 🙂",
                    "Thanks for asking! [Give detailed response]. Let us know if you need more info! 💡",
                    "Good one! [Answer]. Feel free to ask if you have more questions! ✨",
                    "We'd be happy to share! [Provide answer]. Check out [link/resource] for more details! 🎯",
                    "Excellent question! [Response]. We might make a detailed video on this! 📹"
                ],
                'tips': 'Answer promptly and thoroughly. These users are engaged. Consider making content from popular questions.'
            }
        }
    
    def get_response_template(self, category):
        """
        Get response template for a specific category
        
        Args:
            category: The comment category
        
        Returns:
            Dictionary with action, priority, responses, and tips
        """
        return self.templates.get(category, {
            'action': '⚠️ Review Manually',
            'priority': 'Medium',
            'suggested_responses': ['Please review this comment manually.'],
            'tips': 'Unknown category - handle with care.'
        })
    
    def get_random_response(self, category):
        """Get a random response suggestion for the category"""
        template = self.get_response_template(category)
        return random.choice(template['suggested_responses'])
    
    def get_action_plan(self, category):
        """Get recommended action for the category"""
        template = self.get_response_template(category)
        return template['action']
    
    def get_priority(self, category):
        """Get priority level for the category"""
        template = self.get_response_template(category)
        return template['priority']
    
    def generate_response_guide(self, category, comment):
        """
        Generate a complete response guide for a comment
        
        Args:
            category: Predicted category
            comment: Original comment text
        
        Returns:
            Dictionary with complete response guidance
        """
        template = self.get_response_template(category)
        
        return {
            'original_comment': comment,
            'category': category,
            'action': template['action'],
            'priority': template['priority'],
            'suggested_responses': template['suggested_responses'],
            'tips': template['tips']
        }
    
    def print_response_guide(self, guide):
        """Pretty print a response guide"""
        print(f"\n{'='*70}")
        print(f"COMMENT: {guide['original_comment']}")
        print(f"{'='*70}")
        print(f"Category: {guide['category']}")
        print(f"Action: {guide['action']}")
        print(f"Priority: {guide['priority']}")
        print(f"\nSuggested Responses:")
        for i, response in enumerate(guide['suggested_responses'], 1):
            print(f"  {i}. {response}")
        print(f"\n💡 Tips: {guide['tips']}")
        print(f"{'='*70}")


# Example usage
if __name__ == "__main__":
    generator = ResponseGenerator()
    
    # Test examples
    test_cases = [
        ("Praise", "Amazing work! Loved the animation."),
        ("Constructive Criticism", "The animation was okay but the voiceover felt off."),
        ("Hate", "This is trash, quit now."),
        ("Question", "Can you make one on Python programming?"),
        ("Emotional", "This reminded me of my childhood."),
        ("Threat", "I'll report you if this continues.")
    ]
    
    print("RESPONSE TEMPLATE GENERATOR - EXAMPLES")
    
    for category, comment in test_cases:
        guide = generator.generate_response_guide(category, comment)
        generator.print_response_guide(guide)