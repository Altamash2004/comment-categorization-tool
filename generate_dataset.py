import pandas as pd
import random

# Define comment templates for each category
comment_templates = {
    'Praise': [
        "This is amazing work!",
        "Incredible content! Loved every second.",
        "Absolutely brilliant execution!",
        "Outstanding quality! Very impressed.",
        "Perfect! This exceeded expectations.",
        "Gold standard content right here.",
        "Phenomenal work! Keep it up.",
        "You nailed this! Masterpiece.",
        "Love this so much! Great job.",
        "Fire content! Absolutely stellar.",
        "Beautiful work! Very professional.",
        "Top-notch! Really love it.",
        "Stunning quality! So impressive.",
        "Fantastic job! You're talented.",
        "This is pure art! Well done.",
        "Superb! Keep creating more.",
        "Awesome! Can't stop replaying this.",
        "Perfect execution! Exactly what I wanted.",
        "Incredible talent on display here!",
        "Elite level content! Amazing.",
        "Magnificent! You outdid yourself.",
        "Brilliant! Hats off to you.",
        "Loved every bit! Great effort.",
        "Premium quality! Super impressive.",
        "Excellent! Very well executed.",
        "This is wonderful! Great creativity.",
        "Superb job! Really enjoyed this.",
        "Amazing! This made my day.",
        "Brilliant work! Very inspiring.",
        "Outstanding! This is top tier."
    ],
    
    'Support': [
        "Keep going, you're doing great!",
        "Don't give up! We believe in you.",
        "You've got this! Stay strong.",
        "Keep pushing! Your work matters.",
        "We're with you! Keep creating.",
        "Stay consistent! You'll succeed.",
        "Keep improving! You're on the right path.",
        "Don't stop now! You're making progress.",
        "Keep the momentum going! Proud of you.",
        "You inspire us! Keep it up.",
        "Rooting for you! Keep working hard.",
        "You're doing amazing! Don't doubt yourself.",
        "Keep shining! The world needs your content.",
        "Stay motivated! You're growing every day.",
        "Keep fighting! Your journey is inspiring.",
        "We support you! Keep moving forward.",
        "You're making a difference! Keep going.",
        "Stay focused! You're destined for greatness.",
        "Keep creating! Your voice matters.",
        "You got talent! Keep developing it.",
        "Believe in yourself! You're capable.",
        "Keep experimenting! You'll find your style.",
        "Stay authentic! That's your strength.",
        "Keep learning! You're improving fast.",
        "Never give up! Success is coming."
    ],
    
    'Constructive Criticism': [
        "The animation was okay but the voiceover felt off.",
        "Good effort, but the pacing could be improved.",
        "Nice concept, however the execution needs work.",
        "I appreciate the effort, but the audio quality is low.",
        "The idea is great, but the editing feels rushed.",
        "Well done overall, but the colors seem too bright.",
        "Good start, though the transitions could be smoother.",
        "Interesting content, but it's a bit too long.",
        "The message is clear, but the visuals are distracting.",
        "I like it, but the background music is too loud.",
        "Great concept, but the text is hard to read.",
        "Nice try, but the timing feels off in some places.",
        "Good work, however the intro is too slow.",
        "Solid effort, but the thumbnail could be better.",
        "I enjoyed it, but some parts felt repetitive.",
        "The content is good, but the delivery needs polish.",
        "Appreciate the work, but the lighting is inconsistent.",
        "Good job, though the script could be tighter.",
        "Nice idea, but the resolution is a bit low.",
        "I see potential, but the pacing drags in the middle.",
        "Decent work, but the tone doesn't match the content.",
        "Good attempt, but the voiceover lacks energy.",
        "I like the direction, but it needs more examples.",
        "Well researched, but the presentation could improve.",
        "Good message, but the format is a bit confusing."
    ],
    
    'Hate': [
        "This is trash, quit now.",
        "Worst thing I've ever seen. Delete this.",
        "You have no talent. Stop wasting our time.",
        "Absolute garbage. Why even bother?",
        "This is pathetic. You're embarrassing yourself.",
        "Terrible content. You should be ashamed.",
        "Complete waste of time. Total failure.",
        "You're a joke. Nobody cares about this.",
        "This is horrible. Delete your account.",
        "Pathetic attempt. You're useless.",
        "Disgusting work. Stop creating.",
        "You're terrible at this. Give up already.",
        "This is the worst. Pure cringe.",
        "Awful. You have zero skills.",
        "Trash content from a trash creator.",
        "You're so bad at this it hurts to watch.",
        "This is stupid. Total waste of space.",
        "Embarrassing. You should quit.",
        "Horrible. Please stop posting.",
        "You're talentless. This is unwatchable.",
        "Pure garbage. No redeeming qualities.",
        "This is insulting to viewers. Terrible.",
        "Absolutely worthless content.",
        "You're a failure. Accept it.",
        "Atrocious work. Stop now."
    ],
    
    'Threat': [
        "I will report your account immediately.",
        "You better remove this or face consequences.",
        "I'm going to sue you for this content.",
        "Remove this now or I'll take legal action.",
        "I'll make sure you get banned permanently.",
        "Take this down or I'm reporting to authorities.",
        "You will regret posting this, mark my words.",
        "I'm contacting your sponsors about this issue.",
        "I'll get you banned from this platform.",
        "This violates laws. Reporting to authorities now.",
        "Better stop or you'll face serious consequences.",
        "I'm forwarding this to my legal team today.",
        "I will ensure everyone knows about this.",
        "Remove it immediately or face the consequences.",
        "You're in serious trouble for this.",
        "I'm filing an official complaint against you.",
        "Better watch what you post going forward.",
        "Reporting this to platform moderators now.",
        "Legal action will follow if not removed.",
        "I'll ensure you face the consequences.",
        "I will not let this slide. Reporting now.",
        "You've crossed a line. Expect repercussions.",
        "Taking this to the authorities immediately.",
        "I'm documenting this for legal purposes.",
        "You better take this seriously. I'm not joking.",
        "Expect a cease and desist letter soon.",
        "I'm escalating this to the legal department.",
        "You will hear from my lawyer about this.",
        "This is unacceptable. Authorities will be notified.",
        "I'm reporting this violation right now."
    ],
    
    'Emotional': [
        "This reminded me of my childhood.",
        "I'm crying. This hit me so hard.",
        "This brings back so many memories.",
        "I felt this in my soul. Thank you.",
        "This made me emotional. Beautiful.",
        "I'm not crying, you're crying.",
        "This touched my heart deeply.",
        "I'm feeling all the feels right now.",
        "This resonates with me on so many levels.",
        "I needed to see this today. Thank you.",
        "This made me tear up. So powerful.",
        "I'm overwhelmed with emotions watching this.",
        "This speaks to my experience perfectly.",
        "I felt every word of this.",
        "This is so relatable it hurts.",
        "I'm having flashbacks. This is intense.",
        "This made me think of my grandmother.",
        "I'm sobbing. This is too real.",
        "This captures exactly how I feel.",
        "I'm emotionally moved by this.",
        "This brought me to tears. Beautiful.",
        "I feel seen. Thank you for this.",
        "This is healing for me. Grateful.",
        "I'm shaking. This is powerful.",
        "This reminds me of better times."
    ],
    
    'Spam': [
        "Follow me for followers!",
        "Check out my channel for awesome content!",
        "Click the link in my bio for free stuff!",
        "Sub4sub? Let me know!",
        "Visit my profile for amazing deals!",
        "F O L L O W M E B A C K",
        "Want free followers? DM me now!",
        "Check out my latest post! Link in bio!",
        "Promoting my new channel, please support!",
        "Free giveaway on my page! Check it out!",
        "Like for like? Comment below!",
        "Subscribe to my channel and I'll sub back!",
        "Click here for exclusive content!",
        "Follow for follow? Let's grow together!",
        "Visit my website for amazing products!",
        "Shoutout for shoutout? Anyone?",
        "Check my profile for cool stuff!",
        "Drop a follow and I'll follow back!",
        "Promoting my new business! Check it out!",
        "Free ebook in my bio! Get it now!",
        "Subscribe and turn on notifications!",
        "First person to follow gets a shoutout!",
        "Link in bio for exclusive deals!",
        "Everyone follow me! I follow back 100%!",
        "Visit my page for daily motivation!"
    ],
    
    'Question': [
        "Can you make one on topic X?",
        "What software did you use for this?",
        "How long did this take to create?",
        "Could you do a tutorial on this?",
        "Where can I learn to do this?",
        "What camera do you use?",
        "Can you explain your process?",
        "Do you offer any courses?",
        "How did you learn this skill?",
        "What's your editing workflow?",
        "Could you make a behind-the-scenes video?",
        "What tools do you recommend for beginners?",
        "How do you come up with these ideas?",
        "Can I use this in my project?",
        "Where did you get the music from?",
        "What inspired you to make this?",
        "Do you have any tips for beginners?",
        "How much does equipment like this cost?",
        "Can you do a collaboration?",
        "What's your upload schedule?",
        "How do you stay consistent?",
        "Do you have a Discord server?",
        "Can you share your settings?",
        "What's the best way to improve?",
        "Where can I find more content like this?"
    ]
}

# Generate the dataset
data = []
for category, comments in comment_templates.items():
    for comment in comments:
        data.append({
            'comment': comment,
            'category': category
        })

# Create DataFrame
df = pd.DataFrame(data)

# Augment the dataset by adding variations
# This will double the dataset size with slight variations
augmented_data = []
for _, row in df.iterrows():
    comment = row['comment']
    category = row['category']
    
    # Add original
    augmented_data.append({'comment': comment, 'category': category})
    
    # Add variations (simple augmentation)
    if random.random() > 0.5:  # 50% chance to augment
        # Add some variations
        variations = [
            comment + " 👍",
            comment + "!",
            comment.replace("!", "."),
            comment.lower(),
        ]
        augmented_data.append({
            'comment': random.choice(variations),
            'category': category
        })

df = pd.DataFrame(augmented_data)

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to CSV
df.to_csv('comments_dataset.csv', index=False)

print(f"Dataset created successfully!")
print(f"Total comments: {len(df)}")
print(f"\nCategory distribution:")
print(df['category'].value_counts())
print(f"\nDataset saved as 'comments_dataset.csv'")
print(f"\nFirst 5 rows:")
print(df.head())