# real_pegasos_app.py
import pickle
import numpy as np
import gradio as gr

print("🤖 Loading REAL Pegasos algorithm...")

# Load your compact model
try:
    with open('pegasos_model_compact.pkl', 'rb') as f:
        model = pickle.load(f)
    
    theta = model['theta']
    theta_0 = model['theta_0']
    dictionary = model['dictionary']
    
    print(f"✅ Loaded Pegasos model with {len(dictionary)} words")
    print(f"   Top positive word: {model['top_positive'][0]}")
    print(f"   Model accuracy: {model['accuracy']:.1%}")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    # Fallback to simple version
    theta, theta_0, dictionary = None, None, {}

def real_pegasos_predict(review_text):
    """Use your ACTUAL Pegasos algorithm for prediction"""
    if theta is None:
        return "Model not loaded properly"
    
    # 1. Convert text to features (same as your project)
    features = np.zeros(len(theta), dtype=np.float32)
    
    words = review_text.lower().split()
    for word in words:
        if word in dictionary:
            word_id = dictionary[word]
            features[word_id] = 1.0  # Binary features
    
    # 2. Compute Pegasos prediction: sign(theta·x + theta_0)
    score = np.dot(features, theta) + theta_0
    
    # 3. Determine sentiment
    if score > 0:
        confidence = min(abs(score) / 5.0, 0.99)
        sentiment = "😊 POSITIVE"
    else:
        confidence = min(abs(score) / 5.0, 0.99)
        sentiment = "😠 NEGATIVE"
    
    # 4. Find influential words in this review
    influential_words = []
    for word in words:
        if word in dictionary:
            word_id = dictionary[word]
            weight = theta[word_id]
            if abs(weight) > 0.5:  # Only show strong indicators
                influential_words.append(f"{word}({weight:+.2f})")
    
    result = f"{sentiment}\n"
    result += f"Confidence: {confidence:.0%}\n"
    result += f"Pegasos Score: {score:.3f}\n"
    if influential_words:
        result += f"Influential words: {', '.join(influential_words[:5])}"
    
    return result

def compare_with_simple(review_text):
    """Show how REAL Pegasos differs from simple keyword matching"""
    pegasos_result = real_pegasos_predict(review_text)
    
    # Simple keyword version (for comparison)
    simple_pos = ["delicious", "great", "best", "perfect", "love", "amazing"]
    simple_neg = ["terrible", "awful", "bad", "horrible", "disappointed"]
    
    review_lower = review_text.lower()
    simple_score = sum(1 for w in simple_pos if w in review_lower) - sum(1 for w in simple_neg if w in review_lower)
    
    simple_result = "😊 POSITIVE" if simple_score > 0 else "😠 NEGATIVE" if simple_score < 0 else "🤔 NEUTRAL"
    
    comparison = f"**REAL Pegasos:** {pegasos_result}\n\n"
    comparison += f"**Simple Keywords:** {simple_result} (score: {simple_score})"
    comparison += f"\n\n💡 Pegasos considers {len(dictionary)} words, not just {len(simple_pos)+len(simple_neg)}"
    
    return comparison

# Create the Gradio interface
demo = gr.Interface(
    fn=compare_with_simple,  # Use comparison function
    inputs=gr.Textbox(
        label="🍽️ Enter Amazon Food Review",
        placeholder="Try: 'This chocolate is absolutely delicious! Best purchase ever!'",
        lines=3
    ),
    outputs=gr.Markdown(label="🤖 Analysis Results"),
    title="Amazon Reviews - REAL Pegasos Algorithm vs Simple Keywords",
    description=f"""
    ## This is my ACTUAL machine learning model!
    
    **Model Details:**
    - Algorithm: Pegasos (implemented from scratch in project1.py)
    - Vocabulary: {len(dictionary) if 'dictionary' in locals() else 'Loading...'} words
    - Test Accuracy: 80.8%
    - Training Data: 5,000+ Amazon food reviews
    
    **How it works:**
    1. Converts text to numerical features (bag-of-words)
    2. Computes: sign(θ·x + θ₀) where θ is learned from data
    3. Returns sentiment + confidence score
    
    **Top words my model learned:**
    - 😊 Most Positive: {model['top_positive'][0] if 'model' in locals() else 'delicious'}, {model['top_positive'][1] if 'model' in locals() else 'great'}, {model['top_positive'][2] if 'model' in locals() else '!'}
    - 😠 Most Negative: {model['top_negative'][-1] if 'model' in locals() else 'terrible'}, {model['top_negative'][-2] if 'model' in locals() else 'awful'}
    """,
    examples=[
        ["This cereal is amazing! My kids love it every morning."],
        ["Terrible quality. Cookies arrived stale and broken."],
        ["Perfect chocolate chip cookies! Will definitely buy again!"],
        ["The product was okay, nothing special but gets the job done."],
        ["Worst granola bars I've ever tasted. Dry and flavorless."]
    ]
)

if __name__ == "__main__":
    demo.launch(share=False)