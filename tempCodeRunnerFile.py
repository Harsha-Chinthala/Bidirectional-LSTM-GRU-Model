@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')
@app.route('/demo')
def demo():
    """Render the demo page."""
    return render_template('demo.html')