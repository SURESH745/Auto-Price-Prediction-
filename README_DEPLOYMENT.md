# Auto Price Prediction - Deployment Guide

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd auto-price-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser**
   Navigate to `http://localhost:8501`

## 🌐 Deployment Options

### Option 1: Streamlit Cloud (Recommended)

1. **Push your code to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file path: `app.py`
   - Click "Deploy"

3. **Your app will be live at**: `https://your-app-name.streamlit.app`

### Option 2: Render

1. **Create account on Render.com**

2. **Create new Web Service**
   - Connect your GitHub repository
   - Set build command: `pip install -r requirements.txt`
   - Set start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

3. **Environment Variables** (if needed)
   - Add any required environment variables in Render dashboard

### Option 3: Heroku

1. **Install Heroku CLI**

2. **Create Heroku app**
   ```bash
   heroku create your-app-name
   ```

3. **Create Procfile**
   ```
   web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
   ```

4. **Deploy**
   ```bash
   git add .
   git commit -m "Deploy to Heroku"
   git push heroku main
   ```

### Option 4: Docker Deployment

1. **Create Dockerfile**
   ```dockerfile
   FROM python:3.9-slim

   WORKDIR /app

   COPY requirements.txt .
   RUN pip install -r requirements.txt

   COPY . .

   EXPOSE 8501

   CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
   ```

2. **Build and run**
   ```bash
   docker build -t auto-price-prediction .
   docker run -p 8501:8501 auto-price-prediction
   ```

## 📁 Project Structure

```
auto-price-prediction/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── Data/
│   └── auto_imports.csv  # Dataset
├── README.md             # Project documentation
├── README_DEPLOYMENT.md  # This deployment guide
└── .gitignore           # Git ignore file
```

## 🔧 Configuration

### Environment Variables (Optional)
If you need to add environment variables, create a `.env` file:
```
DEBUG=False
MODEL_PATH=models/
```

### Streamlit Configuration
Create `.streamlit/config.toml` for custom settings:
```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"

[server]
maxUploadSize = 200
```

## 🐛 Troubleshooting

### Common Issues:

1. **Import Errors**
   - Ensure all dependencies are in `requirements.txt`
   - Check Python version compatibility

2. **Data Loading Issues**
   - Verify data file path is correct
   - Check file permissions

3. **Memory Issues**
   - Consider using `@st.cache_data` for large datasets
   - Optimize data loading and processing

4. **Port Issues (Local)**
   - Default port is 8501
   - Use `--server.port` flag to change port

### Performance Optimization:

1. **Use Caching**
   ```python
   @st.cache_data
   def load_data():
       # Your data loading code
   ```

2. **Optimize Imports**
   - Import only necessary libraries
   - Use lazy imports where possible

3. **Data Preprocessing**
   - Cache preprocessed data
   - Use efficient data structures

## 📊 Monitoring & Analytics

### Streamlit Cloud Analytics
- View app usage statistics in Streamlit Cloud dashboard
- Monitor performance and errors

### Custom Analytics (Optional)
Add Google Analytics or other tracking:
```python
# Add to app.py
st.markdown("""
<script async src="https://www.googletagmanager.com/gtag/js?id=GA_MEASUREMENT_ID"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'GA_MEASUREMENT_ID');
</script>
""", unsafe_allow_html=True)
```

## 🔒 Security Considerations

1. **Environment Variables**
   - Never commit sensitive data to Git
   - Use environment variables for API keys

2. **Input Validation**
   - Validate user inputs
   - Sanitize file uploads

3. **HTTPS**
   - Most platforms provide HTTPS by default
   - Ensure secure connections

## 📈 Scaling

### For High Traffic:
1. **Use Professional Hosting**
   - Consider AWS, GCP, or Azure
   - Use load balancers

2. **Database Integration**
   - Move from CSV to database
   - Use connection pooling

3. **Caching Strategy**
   - Implement Redis for caching
   - Use CDN for static assets

## 🆘 Support

If you encounter issues:
1. Check the [Streamlit Documentation](https://docs.streamlit.io)
2. Review deployment platform specific guides
3. Check GitHub Issues for common problems
4. Contact the development team

---

**Happy Deploying! 🚀**