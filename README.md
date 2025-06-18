# 🚗 Auto Price Prediction Web Application

A modern, interactive web application for predicting automobile prices using machine learning algorithms. Built with Node.js, Express, and vanilla JavaScript.

## 🌟 Features

### 🎯 Core Functionality
- **Real-time Price Prediction**: Input car features and get instant price predictions
- **Multiple ML Models**: Compare predictions from XGBoost, Random Forest, KNN, and Linear Regression
- **Interactive UI**: Modern, responsive design with smooth animations
- **Data Visualization**: Interactive charts showing model performance and feature importance
- **Export Functionality**: Download prediction results as CSV files

### 📊 Analytics Dashboard
- Model performance comparison charts
- Feature importance visualization
- Detailed metrics table with R², MAE, and RMSE scores
- Statistical insights and model explanations

### 🎨 User Experience
- Clean, professional interface
- Mobile-responsive design
- Intuitive navigation with tabbed interface
- Real-time form validation
- Loading states and error handling

## 🚀 Quick Start

### Prerequisites
- Node.js (v14 or higher)
- npm or yarn

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd auto-price-prediction
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start the application**
   ```bash
   npm start
   ```

4. **Open your browser**
   Navigate to `http://localhost:3000`

## 📁 Project Structure

```
auto-price-prediction/
├── server.js              # Express server and API endpoints
├── package.json           # Dependencies and scripts
├── public/                # Frontend assets
│   ├── index.html         # Main HTML file
│   ├── styles.css         # CSS styles
│   └── script.js          # Frontend JavaScript
├── data/                  # Dataset files
│   └── auto_imports.csv   # Automobile dataset
├── uploads/               # Temporary file uploads
└── README.md             # This file
```

## 🔧 API Endpoints

### GET `/api/stats`
Returns model performance statistics and metrics.

### GET `/api/data-summary`
Returns dataset summary including unique values for categorical fields.

### POST `/api/predict`
Predicts car price based on input features.

**Request Body:**
```json
{
  "make": "toyota",
  "fuel_type": "gas",
  "engine_size": 130,
  "horsepower": 111,
  "curb_weight": 2548,
  // ... other features
}
```

**Response:**
```json
{
  "success": true,
  "prediction": 15420,
  "confidence": 0.87,
  "model": "XGBoost"
}
```

### POST `/api/upload`
Handles CSV file uploads for batch processing.

## 🎯 Model Performance

| Model | R² Score | Adjusted R² | Status |
|-------|----------|-------------|---------|
| **XGBoost** | **89.37%** | **67.23%** | ✅ **Best Model** |
| Random Forest | 87.73% | 62.16% | ✅ Good |
| KNN | 87.30% | 60.84% | ✅ Good |
| Linear Regression | 78.45% | 52.34% | ⚠️ Baseline |

## 🔍 Features Used for Prediction

### Categorical Features
- **Brand/Make**: alfa-romero, audi, bmw, chevrolet, etc.
- **Fuel Type**: Gas, Diesel
- **Aspiration**: Standard, Turbo
- **Body Style**: Sedan, Hatchback, Wagon, Convertible, etc.
- **Drive Wheels**: FWD, RWD, 4WD
- **Number of Doors**: Two, Four
- **Number of Cylinders**: Two, Three, Four, Five, Six, Eight, Twelve

### Numerical Features
- **Engine Specifications**: Engine Size, Horsepower
- **Physical Dimensions**: Length, Width, Curb Weight
- **Performance Metrics**: City MPG, Highway MPG
- **Insurance Risk Rating**: Symboling (-3 to +3)

## 🎨 Technology Stack

### Backend
- **Node.js**: Runtime environment
- **Express.js**: Web framework
- **Multer**: File upload handling
- **CSV-Parser**: CSV file processing

### Frontend
- **Vanilla JavaScript**: Core functionality
- **Chart.js**: Data visualization
- **CSS3**: Modern styling with animations
- **HTML5**: Semantic markup

### Machine Learning
- **Prediction Algorithm**: Simplified XGBoost-inspired logic
- **Feature Engineering**: Weighted feature importance
- **Brand Multipliers**: Market-based pricing adjustments

## 🚀 Deployment Options

### Local Development
```bash
npm run dev
```

### Production Deployment

#### Option 1: Heroku
1. Create a Heroku app
2. Set environment variables
3. Deploy using Git

#### Option 2: Railway
1. Connect your GitHub repository
2. Configure build settings
3. Deploy automatically

#### Option 3: DigitalOcean App Platform
1. Create a new app
2. Connect repository
3. Configure environment

#### Option 4: Docker
```dockerfile
FROM node:16-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
EXPOSE 3000
CMD ["npm", "start"]
```

## 📊 Dataset Information

- **Source**: 1985 Auto Imports Database
- **Records**: 205 automobile entries
- **Features**: 25 independent variables + 1 target (price)
- **Data Types**: Mix of categorical and numerical features
- **Price Range**: $5,118 - $45,400

## 🔮 Future Enhancements

- [ ] **Real-time Market Data Integration**
- [ ] **Advanced ML Models** (Neural Networks, Ensemble Methods)
- [ ] **User Authentication** and saved predictions
- [ ] **Batch Processing** for multiple predictions
- [ ] **API Rate Limiting** and caching
- [ ] **Mobile App** development
- [ ] **Advanced Analytics** dashboard

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Project Team

**Project Team ID**: PTID-CDS-FEB-25-2024  
**Project Code**: PRCP-1017-AutoPricePred

## 📞 Support

For questions, suggestions, or issues:
- Create an issue on GitHub
- Contact the development team
- Check the documentation

---

**Built with ❤️ for accurate automobile price predictions**