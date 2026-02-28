financial-pulse-ai : Financial Anomaly Detection Suite

AI Market Sentinel is a high-performance quantitative analysis dashboard built to detect irregularities in financial markets. By combining Unsupervised Machine Learning (Isolation Forest) with Classical Statistics (Z-Score), the system provides a multi-layered "Sentinel" approach to risk monitoring.

🚀 Key Features:

Hybrid Anomaly Detection: Integrated logic that identifies patterns using AI while validating spikes via standard deviations.

Stochastic Forecasting: A Monte Carlo simulation engine generating 100 possible price paths based on historical volatility.

Market Correlation Matrix: Heatmap analysis comparing the selected asset against benchmarks like the S&P 500 (^GSPC), Gold (GC=F), and Bitcoin (BTC-USD).
S
trategy Backtester: Simulates the ROI of a "Buy-on-Anomaly" strategy to verify if AI signals correlate with profitable entry points.

Real-time Risk Gauge: A sidebar-integrated status monitor providing instant "Low/Medium/High" risk ratings based on live data.

🛠️ Technical Architecture:

The project is modularized into three core layers:
Data Layer (src/processor.py): Handles API ingestion from Yahoo Finance. It includes robust error handling and standardizes data frames (dropping NaN values) to ensure mathematical consistency.
Logic Engine (src/engine.py):
AI Layer: Uses sklearn.ensemble.IsolationForest to detect outliers in high-dimensional space (Price + Volume).
Statistical Layer: Calculates Rolling Z-Scores over a 20-period window to identify extreme price deviations.
Hybrid Logic: Flags an anomaly only if the AI model or the Z-Score threshold ($|Z| > 3$) is triggered
Presentation Layer (app.py): A Streamlit-based Web UI using Plotly for high-fidelity interactive charting.

📦 Dependencies:

To run this project, the following libraries are required:
streamlit: For the web dashboard interface.yfinance: To fetch real-time market data.

pandas & numpy: For complex data manipulation and array math.plotly: For interactive financial charts and heatmaps.

scikit-learn: Specifically the IsolationForest model for unsupervised ML.

🔧Installation & SetupClone the repository:git clone https://github.com/nishmitharamin/financial-pulse-ai.git
mkdir financial_pulse
cd financial_pulse
python -m venv venv
pip install streamlit pandas yfinance scikit-learn plotly fastapi uvicorn
Install dependencies:pip install -r requirements.txt
Run the Application:streamlit run app.py

📊 Scientific Methodology:

1. The Anomaly Detection FormulaThe system defines an anomaly through a hybrid approach:
   Unsupervised ML: The Isolation Forest "isolates" observations by randomly selecting a feature and a split value. Outliers are those that require fewer splits to isolate.
   Z-Score: Calculated as:$$Z = \frac{x - \mu}{\sigma}$$Where $x$ is the price, $\mu$ is the rolling mean, and $\sigma$ is the rolling standard deviation.
2. Monte Carlo SimulationThe projection tool uses Geometric Brownian Motion (GBM) logic to simulate 30 days of future price action based on the daily volatility ($\sigma$) of the selected lookback window.

🧪 Testing:
The project includes automated unit tests for the engine logic.
Run tests to ensure the engine is outputting correct data shapes
py test src/test_engine.py

👨‍💻 Project Structure:

financial-pulse-ai/
├── .github/
│ └── workflows/  
│ └── streamlit_app.yml
├── data/  
│ └── sample_market.csv
├── src/  
│ ├── **init**.py
│ ├── engine.py  
│ ├── utils.py  
│ └── processor.py  
├── assets/  
│ └── logo.png
├── tests/  
│ └── test_engine.py
├── app.py  
├── requirements.txt  
├── README.md  
└── .gitignore
