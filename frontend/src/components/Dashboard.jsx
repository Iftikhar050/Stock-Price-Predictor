import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Select from 'react-select';
import { API_BASE_URL } from '../config';
import { ComposedChart, Area, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';

const MarketPerformers = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchPerformers = async () => {
      try {
        const res = await axios.get(`${API_BASE_URL}/api/market_performers`);
        setData(res.data);
      } catch (err) {
        console.error("Failed to fetch market performers", err);
      } finally {
        setLoading(false);
      }
    };
    fetchPerformers();
    const interval = setInterval(fetchPerformers, 30000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return <div className="animate-pulse bg-slate-800 h-64 rounded-xl border border-slate-700 mb-6"></div>;
  }

  if (!data) return null;

  const renderTable = (title, items) => (
    <div className="w-full">
      <h3 className="text-[13px] font-bold text-slate-300 uppercase tracking-wider mb-2">{title}</h3>
      <div className="overflow-hidden border border-slate-700 rounded-lg">
        <table className="w-full text-left text-xs">
          <thead className="bg-slate-900/50 text-slate-300 border-b border-slate-700">
            <tr>
              <th className="px-3 py-2 font-bold">SYMBOL</th>
              <th className="px-3 py-2 font-bold text-right">PRICE</th>
              <th className="px-3 py-2 font-bold text-right">CHANGE</th>
              <th className="px-3 py-2 font-bold text-right">VOLUME</th>
            </tr>
          </thead>
          <tbody className="bg-slate-800">
            {items.map((item, idx) => (
              <tr key={idx} className="border-b border-slate-700/50 last:border-0 hover:bg-slate-700/50 transition-colors text-slate-200 font-bold">
                <td className="px-3 py-2.5">{item.symbol}</td>
                <td className="px-3 py-2.5 text-right font-normal">{item.price.toFixed(2)}</td>
                <td className={`px-3 py-2.5 text-right flex items-center justify-end ${item.change >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                  {item.change >= 0 ? <span className="mr-1 text-[10px]">▲</span> : <span className="mr-1 text-[10px]">▼</span>}
                  {Math.abs(item.change).toFixed(2)} ({item.change_percent >= 0 ? '+' : ''}{item.change_percent.toFixed(2)}%)
                </td>
                <td className="px-3 py-2.5 text-right font-normal text-slate-400">{item.volume.toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );

  return (
    <div className="bg-slate-800 p-6 rounded-xl shadow-lg border border-slate-700 mb-6">
      <h2 className="text-xl font-bold text-slate-100 mb-6 border-b border-slate-700 pb-3">Market Performers</h2>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
        {renderTable('TOP ACTIVE STOCKS', data.top_active)}
        {renderTable('TOP ADVANCERS', data.top_advancers)}
        {renderTable('TOP DECLINERS', data.top_decliners)}
      </div>
    </div>
  );
};

const AllCompaniesTable = ({ modelType, availableTickers }) => {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!availableTickers || availableTickers.length === 0) return;
    const fetchAll = async () => {
      setLoading(true);
      try {
        const promises = availableTickers.map(t => axios.post(`${API_BASE_URL}/api/predict`, { ticker: t.value }));
        const responses = await Promise.all(promises);
        const parsedData = responses.map(res => {
          const d = res.data;
          const lastRecord = d.historical_data[d.historical_data.length - 1];
          const prefix = modelType === 'RF' ? 'rf' : modelType === 'LSTM' ? 'lstm' : modelType === 'LR' ? 'lr' : 'xgb';
          const predToday = lastRecord[`${prefix}_pred`];
          const actualToday = d.current_price;
          const diff = predToday ? actualToday - predToday : 0;
          const diffPercent = predToday ? (diff / predToday) * 100 : 0;
          const predTomorrow = d[`${prefix}_predicted_price`];
          
          return {
            ticker: d.ticker,
            actualToday,
            predToday,
            diff,
            diffPercent,
            predTomorrow
          };
        });
        setData(parsedData);
      } catch (err) {
        console.error("Failed to fetch all predictions", err);
      } finally {
        setLoading(false);
      }
    };
    
    fetchAll();
    const interval = setInterval(fetchAll, 60000);
    return () => clearInterval(interval);
  }, [modelType, availableTickers]);

  if (loading && data.length === 0) {
    return <div className="bg-slate-800 p-6 rounded-xl shadow-lg border border-slate-700 h-64 animate-pulse mt-6"></div>;
  }

  return (
    <div className="bg-slate-800 p-6 rounded-xl shadow-lg border border-slate-700 mt-6">
      <h3 className="text-xl font-bold text-slate-100 mb-4 border-b border-slate-700 pb-3 flex items-center">
        <svg className="w-5 h-5 mr-2 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 10h18M3 14h18m-9-4v8m-7 0h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"></path>
        </svg>
        Model Performance ({modelType})
      </h3>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead className="bg-slate-900/50 text-slate-300 border-b border-slate-700">
            <tr>
              <th className="px-4 py-3 font-bold uppercase text-[11px] tracking-wider">Company</th>
              <th className="px-4 py-3 font-bold text-right uppercase text-[11px] tracking-wider">Actual Today</th>
              <th className="px-4 py-3 font-bold text-right uppercase text-[11px] tracking-wider">Pred Today</th>
              <th className="px-4 py-3 font-bold text-right uppercase text-[11px] tracking-wider">Error (Act - Pred)</th>
              <th className="px-4 py-3 font-bold text-right text-emerald-400 uppercase text-[11px] tracking-wider">Pred Tomorrow</th>
            </tr>
          </thead>
          <tbody className="bg-slate-800">
            {data.map((row, idx) => (
              <tr key={idx} className="border-b border-slate-700/50 last:border-0 hover:bg-slate-700/50 transition-colors text-slate-200 font-medium">
                <td className="px-4 py-3 font-bold">{row.ticker}</td>
                <td className="px-4 py-3 text-right">Rs. {row.actualToday.toFixed(2)}</td>
                <td className="px-4 py-3 text-right">
                  {row.predToday ? `Rs. ${row.predToday.toFixed(2)}` : 'N/A'}
                </td>
                <td className={`px-4 py-3 text-right font-semibold ${row.diff >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                  {row.predToday ? (
                    <>
                      {row.diff > 0 ? '+' : ''}{row.diff.toFixed(2)} ({row.diffPercent > 0 ? '+' : ''}{row.diffPercent.toFixed(2)}%)
                    </>
                  ) : 'N/A'}
                </td>
                <td className="px-4 py-3 text-right font-bold text-emerald-400 bg-emerald-900/10">
                  Rs. {row.predTomorrow.toFixed(2)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};


const Dashboard = () => {
  const [ticker, setTicker] = useState('PSO');
  const [availableTickers, setAvailableTickers] = useState([]);
  const [data, setData] = useState(null);
  const [company, setCompany] = useState(null);
  const [liveData, setLiveData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [modelType, setModelType] = useState('RF'); // 'RF', 'LR', 'XGB', 'LSTM'
  const [timeRange, setTimeRange] = useState('30D'); // '7D', '30D', '90D', '1Y'
  const [isMarketClosed, setIsMarketClosed] = useState(false);

  useEffect(() => {
    const fetchTickers = async () => {
      try {
        const res = await axios.get(`${API_BASE_URL}/api/tickers`);
        const options = res.data.map(t => ({
          value: t.ticker,
          label: `${t.ticker} - ${t.name} (${t.sector})`
        }));
        setAvailableTickers(options);
      } catch (err) {
        console.error("Failed to load tickers", err);
      }
    };
    fetchTickers();
  }, []);

  useEffect(() => {
    const checkMarketStatus = () => {
      const now = new Date();
      const options = { timeZone: 'Asia/Karachi', hour12: false, hour: 'numeric', minute: 'numeric', weekday: 'short' };
      const formatter = new Intl.DateTimeFormat('en-US', options);
      const parts = formatter.formatToParts(now);
      
      let hour = 0;
      let weekday = '';
      
      parts.forEach(part => {
        if (part.type === 'hour') hour = parseInt(part.value, 10);
        if (part.type === 'weekday') weekday = part.value;
      });

      // PSX closes at 4 PM (16:00). Also closed on weekends.
      if (weekday === 'Sat' || weekday === 'Sun' || hour >= 16) {
        setIsMarketClosed(true);
      } else {
        setIsMarketClosed(false);
      }
    };
    
    checkMarketStatus();
    const interval = setInterval(checkMarketStatus, 60000);
    return () => clearInterval(interval);
  }, []);

  const fetchData = async (symbol) => {
    setLoading(true);
    setError(null);
    try {
      const [predRes, compRes, liveRes] = await Promise.all([
        axios.post(`${API_BASE_URL}/api/predict`, { ticker: symbol }),
        axios.get(`${API_BASE_URL}/api/company/${symbol}`),
        axios.get(`${API_BASE_URL}/api/realtime/${symbol}`).catch(() => null)
      ]);
      
      setData(predRes.data);
      setCompany(compRes.data);
      setLiveData(liveRes ? liveRes.data : null);
    } catch (err) {
      setError(err.response?.data?.detail || "Failed to fetch data from the server.");
      setData(null);
      setCompany(null);
      setLiveData(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData(ticker);
  }, []);

  // Poll for live data every 15 seconds if a ticker is active
  useEffect(() => {
    if (!data || !ticker) return;
    const interval = setInterval(async () => {
      try {
        const res = await axios.get(`${API_BASE_URL}/api/realtime/${ticker}`);
        setLiveData(res.data);
      } catch (err) {
        console.error("Failed to fetch live data:", err);
      }
    }, 15000);
    return () => clearInterval(interval);
  }, [data, ticker]);

  const handleSearch = (e) => {
    e.preventDefault();
    if (ticker.trim()) fetchData(ticker);
  };

  // Compute stats and slice data based on time range
  const chartData = React.useMemo(() => {
    if (!data) return [];
    
    let history = [...data.historical_data];
    
    // The date of the latest actual data point
    const lastRecordDate = new Date(history[history.length - 1].date);
    const cutoffDate = new Date(lastRecordDate);
    
    // Slice based on exact Calendar Time, not trading day row counts
    if (timeRange === '7D') cutoffDate.setDate(cutoffDate.getDate() - 7);
    if (timeRange === '30D') cutoffDate.setMonth(cutoffDate.getMonth() - 1);
    if (timeRange === '90D') cutoffDate.setMonth(cutoffDate.getMonth() - 3);
    if (timeRange === '1Y') cutoffDate.setFullYear(cutoffDate.getFullYear() - 1);
    if (timeRange === '5Y') cutoffDate.setFullYear(cutoffDate.getFullYear() - 5);
    
    history = history.filter(row => new Date(row.date) >= cutoffDate);
    
    // Map backend predicted values to a single 'predictedClose' key based on selected model
    history = history.map(row => ({
      ...row,
      predictedClose: modelType === 'RF' ? row.rf_pred :
                      modelType === 'LSTM' ? row.lstm_pred :
                      modelType === 'LR' ? row.lr_pred :
                      row.xgb_pred
    }));
    
    const predictedPrice = modelType === 'RF' ? data.rf_predicted_price :
                           modelType === 'LSTM' ? data.lstm_predicted_price :
                           modelType === 'LR' ? data.lr_predicted_price :
                           data.xgb_predicted_price;
    const nextDate = new Date(lastRecordDate);
    nextDate.setDate(lastRecordDate.getDate() + 1);
    
    history.push({
      date: nextDate.toISOString().split('T')[0],
      predictedClose: predictedPrice,
      isPrediction: true
    });
    
    return history;
  }, [data, modelType, timeRange]);

  // Calculate Daily Change Indicators from the raw backend data
  const { currentPrice, dailyChange, dailyChangePercent } = React.useMemo(() => {
    if (!data || data.historical_data.length < 2) return { currentPrice: 0, dailyChange: 0, dailyChangePercent: 0 };
    const history = data.historical_data;
    const current = history[history.length - 1].close;
    const previous = history[history.length - 2].close;
    
    const change = current - previous;
    const percent = (change / previous) * 100;
    
    return {
      currentPrice: current,
      dailyChange: change,
      dailyChangePercent: percent
    };
  }, [data]);

  // Format the Timestamp beautifully assuming 4 PM Market Close
  const formattedTimestamp = React.useMemo(() => {
    if (!data) return '';
    try {
      const dateObj = new Date(data.latest_date + 'T16:00:00');
      return new Intl.DateTimeFormat('en-US', {
        weekday: 'short', month: 'short', day: 'numeric', year: 'numeric',
        hour: 'numeric', minute: '2-digit', hour12: true
      }).format(dateObj);
    } catch {
      return data.latest_date;
    }
  }, [data]);

  const activePrediction = data ? (
    modelType === 'RF' ? data.rf_predicted_price :
    modelType === 'LSTM' ? data.lstm_predicted_price :
    modelType === 'LR' ? data.lr_predicted_price :
    data.xgb_predicted_price
  ) : 0;
  
  const displayPrice = liveData ? liveData.price : currentPrice;
  const displayChange = liveData ? liveData.change : dailyChange;
  const displayPercent = liveData ? liveData.change_percent : dailyChangePercent;
  const predictionDiff = activePrediction - displayPrice;

  // Dynamic X-Axis Formatter based on selected time range
  const formatXAxisDate = (tickStr) => {
    if (!tickStr) return '';
    const parts = tickStr.split('-');
    if (parts.length !== 3) return tickStr;
    const [year, monthStr, day] = parts;
    const dateObj = new Date(year, parseInt(monthStr)-1, day);
    
    if (timeRange === '5Y') {
      return year; // For 5Y, only show the Year (e.g. 2024)
    } else if (timeRange === '1Y') {
      return dateObj.toLocaleDateString('en-US', { month: 'short', year: '2-digit' }); // Jan 24
    } else {
      return dateObj.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }); // Jan 5
    }
  };

  const getTickGap = () => {
    if (timeRange === '7D') return 0;
    if (timeRange === '30D') return 10;
    if (timeRange === '90D') return 20;
    if (timeRange === '1Y') return 40;
    return 80;
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 p-4 md:p-8 font-sans">
      <div className="max-w-6xl mx-auto space-y-6">
        
        {/* Top Navbar */}
        <div className="flex flex-col md:flex-row justify-between items-center bg-slate-800 p-6 rounded-xl shadow-lg border border-slate-700">
          <div>
            <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-emerald-400 to-cyan-400">
              AI Stock Predictor
            </h1>
            <div className="flex items-center space-x-2 mt-2">
              <button 
                onClick={() => setModelType('RF')}
                className={`text-xs px-3 py-1 rounded-full font-semibold transition-all ${
                  modelType === 'RF' 
                  ? 'bg-emerald-500 text-white ring-2 ring-emerald-300 ring-offset-2 ring-offset-slate-800' 
                  : 'bg-slate-700 text-slate-400 hover:bg-slate-600'
                }`}
              >
                Random Forest
              </button>
              <button 
                onClick={() => setModelType('LR')}
                className={`text-xs px-3 py-1 rounded-full font-semibold transition-all ${
                  modelType === 'LR' 
                  ? 'bg-orange-500 text-white ring-2 ring-orange-300 ring-offset-2 ring-offset-slate-800' 
                  : 'bg-slate-700 text-slate-400 hover:bg-slate-600'
                }`}
              >
                Linear Regression
              </button>
              <button 
                onClick={() => setModelType('XGB')}
                className={`text-xs px-3 py-1 rounded-full font-semibold transition-all ${
                  modelType === 'XGB' 
                  ? 'bg-purple-500 text-white ring-2 ring-purple-300 ring-offset-2 ring-offset-slate-800' 
                  : 'bg-slate-700 text-slate-400 hover:bg-slate-600'
                }`}
              >
                XGBoost
              </button>
              <button 
                onClick={() => setModelType('LSTM')}
                className={`text-xs px-3 py-1 rounded-full font-semibold transition-all ${
                  modelType === 'LSTM' 
                  ? 'bg-cyan-500 text-white ring-2 ring-cyan-300 ring-offset-2 ring-offset-slate-800' 
                  : 'bg-slate-700 text-slate-400 hover:bg-slate-600'
                }`}
              >
                Deep Learning (LSTM)
              </button>
            </div>
          </div>
          
          <form onSubmit={handleSearch} className="mt-4 md:mt-0 flex space-x-2">
            <div className="w-64 md:w-80 text-slate-900 text-sm">
              <Select
                value={availableTickers.find(opt => opt.value === ticker) || null}
                onChange={(selected) => {
                  setTicker(selected.value);
                  fetchData(selected.value);
                }}
                options={availableTickers}
                classNamePrefix="react-select"
                placeholder="Search Company..."
              />
            </div>
            <button 
              type="submit" 
              disabled={loading}
              className="bg-emerald-500 hover:bg-emerald-600 text-white px-6 py-2 rounded-lg font-semibold transition-colors disabled:opacity-50 flex items-center justify-center min-w-[100px]"
            >
              {loading ? (
                <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
              ) : 'Analyze'}
            </button>
          </form>
        </div>

        {/* Empty / Error State */}
        {error && (
          <div className="bg-red-900/50 border-l-4 border-red-500 p-6 rounded-r-xl text-red-200 flex items-center space-x-4 shadow-lg">
            <svg className="w-8 h-8 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <div>
              <p className="font-bold text-lg">Analysis Failed</p>
              <p className="text-sm">{error}</p>
            </div>
          </div>
        )}

        {/* Main Content Area */}
        {data && company && !error && (
          <div className="space-y-6">
            
            {/* Split Layout: Stats (Left) & Chart (Right) */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              
              {/* Left Column: Stats */}
              <div className="md:col-span-1 flex flex-col space-y-6">
                
                {/* Current Price Card (Enhanced) */}
                <div className="bg-slate-800 p-6 rounded-xl shadow-lg border border-slate-700">
                  <h3 className="text-slate-300 text-xs font-bold uppercase tracking-wider mb-2 flex items-center">
                    {liveData && !isMarketClosed ? (
                      <>
                        <span className="relative flex h-3 w-3 mr-2">
                          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                          <span className="relative inline-flex rounded-full h-3 w-3 bg-emerald-500"></span>
                        </span>
                        Live Price
                      </>
                    ) : liveData && isMarketClosed ? (
                      <>
                        <span className="relative flex h-3 w-3 mr-2">
                          <span className="relative inline-flex rounded-full h-3 w-3 bg-rose-500"></span>
                        </span>
                        Market Closed
                      </>
                    ) : (
                      "Current Price"
                    )}
                  </h3>
                  <div className="text-4xl font-bold text-white mb-2">
                    Rs. {displayPrice.toFixed(2)}
                  </div>
                  
                  {/* Real-time Change Indicator */}
                  <div className="flex items-center space-x-2 font-bold text-lg">
                    {displayChange >= 0 ? (
                      <span className="text-emerald-500 flex items-center">
                        <span className="mr-1 text-sm">▲</span> 
                        {displayChange.toFixed(2)} ({displayPercent.toFixed(2)}%)
                      </span>
                    ) : (
                      <span className="text-rose-500 flex items-center">
                        <span className="mr-1 text-sm">▼</span> 
                        {Math.abs(displayChange).toFixed(2)} ({displayPercent.toFixed(2)}%)
                      </span>
                    )}
                  </div>
                  
                  {/* Company Name & Sector Context */}
                  <div className="mt-6 pt-4 border-t border-slate-700/50">
                    <p className="text-slate-100 font-semibold truncate" title={company.name}>{company.name}</p>
                    <p className="text-slate-400 text-xs mt-1 uppercase tracking-wider">{company.sector}</p>
                  </div>
                </div>

                {/* AI Prediction Card */}
                <div className={`p-6 rounded-xl shadow-lg border relative overflow-hidden ${
                  modelType === 'RF' ? 'bg-gradient-to-br from-slate-800 to-emerald-900/30 border-emerald-900/50' :
                  modelType === 'LSTM' ? 'bg-gradient-to-br from-slate-800 to-cyan-900/30 border-cyan-900/50' :
                  modelType === 'LR' ? 'bg-gradient-to-br from-slate-800 to-orange-900/30 border-orange-900/50' :
                  'bg-gradient-to-br from-slate-800 to-purple-900/30 border-purple-900/50'
                }`}>
                  <h3 className={`text-xs font-bold uppercase tracking-wider mb-2 ${
                    modelType === 'RF' ? 'text-emerald-400' :
                    modelType === 'LSTM' ? 'text-cyan-400' :
                    modelType === 'LR' ? 'text-orange-400' :
                    'text-purple-400'
                  }`}>
                    AI Target ({modelType})
                  </h3>
                  <div className="text-4xl font-bold text-white mb-3">
                    Rs. {activePrediction.toFixed(2)}
                  </div>
                  <div className="inline-flex items-center text-sm font-semibold bg-slate-900/60 px-3.5 py-1.5 rounded-full border border-slate-700/50 shadow-inner">
                    {predictionDiff > 0 ? (
                      <span className="text-emerald-400 flex items-center">
                        <svg className="w-4 h-4 mr-1.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="3" d="M5 15l7-7 7 7"></path>
                        </svg>
                        Target Increase
                      </span>
                    ) : (
                      <span className="text-rose-400 flex items-center">
                        <svg className="w-4 h-4 mr-1.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="3" d="M19 9l-7 7-7-7"></path>
                        </svg>
                        Target Decrease
                      </span>
                    )}
                    <div className="w-px h-3.5 bg-slate-700/80 mx-2.5"></div>
                    <span className="text-slate-200">{Math.abs(predictionDiff).toFixed(2)} Rs.</span>
                  </div>
                  
                  {/* Ensemble Range & Confidence */}
                  <div className="mt-5 space-y-3 relative z-10">
                    <div>
                      <div className="text-[10px] uppercase text-slate-400 font-bold tracking-wider mb-1">Ensemble Range</div>
                      <div className="text-sm font-medium text-slate-200 bg-slate-900/40 py-1.5 px-2.5 rounded border border-slate-700/50 inline-block">
                        Rs. {data.ensemble_min.toFixed(2)} <span className="text-slate-500 mx-1">—</span> Rs. {data.ensemble_max.toFixed(2)}
                      </div>
                    </div>
                    
                    <div>
                      <div className="text-[10px] uppercase text-slate-400 font-bold tracking-wider mb-1 flex justify-between items-center">
                        <span>Model Agreement</span>
                        <span className={`font-bold ${
                          data.model_agreement_score >= 80 ? 'text-emerald-400' : 
                          data.model_agreement_score >= 60 ? 'text-amber-400' : 'text-rose-400'
                        }`}>{data.model_agreement_score}%</span>
                      </div>
                      <div className="w-full bg-slate-900/60 rounded-full h-1.5 border border-slate-700/50 overflow-hidden">
                        <div 
                          className={`h-1.5 rounded-full transition-all duration-1000 ease-out ${
                            data.model_agreement_score >= 80 ? 'bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]' : 
                            data.model_agreement_score >= 60 ? 'bg-amber-500 shadow-[0_0_8px_rgba(245,158,11,0.5)]' : 
                            'bg-rose-500 shadow-[0_0_8px_rgba(244,63,94,0.5)]'
                          }`}
                          style={{ width: `${Math.min(100, Math.max(0, data.model_agreement_score))}%` }}
                        ></div>
                      </div>
                    </div>
                  </div>
                  
                  {/* Decorative Background Icon */}
                  <div className="absolute right-4 top-8 opacity-40 pointer-events-none text-white/50">
                    <svg className="w-28 h-28 drop-shadow-md" fill="currentColor" viewBox="0 0 24 24">
                      {predictionDiff > 0 ? (
                        <path d="M16 6l2.29 2.29-4.88 4.88-4-4L2 16.59 3.41 18l6-6 4 4 6.3-6.29L22 12V6z" />
                      ) : (
                        <path d="M16 18l2.29-2.29-4.88-4.88-4 4L2 7.41 3.41 6l6 6 4-4 6.3 6.29L22 12v6z" />
                      )}
                    </svg>
                  </div>
                </div>

                {/* Market Sentiment Card */}
                <div className={`p-6 rounded-xl shadow-lg border relative overflow-hidden ${
                  data.latest_sentiment > 0.05 ? 'bg-gradient-to-br from-slate-800 to-emerald-900/30 border-emerald-900/50' :
                  data.latest_sentiment < -0.05 ? 'bg-gradient-to-br from-slate-800 to-rose-900/30 border-rose-900/50' :
                  'bg-gradient-to-br from-slate-800 to-amber-900/30 border-amber-900/50'
                }`}>
                  <h3 className="text-xs font-bold uppercase tracking-wider mb-2 text-slate-300">
                    Market Sentiment
                  </h3>
                  <div className="text-3xl font-bold text-white mb-2 flex items-center">
                    {data.latest_sentiment > 0.05 ? 'Bullish' : data.latest_sentiment < -0.05 ? 'Bearish' : 'Neutral'}
                  </div>
                  <div className="text-sm font-medium text-slate-400 mb-4">
                    Score: {data.latest_sentiment.toFixed(2)}
                  </div>
                  
                  {/* Sentiment Gauge */}
                  <div className="w-full bg-slate-900/60 rounded-full h-2 border border-slate-700/50 overflow-hidden relative">
                    <div className="absolute left-1/2 top-0 bottom-0 w-px bg-slate-500/50 z-10"></div>
                    <div 
                      className={`h-2 rounded-full absolute top-0 bottom-0 transition-all duration-1000 ease-out ${
                        data.latest_sentiment > 0 ? 'bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]' : 
                        'bg-rose-500 shadow-[0_0_8px_rgba(243,33,33,0.5)]'
                      }`}
                      style={{ 
                        width: `${Math.min(50, Math.abs(data.latest_sentiment) * 50)}%`,
                        [data.latest_sentiment > 0 ? 'left' : 'right']: '50%'
                      }}
                    ></div>
                  </div>
                  <div className="flex justify-between text-[10px] text-slate-500 font-bold uppercase mt-1">
                    <span>Bearish</span>
                    <span>Neutral</span>
                    <span>Bullish</span>
                  </div>
                </div>

                {/* Recent Dividend Card */}
                {data.latest_dividend && (
                  <div className="p-6 rounded-xl shadow-lg border border-slate-700 bg-gradient-to-br from-slate-800 to-indigo-900/20 relative overflow-hidden">
                    <h3 className="text-xs font-bold uppercase tracking-wider mb-2 text-slate-300 flex items-center">
                      <svg className="w-4 h-4 mr-1.5 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                      </svg>
                      Recent Dividend
                    </h3>
                    <div className="text-3xl font-bold text-white mb-1">
                      Rs. {data.latest_dividend.amount.toFixed(2)}
                    </div>
                    <div className="text-sm font-medium text-slate-400 mb-3">
                      Yield: {((data.latest_dividend.amount / data.current_price) * 100).toFixed(2)}%
                    </div>
                    
                    <div className="flex items-center justify-between text-[11px] font-semibold bg-slate-900/50 px-3 py-2 rounded-lg border border-slate-700/50">
                      <span className="text-slate-400">Ex-Date</span>
                      <span className="text-indigo-300">{new Date(data.latest_dividend.ex_date).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}</span>
                    </div>
                  </div>
                )}

              </div>

              {/* Right Column: Chart and Table */}
              <div className="md:col-span-2 flex flex-col">
                
                {/* Chart Container */}
                <div className="bg-slate-800 p-6 rounded-xl shadow-lg border border-slate-700 h-[450px] flex flex-col">
                  <div className="flex justify-between items-start mb-4">
                    {/* Time Range Tabs */}
                    <div className="flex space-x-1 bg-slate-900/50 p-1 rounded-lg border border-slate-700/50">
                      {['7D', '30D', '90D', '1Y', '5Y'].map(range => (
                        <button
                          key={range}
                          onClick={() => setTimeRange(range)}
                          className={`px-3 py-1 text-xs font-bold rounded-md transition-all ${
                            timeRange === range 
                              ? 'bg-emerald-500 text-white shadow-sm' 
                              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
                          }`}
                        >
                          {range}
                        </button>
                      ))}
                    </div>
                    
                    {/* Beautiful Timestamp */}
                    <div className="text-right">
                      <p className="text-slate-300 text-xs font-medium">As of {formattedTimestamp}</p>
                      <p className="text-slate-500 text-[10px] mt-0.5">Market Close Data</p>
                    </div>
                  </div>

                  <div className="flex-grow pt-2">
                    <ResponsiveContainer width="100%" height="100%">
                      <ComposedChart data={chartData} margin={{ top: 10, right: 0, left: 10, bottom: 0 }}>
                        <defs>
                          <linearGradient id="colorClose" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor={modelType === 'RF' ? "#10b981" : modelType === 'LSTM' ? "#06b6d4" : modelType === 'LR' ? "#f97316" : "#a855f7"} stopOpacity={0.5}/>
                            {/* Fades out much earlier to look professional */}
                            <stop offset="60%" stopColor={modelType === 'RF' ? "#10b981" : modelType === 'LSTM' ? "#06b6d4" : modelType === 'LR' ? "#f97316" : "#a855f7"} stopOpacity={0}/>
                          </linearGradient>
                        </defs>
                        
                        {/* Subtle, low-opacity horizontal gridlines only */}
                        <CartesianGrid stroke="#334155" strokeOpacity={0.4} vertical={false} strokeDasharray="4 4" />
                        
                        <XAxis 
                          dataKey="date" 
                          stroke="#64748b" 
                          tick={{fill: '#64748b', fontSize: 11}}
                          tickLine={false}
                          axisLine={{stroke: '#334155'}}
                          tickFormatter={formatXAxisDate} 
                          minTickGap={getTickGap()}
                        />
                        
                        {/* Professional Right-Aligned Y-Axis */}
                        <YAxis 
                          domain={['auto', 'auto']} 
                          orientation="right"
                          stroke="#64748b" 
                          tick={{fill: '#64748b', fontSize: 11}}
                          tickLine={false}
                          axisLine={{stroke: '#334155'}}
                          tickFormatter={(tick) => `${tick}`}
                        />
                        
                        {/* Interactive Tooltip with Crosshair */}
                        <Tooltip 
                          contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px', color: '#f8fafc', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.3)' }}
                          itemStyle={{ color: modelType === 'RF' ? '#34d399' : modelType === 'LSTM' ? '#22d3ee' : modelType === 'LR' ? '#fb923c' : '#c084fc', fontWeight: 'bold' }}
                          labelStyle={{ color: '#94a3b8', marginBottom: '4px', fontSize: '12px' }}
                          cursor={{ stroke: '#64748b', strokeWidth: 1, strokeDasharray: '3 3' }}
                        />
                        
                        <ReferenceLine x={data.latest_date} stroke="#64748b" strokeDasharray="3 3" label={{ position: 'insideTopLeft', value: 'Today', fill: '#94a3b8', fontSize: 11 }} />
                        
                        {/* Linear interpolation for jagged, realistic lines */}
                        <Area 
                          type="linear" 
                          dataKey="close" 
                          name="Actual Price"
                          stroke={modelType === 'RF' ? "#10b981" : modelType === 'LSTM' ? "#06b6d4" : modelType === 'LR' ? "#f97316" : "#a855f7"} 
                          strokeWidth={2}
                          fillOpacity={1} 
                          fill="url(#colorClose)" 
                          activeDot={{ r: 5, strokeWidth: 2, stroke: '#0f172a', fill: modelType === 'RF' ? '#34d399' : modelType === 'LSTM' ? '#22d3ee' : modelType === 'LR' ? '#fb923c' : '#c084fc' }}
                        />
                        <Line
                          type="linear"
                          dataKey="predictedClose"
                          name="Predicted Price"
                          stroke="#fbbf24" // Amber color for visibility
                          strokeWidth={2}
                          strokeDasharray="4 4"
                          dot={false}
                          activeDot={{ r: 5, strokeWidth: 0, fill: '#fbbf24' }}
                          connectNulls={true}
                        />
                      </ComposedChart>
                    </ResponsiveContainer>
                  </div>
                </div>
                
                {/* All Companies Table positioned beneath the Chart */}
                <AllCompaniesTable modelType={modelType} availableTickers={availableTickers} />

              </div>
            </div>

            {/* Market Performers Component - Positioned exactly above Company Profile */}
            <MarketPerformers />

            {/* Recent News Feed Component */}
            <div className="bg-slate-800 p-6 rounded-xl shadow-lg border border-slate-700">
              <h2 className="text-xl font-bold text-slate-100 mb-6 border-b border-slate-700 pb-3 flex items-center">
                <svg className="w-5 h-5 mr-2 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 20H5a2 2 0 01-2-2V6a2 2 0 012-2h10a2 2 0 012 2v1m2 13a2 2 0 01-2-2V7m2 13a2 2 0 002-2V9.5a2.5 2.5 0 00-2.5-2.5H15"></path>
                </svg>
                Latest News Intelligence
              </h2>
              {data.recent_news && data.recent_news.length > 0 ? (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {data.recent_news.map((news, idx) => (
                    <a 
                      key={idx}
                      href={news.url} 
                      target="_blank" 
                      rel="noreferrer"
                      className="group bg-slate-900/50 p-4 rounded-lg border border-slate-700 hover:border-blue-500/50 hover:bg-slate-700 transition-all flex flex-col justify-between"
                    >
                      <div>
                        <div className="flex justify-between items-start mb-2">
                          <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">{news.source}</span>
                          <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${
                            news.sentiment_score > 0.05 ? 'bg-emerald-900/50 text-emerald-400 border border-emerald-800/50' : 
                            news.sentiment_score < -0.05 ? 'bg-rose-900/50 text-rose-400 border border-rose-800/50' : 
                            'bg-amber-900/50 text-amber-400 border border-amber-800/50'
                          }`}>
                            {news.sentiment_score > 0.05 ? 'Bullish' : news.sentiment_score < -0.05 ? 'Bearish' : 'Neutral'}
                          </span>
                        </div>
                        <h4 className="text-sm font-semibold text-slate-200 group-hover:text-blue-400 transition-colors line-clamp-2 leading-snug">
                          {news.headline}
                        </h4>
                      </div>
                      <div className="mt-4 text-[10px] text-slate-500">
                        {new Date(news.published_at).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
                      </div>
                    </a>
                  ))}
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center py-10 bg-slate-900/30 rounded-lg border border-slate-700 border-dashed">
                  <svg className="w-12 h-12 text-slate-600 mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M19 20H5a2 2 0 01-2-2V6a2 2 0 012-2h10a2 2 0 012 2v1m2 13a2 2 0 01-2-2V7m2 13a2 2 0 002-2V9.5a2.5 2.5 0 00-2.5-2.5H15"></path>
                  </svg>
                  <p className="text-slate-400 font-medium">No recent financial news detected for this company.</p>
                  <p className="text-slate-500 text-sm mt-1">Sentiment will remain Neutral (0.00) until new market activity is recorded.</p>
                </div>
              )}
            </div>

            {/* Scraped Company Profile Section */}
            <div className="bg-slate-800 p-6 rounded-xl shadow-lg border border-slate-700">
              <h2 className="text-xl font-bold text-slate-100 mb-6 border-b border-slate-700 pb-3">Company Profile</h2>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                {/* Left Column */}
                <div className="space-y-6">
                  <div>
                    <h3 className="text-xs font-bold text-slate-300 uppercase tracking-wider mb-2">Business Description</h3>
                    <p className="text-slate-400 text-sm leading-relaxed">
                      {company.description}
                    </p>
                  </div>
                  
                  {company.details.ADDRESS && (
                    <div>
                      <h3 className="text-xs font-bold text-slate-300 uppercase tracking-wider mb-2">Address</h3>
                      <p className="text-slate-400 text-sm">{company.details.ADDRESS}</p>
                    </div>
                  )}

                  {company.details.WEBSITE && (
                    <div>
                      <h3 className="text-xs font-bold text-slate-300 uppercase tracking-wider mb-2">Website</h3>
                      <a href={company.details.WEBSITE.startsWith('http') ? company.details.WEBSITE : `http://${company.details.WEBSITE}`} target="_blank" rel="noreferrer" className="text-emerald-400 hover:text-emerald-300 text-sm font-medium">
                        {company.details.WEBSITE}
                      </a>
                    </div>
                  )}
                </div>

                {/* Right Column */}
                <div className="space-y-6">
                  <div>
                    <h3 className="text-xs font-bold text-slate-300 uppercase tracking-wider mb-2">Key People</h3>
                    <div className="bg-slate-900/50 rounded-lg border border-slate-700 overflow-hidden">
                      <table className="w-full text-sm text-left">
                        <tbody>
                          {company.people.map((person, idx) => (
                            <tr key={idx} className="border-b border-slate-700/50 last:border-0 hover:bg-slate-800/50 transition-colors">
                              <td className="px-4 py-3 font-semibold text-slate-200">{person.name}</td>
                              <td className="px-4 py-3 text-slate-400">{person.role}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>

                  {company.details.REGISTRAR && (
                    <div>
                      <h3 className="text-xs font-bold text-slate-300 uppercase tracking-wider mb-2">Registrar</h3>
                      <p className="text-slate-400 text-sm">{company.details.REGISTRAR}</p>
                    </div>
                  )}

                  {company.details.AUDITOR && (
                    <div>
                      <h3 className="text-xs font-bold text-slate-300 uppercase tracking-wider mb-2">Auditor</h3>
                      <p className="text-slate-400 text-sm">{company.details.AUDITOR}</p>
                    </div>
                  )}
                </div>
              </div>
            </div>

          </div>
        )}
      </div>
    </div>
  );
};

export default Dashboard;
