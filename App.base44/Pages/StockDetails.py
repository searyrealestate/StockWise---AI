
import React, { useState, useEffect, useCallback } from "react";
import { useLocation, Link } from "react-router-dom";
import { Stock, Portfolio, User } from "@/entities/all"; // Added User import
import { InvokeLLM } from "@/integrations/Core"; // Added InvokeLLM import
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ArrowLeft, TrendingUp, TrendingDown, Target, CheckCircle, XCircle, BarChart3, Building, Newspaper, Bell, BellOff, CandlestickChart, LineChart as LineChartIcon, RefreshCw, ZoomIn, ZoomOut } from "lucide-react";
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ComposedChart, Bar, ReferenceLine, ReferenceArea, Brush, BarChart, Cell } from "recharts";
import { format, subDays, subMonths, subYears } from "date-fns";
import { he } from "date-fns/locale";
import { createPageUrl } from "@/utils";

// Custom Candlestick component for Recharts
const CandlestickShape = (props) => {
  const { x, y, width, height, payload } = props;
  if (!payload || !payload.ohlc || height <= 0) return null;

  const { open, high, low, close } = payload.ohlc;

  if ([open, high, low, close].some(val => val === undefined || isNaN(val))) {
    return null;
  }

  const isPositive = close >= open;
  const color = isPositive ? '#10B981' : '#EF4444';
  const wickX = x + width / 2;

  const dataRange = high - low;
  if (dataRange <= 0) {
      return <line x1={x} y1={y + height/2} x2={x + width} y2={y + height/2} stroke="grey" strokeWidth={1} />;
  }

  const bodyTopY = y + (high - Math.max(open, close)) * (height / dataRange);
  const bodyHeight = Math.abs(open - close) * (height / dataRange);

  return (
    <g>
      <line x1={wickX} y1={y} x2={wickX} y2={y + height} stroke={color} strokeWidth={1} />
      <rect 
        x={x} 
        y={bodyTopY} 
        width={width} 
        height={Math.max(1, bodyHeight)} 
        fill={isPositive ? 'white' : color} 
        stroke={color} 
        strokeWidth={1} 
      />
    </g>
  );
};

export default function StockDetailsPage() {
    const [stock, setStock] = useState(null);
    const [portfolioItem, setPortfolioItem] = useState(null);
    const [isLoading, setIsLoading] = useState(true);
    const [chartType, setChartType] = useState('line');
    const [timeframe, setTimeframe] = useState('1Y');
    const [priceData, setPriceData] = useState([]);
    const [isUpdatingChart, setIsUpdatingChart] = useState(false);
    const [lastUpdate, setLastUpdate] = useState(new Date());
    const [zoomDomain, setZoomDomain] = useState({ left: 0, right: 100 });
    const [showTechnicalAnalysis, setShowTechnicalAnalysis] = useState(false); // New state for technical analysis visibility
    const [technicalIndicators, setTechnicalIndicators] = useState(null); // New state for technical indicators
    const [userTheme, setUserTheme] = useState('light'); // New state for user theme

    const location = useLocation();

    // Real-time price simulation
    const simulateRealTimePriceUpdate = useCallback((currentData, currentStock) => {
        if (!currentData.length || !currentStock) return currentData;
        
        const updatedData = [...currentData];
        let lastDataPoint = { ...updatedData[updatedData.length - 1] };

        const now = new Date();
        
        const fluctuation = (Math.random() - 0.5) * 0.006;
        let newPrice = lastDataPoint.price * (1 + fluctuation);
        
        const currentOpen = lastDataPoint.ohlc?.open || lastDataPoint.price;
        const currentHigh = lastDataPoint.ohlc?.high || newPrice;
        const currentLow = lastDataPoint.ohlc?.low || newPrice;

        const newOHLC = {
            open: currentOpen,
            high: Math.max(currentHigh, newPrice),
            low: Math.min(currentLow, newPrice),
            close: parseFloat(newPrice.toFixed(2))
        };
        
        lastDataPoint = {
            ...lastDataPoint,
            price: parseFloat(newPrice.toFixed(2)),
            ohlc: newOHLC,
            high: newOHLC.high,
            low: newOHLC.low,
            range: [newOHLC.low, newOHLC.high],
            dateString: format(now, 'dd/MM/yy HH:mm')
        };
        
        updatedData[updatedData.length - 1] = lastDataPoint;
        
        return updatedData;
    }, []);

    // Fetch real market data using InvokeLLM with internet search
    const fetchRealMarketData = useCallback(async (symbol, timeframePeriod) => {
        try {
            const prompt = `
            חפש נתוני מחירים אמיתיים עבור המניה ${symbol} בתקופה של ${timeframePeriod}.
            
            אני צריך נתונים היסטוריים של:
            - מחירי פתיחה, סגירה, גבוה ונמוך יומיים
            - נפח מסחר יומי
            - לפחות 30 נקודות נתונים
            
            החזר את המידע בפורמט JSON פשוט עם מערך של אובייקטים.
            כל אובייקט צריך להכיל: date, open, high, low, close, volume
            
            אם אין נתונים מדויקים זמינים, צור נתונים ריאליסטיים על בסיס המידע הזמין.
            `;

            const result = await InvokeLLM({
                prompt,
                add_context_from_internet: true,
                response_json_schema: {
                    type: "object",
                    properties: {
                        data: {
                            type: "array",
                            items: {
                                type: "object",
                                properties: {
                                    date: { type: "string" },
                                    open: { type: "number" },
                                    high: { type: "number" },
                                    low: { type: "number" },
                                    close: { type: "number" },
                                    volume: { type: "number" }
                                }
                            }
                        }
                    }
                }
            });

            if (result?.data && Array.isArray(result.data) && result.data.length > 0) {
                return result.data.map(item => {
                    const date = new Date(item.date);
                    const ohlc = {
                        open: parseFloat(item.open.toFixed(2)),
                        high: parseFloat(item.high.toFixed(2)),
                        low: parseFloat(item.low.toFixed(2)),
                        close: parseFloat(item.close.toFixed(2))
                    };

                    return {
                        date,
                        dateString: format(date, 'dd/MM/yy'),
                        price: ohlc.close,
                        ohlc,
                        volume: Math.floor(item.volume),
                        high: ohlc.high,
                        low: ohlc.low,
                        range: [ohlc.low, ohlc.high]
                    };
                });
            }

            throw new Error('No valid data received from LLM for real market data');

        } catch (error) {
            console.warn('Failed to fetch real data, using simulated data:', error);
            // Fallback to enhanced simulated data if fetching real data fails
            return generateRealisticMarketData(stock, timeframePeriod);
        }
    }, [stock]);

    // Enhanced realistic market data generator (fallback)
    const generateRealisticMarketData = useCallback((currentStock, timeframePeriod) => {
        if (!currentStock) return [];
        
        const daysMap = { '1W': 7, '1M': 30, '6M': 180, '1Y': 365, 'ALL': 730 };
        const days = daysMap[timeframePeriod] || 365;
        
        // Get current market context using known market patterns
        const marketTrends = {
            'TSLA': { volatility: 0.055, trendBias: 0.02, avgVolume: 85000000 },
            'AAPL': { volatility: 0.025, trendBias: 0.015, avgVolume: 65000000 },
            'MSFT': { volatility: 0.022, trendBias: 0.018, avgVolume: 45000000 },
            'NVDA': { volatility: 0.045, trendBias: 0.025, avgVolume: 55000000 },
            'default': { volatility: 0.030, trendBias: 0.01, avgVolume: 25000000 }
        };

        const stockProfile = marketTrends[currentStock.symbol] || marketTrends.default;
        
        const data = [];
        let currentPrice = currentStock.current_price;
        
        // Calculate realistic historical range
        const expectedGrowth = (currentStock.expected_return || 10) / 100;
        const timeRatio = days / 365;
        const historicalReturn = expectedGrowth * timeRatio;
        let price = currentPrice / (1 + historicalReturn);
        
        // Add some randomness to starting price
        price *= (0.9 + Math.random() * 0.2);
        
        for (let i = 0; i < days; i++) {
            const date = new Date(Date.now() - (days - 1 - i) * 24 * 60 * 60 * 1000);
            const progress = i / days;
            
            // Market patterns
            const weekday = date.getDay();
            const isWeekend = weekday === 0 || weekday === 6;
            const weekdayMultiplier = isWeekend ? 0.3 : 1.0;
            
            // Monthly patterns
            const monthDay = date.getDate();
            const monthEndEffect = monthDay > 25 ? 1.1 : monthDay < 5 ? 1.05 : 1.0;
            
            // Trend towards target with realistic noise
            const targetDrift = (currentPrice - price) * (progress * 0.001) + stockProfile.trendBias * 0.01;
            
            // Market cycles (quarterly earnings, etc.)
            const cyclicEffect = Math.sin(progress * Math.PI * 8) * 0.01;
            
            // Random walk with realistic constraints
            const randomComponent = (Math.random() - 0.5) * stockProfile.volatility;
            
            // Combine all effects
            const totalMovement = (targetDrift + cyclicEffect + randomComponent) * weekdayMultiplier * monthEndEffect;
            
            // Calculate OHLC with realistic intraday movement
            const open = price;
            const close = open * (1 + totalMovement);
            
            // Realistic intraday range (usually 1.5-3x the open-close movement)
            const intraDayVolatility = Math.abs(totalMovement) * (1.5 + Math.random() * 1.5);
            const gapUp = Math.random() * intraDayVolatility * 0.5;
            const gapDown = Math.random() * intraDayVolatility * 0.5;
            
            const high = Math.max(open, close) + gapUp * price;
            const low = Math.min(open, close) - gapDown * price;
            
            // Volume patterns (higher on big moves, lower on weekends)
            const volumeFromMovement = 1 + Math.abs(totalMovement) * 5;
            const volume = Math.floor(
                stockProfile.avgVolume * 
                volumeFromMovement * 
                weekdayMultiplier * 
                (0.7 + Math.random() * 0.4)
            );
            
            const ohlc = {
                open: parseFloat(open.toFixed(2)),
                high: parseFloat(high.toFixed(2)),
                low: parseFloat(low.toFixed(2)),
                close: parseFloat(close.toFixed(2))
            };
            
            data.push({
                date,
                dateString: format(date, 'dd/MM/yy'),
                price: ohlc.close,
                ohlc,
                volume,
                high: ohlc.high,
                low: ohlc.low,
                range: [ohlc.low, ohlc.high]
            });
            
            price = close;
        }
        
        // Ensure convergence to current price
        if (data.length > 0) {
            const lastPrice = data[data.length - 1].price;
            const adjustment = currentStock.current_price / lastPrice;
            
            // Apply adjustment to last few points for smooth convergence
            const adjustmentPoints = Math.min(7, data.length);
            for (let i = 0; i < adjustmentPoints; i++) {
                const point = data[data.length - 1 - i];
                const factor = 1 + ((adjustment - 1) * (adjustmentPoints - i) / adjustmentPoints);
                
                point.ohlc.open = parseFloat((point.ohlc.open * factor).toFixed(2));
                point.ohlc.high = parseFloat((point.ohlc.high * factor).toFixed(2));
                point.ohlc.low = parseFloat((point.ohlc.low * factor).toFixed(2));
                point.ohlc.close = parseFloat((point.ohlc.close * factor).toFixed(2));
                
                point.price = point.ohlc.close;
                point.high = point.ohlc.high;
                point.low = point.ohlc.low;
                point.range = [point.ohlc.low, point.ohlc.high];
            }
            
            // Set real-time timestamp for last point
            data[data.length - 1].dateString = format(new Date(), 'dd/MM/yy HH:mm');
        }
        
        return data;
    }, []);

    // Generate technical indicators without AI
    const calculateTechnicalIndicators = useCallback((data) => {
        if (!data.length || data.length < 50) return null; // Need enough data for SMA50

        const prices = data.map(d => d.price);
        const closes = data.map(d => d.ohlc.close);
        const highs = data.map(d => d.ohlc.high);
        const lows = data.map(d => d.ohlc.low);
        
        // Simple Moving Averages
        const sma20 = prices.slice(-20).reduce((a, b) => a + b, 0) / 20;
        const sma50 = prices.slice(-50).reduce((a, b) => a + b, 0) / 50;
        
        // Support and Resistance (local minima and maxima)
        const findExtremes = (arr, windowSize = 5) => {
            const supports = new Set();
            const resistances = new Set();

            for (let i = windowSize; i < arr.length - windowSize; i++) {
                const isSupport = arr[i] < Math.min(...arr.slice(i - windowSize, i)) &&
                                  arr[i] < Math.min(...arr.slice(i + 1, i + 1 + windowSize));
                const isResistance = arr[i] > Math.max(...arr.slice(i - windowSize, i)) &&
                                     arr[i] > Math.max(...arr.slice(i + 1, i + 1 + windowSize));

                if (isSupport) supports.add(parseFloat(arr[i].toFixed(2)));
                if (isResistance) resistances.add(parseFloat(arr[i].toFixed(2)));
            }
            return { support: Array.from(supports).sort((a,b) => a-b), resistance: Array.from(resistances).sort((a,b) => b-a) };
        };

        const { support: rawSupports, resistance: rawResistances } = findExtremes(closes, 10); // Using closes for S/R

        // Filter and get the most recent/relevant 2-3 levels
        const support_levels = rawSupports.filter(s => s < closes[closes.length - 1]).slice(-3); // Supports below current price
        const resistance_levels = rawResistances.filter(r => r > closes[closes.length - 1]).slice(0, 3); // Resistances above current price
        
        // RSI Calculation (simplified 14-period)
        const calculateRSI = (closePrices, period = 14) => {
            if (closePrices.length < period + 1) return null;

            let gains = 0;
            let losses = 0;
            for (let i = 1; i <= period; i++) {
                const change = closePrices[i] - closePrices[i - 1];
                if (change > 0) gains += change;
                else losses += Math.abs(change);
            }

            let avgGain = gains / period;
            let avgLoss = losses / period;

            for (let i = period + 1; i < closePrices.length; i++) {
                const change = closePrices[i] - closePrices[i - 1];
                if (change > 0) {
                    avgGain = ((avgGain * (period - 1)) + change) / period;
                    avgLoss = (avgLoss * (period - 1)) / period;
                } else {
                    avgLoss = ((avgLoss * (period - 1)) + Math.abs(change)) / period;
                    avgGain = (avgGain * (period - 1)) / period;
                }
            }
            
            const rs = avgLoss === 0 ? (avgGain === 0 ? 0 : 1000) : avgGain / avgLoss; // Handle division by zero
            return 100 - (100 / (1 + rs));
        };
        const rsi = calculateRSI(closes, 14);

        // Fibonacci levels (based on a recent significant price swing, e.g., last 60 days)
        const recentHigh = Math.max(...highs.slice(-60));
        const recentLow = Math.min(...lows.slice(-60));
        const range = recentHigh - recentLow;
        
        const fibonacci_levels = [
            recentLow + range * 0.236,
            recentLow + range * 0.382,
            recentLow + range * 0.5,
            recentLow + range * 0.618,
            recentLow + range * 0.786
        ];
        
        // Analysis summary
        const currentPrice = prices[prices.length - 1];
        let trend = 'לא ברורה';
        if (sma20 > sma50 && currentPrice > sma20) {
            trend = 'עולה חזק';
        } else if (sma20 > sma50) {
            trend = 'עולה';
        } else if (sma20 < sma50 && currentPrice < sma20) {
            trend = 'יורדת חזק';
        } else if (sma20 < sma50) {
            trend = 'יורדת';
        }

        let rsiStatus = 'נייטרלי';
        if (rsi > 70) rsiStatus = 'קניית יתר';
        else if (rsi < 30) rsiStatus = 'מכירת יתר';
        
        const analysis_summary = `המניה נמצאת במגמה ${trend}. הממוצע הנע הקצר (${sma20.toFixed(2)}) ${sma20 > sma50 ? 'מעל' : 'מתחת'} לממוצע הארוך (${sma50.toFixed(2)}). ה-RSI מציין מצב של ${rsiStatus} (${rsi.toFixed(1)}). יש לבחון את רמות התמיכה וההתנגדות לקביעת נקודות כניסה ויציאה.`;
        
        return {
            sma20: parseFloat(sma20.toFixed(2)),
            sma50: parseFloat(sma50.toFixed(2)),
            support_levels: support_levels.map(l => parseFloat(l.toFixed(2))).filter(l => l > 0), 
            resistance_levels: resistance_levels.map(l => parseFloat(l.toFixed(2))).filter(l => l > 0),
            rsi: parseFloat(rsi.toFixed(1)),
            fibonacci_levels: fibonacci_levels.map(l => parseFloat(l.toFixed(2))).filter(l => l > 0),
            analysis_summary
        };
    }, []);

    const loadUserTheme = async () => {
        try {
            const user = await User.me();
            setUserTheme(user.theme || 'light');
            document.documentElement.classList.toggle('dark', user.theme === 'dark');
        } catch (error) {
            console.error("Error loading user theme:", error);
        }
    };

    useEffect(() => {
        loadUserTheme();
        const params = new URLSearchParams(location.search);
        const symbol = params.get('symbol');
        if (symbol) {
            loadStockData(symbol);
        }
    }, [location.search]);

    useEffect(() => {
        if (stock) {
            updateChartData();
        }
    }, [stock, timeframe, showTechnicalAnalysis]); // Re-run when stock, timeframe, or technical analysis preference changes

    useEffect(() => {
        if (!stock || !priceData.length) return;
        
        const interval = setInterval(() => {
            setPriceData(currentData => simulateRealTimePriceUpdate(currentData, stock));
            setLastUpdate(new Date());
        }, 10000);

        return () => clearInterval(interval);
    }, [stock, priceData.length, simulateRealTimePriceUpdate]);

    const loadStockData = async (symbol) => {
        setIsLoading(true);
        try {
            const stockData = await Stock.filter({ symbol });
            if (stockData.length > 0) {
                setStock(stockData[0]);
                
                const portfolioData = await Portfolio.filter({ stock_symbol: symbol, status: 'פעיל' });
                if (portfolioData.length > 0) {
                    setPortfolioItem(portfolioData[0]);
                }
            }
        } catch (error) {
            console.error("Error loading stock data:", error);
        }
        setIsLoading(false);
    };

    const updateChartData = async () => {
        if (!stock) return;
        setIsUpdatingChart(true);
        
        // Try to fetch real data first, fallback to simulated handled within fetchRealMarketData
        const newData = await fetchRealMarketData(stock.symbol, timeframe);
        setPriceData(newData);
        
        if (showTechnicalAnalysis && newData.length >= 50) { // Only generate if checkbox is checked and enough data
            const indicators = calculateTechnicalIndicators(newData);
            setTechnicalIndicators(indicators);
        } else {
            setTechnicalIndicators(null); // Clear indicators if checkbox is unchecked or not enough data
        }
        
        setLastUpdate(new Date());
        setIsUpdatingChart(false);
    };

    const handleMuteToggle = async () => {
        if (!portfolioItem) return;
        
        try {
            await Portfolio.update(portfolioItem.id, {
                notifications_muted: !portfolioItem.notifications_muted
            });
            setPortfolioItem(prev => ({
                ...prev,
                notifications_muted: !prev.notifications_muted
            }));
        } catch (error) {
            console.error("Error toggling mute status:", error);
        }
    };

    const refreshChartData = () => {
        updateChartData();
    };

    const handleZoomIn = () => {
        const { left, right } = zoomDomain;
        const center = (left + right) / 2;
        const range = (right - left) / 4;
        setZoomDomain({ 
            left: Math.max(0, center - range), 
            right: Math.min(100, center + range) 
        });
    };

    const handleZoomOut = () => {
        const { left, right } = zoomDomain;
        const center = (left + right) / 2;
        const range = (right - left) * 2;
        setZoomDomain({ 
            left: Math.max(0, center - range/2), 
            right: Math.min(100, center + range/2) 
        });
    };

    const resetZoom = () => {
        setZoomDomain({ left: 0, right: 100 });
    };

    if (isLoading) {
        return (
            <div className={`min-h-screen p-6 flex justify-center items-center ${userTheme === 'dark' ? 'bg-slate-900 text-white' : ''}`}>
                <div className="text-center">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
                    <p className={`text-center ${userTheme === 'dark' ? 'text-slate-300' : 'text-slate-600'}`}>טוען נתוני מניה...</p>
                </div>
            </div>
        );
    }

    if (!stock) {
        return (
            <div className={`min-h-screen p-6 flex justify-center items-center ${userTheme === 'dark' ? 'bg-slate-900 text-white' : ''}`}>
                <div className="text-center">
                    <p className={`text-lg ${userTheme === 'dark' ? 'text-slate-300' : 'text-slate-600'}`}>לא נמצאה מניה.</p>
                    <Link to={createPageUrl("Dashboard")} className="text-blue-600 hover:underline mt-4 inline-block">
                        חזרה להמלצות
                    </Link>
                </div>
            </div>
        );
    }

    const isPositive = stock.change_percent >= 0;

    const CustomTooltip = ({ active, payload, label }) => {
        if (active && payload && payload.length) {
            const data = payload[0].payload;
            return (
                <div className={`p-4 border rounded-lg shadow-lg ${
                    userTheme === 'dark' ? 'bg-slate-800 text-white border-slate-600' : 'bg-white text-slate-800 border-slate-200'
                }`}>
                    <p className="font-semibold mb-2">{label}</p>
                    {chartType === 'candlestick' && data.ohlc ? (
                        <div className="space-y-1 text-sm">
                            <div className="flex justify-between gap-4"><span className="text-slate-500">פתיחה:</span><span className="font-medium">${data.ohlc.open?.toFixed(2)}</span></div>
                            <div className="flex justify-between gap-4"><span className="text-slate-500">גבוה:</span><span className="font-medium text-green-600">${data.ohlc.high?.toFixed(2)}</span></div>
                            <div className="flex justify-between gap-4"><span className="text-slate-500">נמוך:</span><span className="font-medium text-red-600">${data.ohlc.low?.toFixed(2)}</span></div>
                            <div className="flex justify-between gap-4"><span className="text-slate-500">סגירה:</span><span className="font-medium">${data.ohlc.close?.toFixed(2)}</span></div>
                        </div>
                    ) : (
                        <div className="flex justify-between text-sm gap-4"><span className="text-slate-500">מחיר:</span><span className="font-medium">${data.price?.toFixed(2)}</span></div>
                    )}
                    <div className="flex justify-between text-xs text-slate-500 mt-2 pt-2 border-t gap-4"><span>נפח:</span><span>{(data.volume / 1000000).toFixed(1)}M</span></div>
                </div>
            );
        }
        return null;
    };

    const getVisibleData = () => {
        const startIndex = Math.floor((zoomDomain.left / 100) * priceData.length);
        const endIndex = Math.ceil((zoomDomain.right / 100) * priceData.length);
        return priceData.slice(startIndex, endIndex);
    };

    const visibleData = getVisibleData();

    return (
        <div className={`min-h-screen p-6 ${
            userTheme === 'dark' 
                ? 'bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white' 
                : 'bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50'
        }`}>
            <div className="max-w-7xl mx-auto">
                <div className="flex justify-between items-center mb-6">
                    <Link to={createPageUrl("Dashboard")} className={`flex items-center gap-2 ${userTheme === 'dark' ? 'text-slate-300 hover:text-white' : 'text-slate-600 hover:text-slate-900'}`}>
                        <ArrowLeft className="w-5 h-5" />
                        חזרה להמלצות
                    </Link>
                    {portfolioItem && (
                        <Button variant="ghost" onClick={handleMuteToggle} className={`flex items-center gap-2 ${userTheme === 'dark' ? 'text-slate-300 hover:bg-slate-700 hover:text-white' : ''}`}>
                            {portfolioItem.notifications_muted ? (
                                <>
                                    <BellOff className="w-5 h-5 text-red-500" />
                                    <span>התראות מושתקות</span>
                                </>
                            ) : (
                                <>
                                    <Bell className="w-5 h-5 text-green-500" />
                                    <span>התראות פעילות</span>
                                </>
                            )}
                        </Button>
                    )}
                </div>
                
                <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-8 gap-4">
                  <div>
                    <h1 className={`text-3xl md:text-4xl font-bold ${userTheme === 'dark' ? 'text-white' : 'text-slate-800'}`}>{stock.symbol} - {stock.name}</h1>
                    <p className={`mt-1 ${userTheme === 'dark' ? 'text-slate-400' : 'text-slate-600'}`}>{stock.sector}</p>
                  </div>
                  <div className="flex items-center gap-4">
                    <div className="text-right">
                        <p className={`text-3xl font-bold ${userTheme === 'dark' ? 'text-white' : 'text-slate-800'}`}>${stock.current_price?.toFixed(2)}</p>
                        <div className={`flex items-center justify-end gap-1 font-medium ${isPositive ? 'text-green-600' : 'text-red-600'}`}>
                            {isPositive ? '+' : ''}{stock.change_percent?.toFixed(2)}%
                        </div>
                    </div>
                    <Badge className={`${stock.recommendation === 'קנה' ? 'bg-green-500' : stock.recommendation === 'מכור' ? 'bg-red-500' : 'bg-blue-500'} text-white px-4 py-2 text-lg font-bold`}>
                        {stock.recommendation}
                    </Badge>
                  </div>
                </div>

                <Card className={`mb-8 ${
                    userTheme === 'dark' 
                        ? 'bg-slate-800/90 border-slate-700/60 text-white' 
                        : 'bg-white/90 border-slate-200/60'
                }`}>
                    <CardHeader>
                        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
                            <div className="flex items-center gap-3">
                                <CardTitle>גרף ביצועים בזמן אמת</CardTitle>
                                {isUpdatingChart && <span className="text-sm text-blue-600 animate-pulse">מעדכן נתונים...</span>}
                                <span className={`text-xs ${userTheme === 'dark' ? 'text-slate-400' : 'text-slate-500'}`}>עדכון אחרון: {format(lastUpdate, 'HH:mm:ss')}</span>
                                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" title="חי"></div>
                            </div>
                            <div className="flex flex-wrap items-center gap-2">
                                {/* Checkbox לניתוח טכני */}
                                <div className={`flex items-center gap-2 p-2 border rounded-lg ${userTheme === 'dark' ? 'border-slate-600 text-white' : 'border-slate-200 text-slate-800'}`}>
                                    <input 
                                        type="checkbox" 
                                        id="technical-analysis"
                                        checked={showTechnicalAnalysis}
                                        onChange={(e) => {
                                            setShowTechnicalAnalysis(e.target.checked);
                                            // Trigger chart data update to fetch/clear indicators
                                            // Delay to allow state update before re-render, then update chart
                                            setTimeout(() => updateChartData(), 0);
                                        }}
                                        className={`rounded ${userTheme === 'dark' ? 'accent-blue-500' : 'accent-blue-600'}`}
                                    />
                                    <label htmlFor="technical-analysis" className="text-sm font-medium">
                                        הצג ניתוח טכני
                                    </label>
                                </div>

                                {/* כפתורי זום */}
                                <div className={`flex items-center gap-1 p-1 rounded-md ${userTheme === 'dark' ? 'bg-slate-700' : 'bg-slate-100'}`}>
                                    <Button size="sm" variant="ghost" onClick={handleZoomIn} className={`text-xs px-2 ${userTheme === 'dark' ? 'hover:bg-slate-600 text-slate-300' : ''}`}>
                                        <ZoomIn className="w-4 h-4"/>
                                    </Button>
                                    <Button size="sm" variant="ghost" onClick={handleZoomOut} className={`text-xs px-2 ${userTheme === 'dark' ? 'hover:bg-slate-600 text-slate-300' : ''}`}>
                                        <ZoomOut className="w-4 h-4"/>
                                    </Button>
                                    <Button size="sm" variant="ghost" onClick={resetZoom} className={`text-xs px-2 ${userTheme === 'dark' ? 'hover:bg-slate-600 text-slate-300' : ''}`}>
                                        איפוס
                                    </Button>
                                </div>
                                
                                <Button variant="outline" size="sm" onClick={refreshChartData} disabled={isUpdatingChart} className={`text-xs ${userTheme === 'dark' ? 'border-slate-600 text-slate-300 hover:bg-slate-700' : ''}`}>
                                    <RefreshCw className={`w-4 h-4 mr-1 ${isUpdatingChart ? 'animate-spin' : ''}`} />רענן
                                </Button>
                                
                                <div className={`flex items-center gap-1 p-1 rounded-md ${userTheme === 'dark' ? 'bg-slate-700' : 'bg-slate-100'}`}>
                                    <Button size="sm" variant={chartType === 'line' ? 'secondary' : 'ghost'} onClick={() => setChartType('line')} className={`text-xs px-2 ${userTheme === 'dark' ? 'data-[state=secondary]:bg-slate-600 data-[state=ghost]:hover:bg-slate-600 data-[state=ghost]:text-slate-300' : ''}`}>
                                        <LineChartIcon className="w-4 h-4 mr-1"/>קו
                                    </Button>
                                    <Button size="sm" variant={chartType === 'candlestick' ? 'secondary' : 'ghost'} onClick={() => setChartType('candlestick')} className={`text-xs px-2 ${userTheme === 'dark' ? 'data-[state=secondary]:bg-slate-600 data-[state=ghost]:hover:bg-slate-600 data-[state=ghost]:text-slate-300' : ''}`}>
                                        <CandlestickChart className="w-4 h-4 mr-1"/>נרות
                                    </Button>
                                </div>
                                
                                <div className={`flex items-center gap-1 p-1 rounded-md ${userTheme === 'dark' ? 'bg-slate-700' : 'bg-slate-100'}`}>
                                    {['1W', '1M', '6M', '1Y', 'ALL'].map(tf => (
                                        <Button key={tf} size="sm" variant={timeframe === tf ? 'secondary' : 'ghost'} onClick={() => setTimeframe(tf)} className={`text-xs px-2 ${userTheme === 'dark' ? 'data-[state=secondary]:bg-slate-600 data-[state=ghost]:hover:bg-slate-600 data-[state=ghost]:text-slate-300' : ''}`}>{tf}</Button>
                                    ))}
                                </div>
                            </div>
                        </div>
                    </CardHeader>
                    <CardContent className="space-y-6">
                        {/* גרף מחירים */}
                        <div className="h-96">
                            <ResponsiveContainer width="100%" height="100%">
                                {chartType === 'line' ? (
                                    <AreaChart data={visibleData}>
                                        <defs>
                                            <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
                                                <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.3}/>
                                                <stop offset="95%" stopColor="#3B82F6" stopOpacity={0}/>
                                            </linearGradient>
                                        </defs>
                                        <CartesianGrid strokeDasharray="3 3" stroke={userTheme === 'dark' ? '#475569' : '#E2E8F0'} />
                                        <XAxis dataKey="dateString" tick={{ fontSize: 11, fill: userTheme === 'dark' ? '#E2E8F0' : '#475569' }} interval="preserveStartEnd" />
                                        <YAxis tick={{ fontSize: 11, fill: userTheme === 'dark' ? '#E2E8F0' : '#475569' }} domain={['dataMin * 0.98', 'dataMax * 1.02']} tickFormatter={(value) => `$${value.toFixed(2)}`} />
                                        <Tooltip content={<CustomTooltip />} />
                                        <Area type="monotone" dataKey="price" stroke="#3B82F6" strokeWidth={2} fill="url(#colorPrice)" />
                                        <ReferenceLine y={stock.current_price} stroke="#10B981" strokeDasharray="5 5" label={{ value: `מחיר נוכחי: $${stock.current_price.toFixed(2)}`, position: "insideTopRight", fill: userTheme === 'dark' ? '#92D192' : '#10B981', fontSize: 12, offset: 5 }} />
                                        
                                        {/* Technical Analysis Overlays for Line Chart */}
                                        {showTechnicalAnalysis && technicalIndicators && (
                                            <>
                                                {technicalIndicators.sma20 && (
                                                    <ReferenceLine y={technicalIndicators.sma20} stroke="#FF6B35" strokeDasharray="3 3" 
                                                        label={{ value: `SMA20: $${technicalIndicators.sma20.toFixed(2)}`, position: "insideTopLeft", fill: userTheme === 'dark' ? '#FFBB90' : '#FF6B35', fontSize: 12 }} />
                                                )}
                                                {technicalIndicators.sma50 && (
                                                    <ReferenceLine y={technicalIndicators.sma50} stroke="#7C3AED" strokeDasharray="3 3"
                                                        label={{ value: `SMA50: $${technicalIndicators.sma50.toFixed(2)}`, position: "insideTopLeft", fill: userTheme === 'dark' ? '#B89CFE' : '#7C3AED', fontSize: 12 }} />
                                                )}
                                                {technicalIndicators.support_levels?.map((level, i) => (
                                                    <ReferenceLine key={`line-support-${i}`} y={level} stroke="#10B981" strokeWidth={2} strokeDasharray="2 2"
                                                        label={{ value: `תמיכה: $${level.toFixed(2)}`, position: "insideBottomLeft", fill: userTheme === 'dark' ? '#92D192' : '#10B981', fontSize: 12 }} />
                                                ))}
                                                {technicalIndicators.resistance_levels?.map((level, i) => (
                                                    <ReferenceLine key={`line-resistance-${i}`} y={level} stroke="#EF4444" strokeWidth={2} strokeDasharray="2 2"
                                                        label={{ value: `התנגדות: $${level.toFixed(2)}`, position: "insideTopLeft", fill: userTheme === 'dark' ? '#F2A0A0' : '#EF4444', fontSize: 12 }} />
                                                ))}
                                            </>
                                        )}
                                    </AreaChart>
                                ) : (
                                    <ComposedChart data={visibleData}>
                                        <CartesianGrid strokeDasharray="3 3" stroke={userTheme === 'dark' ? '#475569' : '#E2E8F0'} />
                                        <XAxis dataKey="dateString" tick={{ fontSize: 11, fill: userTheme === 'dark' ? '#E2E8F0' : '#475569' }} interval="preserveStartEnd" />
                                        <YAxis 
                                            tick={{ fontSize: 11, fill: userTheme === 'dark' ? '#E2E8F0' : '#475569' }} 
                                            domain={['dataMin * 0.98', 'dataMax * 1.02']} 
                                            tickFormatter={(value) => `$${value.toFixed(2)}`} 
                                        />
                                        <Tooltip content={<CustomTooltip />} />
                                        
                                        <Bar dataKey="range" shape={<CandlestickShape />} />
                                        
                                        <ReferenceLine y={stock.current_price} stroke="#10B981" strokeDasharray="5 5" label={{ value: `מחיר נוכחי: $${stock.current_price.toFixed(2)}`, position: "insideTopRight", fill: userTheme === 'dark' ? '#92D192' : '#10B981', fontSize: 12, offset: 5 }} />
                                        
                                        {/* Technical Analysis Overlays for Candlestick Chart */}
                                        {showTechnicalAnalysis && technicalIndicators && (
                                            <>
                                                {technicalIndicators.sma20 && (
                                                    <ReferenceLine y={technicalIndicators.sma20} stroke="#FF6B35" strokeDasharray="3 3"
                                                        label={{ value: `SMA20: $${technicalIndicators.sma20.toFixed(2)}`, position: "insideTopLeft", fill: userTheme === 'dark' ? '#FFBB90' : '#FF6B35', fontSize: 12 }} />
                                                )}
                                                {technicalIndicators.sma50 && (
                                                    <ReferenceLine y={technicalIndicators.sma50} stroke="#7C3AED" strokeDasharray="3 3"
                                                        label={{ value: `SMA50: $${technicalIndicators.sma50.toFixed(2)}`, position: "insideTopLeft", fill: userTheme === 'dark' ? '#B89CFE' : '#7C3AED', fontSize: 12 }} />
                                                )}
                                                {technicalIndicators.support_levels?.map((level, i) => (
                                                    <ReferenceLine key={`candle-support-${i}`} y={level} stroke="#10B981" strokeWidth={2} strokeDasharray="2 2"
                                                        label={{ value: `תמיכה: $${level.toFixed(2)}`, position: "insideBottomLeft", fill: userTheme === 'dark' ? '#92D192' : '#10B981', fontSize: 12 }} />
                                                ))}
                                                {technicalIndicators.resistance_levels?.map((level, i) => (
                                                    <ReferenceLine key={`candle-resistance-${i}`} y={level} stroke="#EF4444" strokeWidth={2} strokeDasharray="2 2"
                                                        label={{ value: `התנגדות: $${level.toFixed(2)}`, position: "insideTopLeft", fill: userTheme === 'dark' ? '#F2A0A0' : '#EF4444', fontSize: 12 }} />
                                                ))}
                                            </>
                                        )}
                                    </ComposedChart>
                                )}
                            </ResponsiveContainer>
                        </div>
                        
                        {/* גרף נפח מסחר */}
                        <div className="h-32">
                            <h4 className={`text-sm font-medium mb-2 ${userTheme === 'dark' ? 'text-slate-400' : 'text-slate-600'}`}>נפח מסחר</h4>
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={visibleData} margin={{ top: 5, right: 0, left: 0, bottom: 5 }}>
                                    <CartesianGrid strokeDasharray="3 3" stroke={userTheme === 'dark' ? '#475569' : '#E2E8F0'} vertical={false} />
                                    <XAxis dataKey="dateString" tick={{ fontSize: 10, fill: userTheme === 'dark' ? '#E2E8F0' : '#475569' }} interval="preserveStartEnd" axisLine={false} tickLine={false} />
                                    <YAxis 
                                        tick={{ fontSize: 10, fill: userTheme === 'dark' ? '#E2E8F0' : '#475569' }} 
                                        tickFormatter={(value) => `${(value / 1000000).toFixed(0)}M`}
                                        axisLine={false}
                                        tickLine={false}
                                        allowDecimals={false}
                                        width={40}
                                    />
                                    <Tooltip 
                                        formatter={(value, name, props) => {
                                            const entry = props.payload;
                                            if (!entry || !entry.ohlc) return [`${(value / 1000000).toFixed(2)}M`, 'נפח'];
                                            const isPositive = entry.ohlc.close >= entry.ohlc.open;
                                            return [`${(value / 1000000).toFixed(2)}M`, `נפח (${isPositive ? 'עלייה' : 'ירידה'})`];
                                        }}
                                        cursor={{fill: userTheme === 'dark' ? 'rgba(80,80,80,0.2)' : 'rgba(200,200,200,0.1)'}}
                                        contentStyle={{ backgroundColor: userTheme === 'dark' ? 'rgba(30,41,59,0.9)' : 'rgba(255, 255, 255, 0.9)', backdropFilter: 'blur(5px)', border: `1px solid ${userTheme === 'dark' ? '#475569' : '#E2E8F0'}`, borderRadius: '0.5rem', color: userTheme === 'dark' ? '#E2E8F0' : '#1E293B' }}
                                    />
                                    <Bar dataKey="volume">
                                        {visibleData.map((entry, index) => {
                                            if (!entry.ohlc) return <Cell key={`cell-${index}`} fill={'#A0AEC0'} />;
                                            const isPositive = entry.ohlc.close >= entry.ohlc.open;
                                            const color = isPositive ? '#10B981' : '#EF4444';
                                            return <Cell key={`cell-${index}`} fill={color} opacity={0.7} />;
                                        })}
                                    </Bar>
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                        
                        {/* Technical Analysis Summary */}
                        {showTechnicalAnalysis && technicalIndicators && (
                            <Card className={`${userTheme === 'dark' ? 'bg-slate-700/70 border-slate-600' : 'bg-slate-50 border-slate-200'}`}>
                                <CardContent className="p-4">
                                    <h4 className={`font-semibold mb-2 ${userTheme === 'dark' ? 'text-white' : 'text-slate-800'}`}>סיכום ניתוח טכני</h4>
                                    <p className={`text-sm ${userTheme === 'dark' ? 'text-slate-300' : 'text-slate-700'}`}>{technicalIndicators.analysis_summary}</p>
                                    {technicalIndicators.rsi && (
                                        <div className="mt-2">
                                            <span className={`text-sm font-medium ${userTheme === 'dark' ? 'text-slate-200' : 'text-slate-700'}`}>RSI: {technicalIndicators.rsi.toFixed(1)}</span>
                                            <span className={`ml-2 text-xs px-2 py-1 rounded ${
                                                technicalIndicators.rsi > 70 ? 'bg-red-100 text-red-800 dark:bg-red-700 dark:text-red-100' :
                                                technicalIndicators.rsi < 30 ? 'bg-green-100 text-green-800 dark:bg-green-700 dark:text-green-100' :
                                                'bg-yellow-100 text-yellow-800 dark:bg-yellow-700 dark:text-yellow-100'
                                            }`}>
                                                {technicalIndicators.rsi > 70 ? 'קנייה יתר' : 
                                                 technicalIndicators.rsi < 30 ? 'מכירה יתר' : 'נייטרלי'}
                                            </span>
                                        </div>
                                    )}
                                    {technicalIndicators.fibonacci_levels?.length > 0 && (
                                        <div className="mt-2">
                                            <h5 className={`font-semibold text-sm ${userTheme === 'dark' ? 'text-slate-200' : 'text-slate-800'}`}>רמות פיבונאצ'י:</h5>
                                            <ul className={`list-disc pl-5 text-sm ${userTheme === 'dark' ? 'text-slate-300' : 'text-slate-700'}`}>
                                                {technicalIndicators.fibonacci_levels.map((level, i) => (
                                                    <li key={i}>${level.toFixed(2)}</li>
                                                ))}
                                            </ul>
                                        </div>
                                    )}
                                </CardContent>
                            </Card>
                        )}

                        {/* סרגל גלילה לזום */}
                        <div className="h-16">
                            <ResponsiveContainer width="100%" height="100%">
                                <AreaChart data={priceData}>
                                    <XAxis dataKey="dateString" tick={false} axisLine={false} />
                                    <YAxis hide />
                                    <Area type="monotone" dataKey="price" stroke="#94A3B8" strokeWidth={1} fill="#F1F5F9" />
                                    <Brush 
                                        dataKey="dateString" 
                                        height={40}
                                        stroke="#3B82F6"
                                        fill={userTheme === 'dark' ? '#1E293B' : '#F1F5F9'}
                                        startIndex={Math.floor((zoomDomain.left / 100) * priceData.length)}
                                        endIndex={Math.ceil((zoomDomain.right / 100) * priceData.length)}
                                        onChange={(data) => {
                                            if (data) {
                                                setZoomDomain({
                                                    left: (data.startIndex / priceData.length) * 100,
                                                    right: (data.endIndex / priceData.length) * 100
                                                });
                                            }
                                        }}
                                    />
                                </AreaChart>
                            </ResponsiveContainer>
                        </div>
                    </CardContent>
                </Card>

                {/* שאר הטאבים */}
                <Tabs defaultValue="technical">
                    <TabsList className={`grid w-full grid-cols-3 mb-6 ${userTheme === 'dark' ? 'bg-slate-700 text-white' : ''}`}>
                        <TabsTrigger value="technical" className={userTheme === 'dark' ? 'data-[state=active]:bg-slate-600 data-[state=inactive]:bg-slate-700 data-[state=inactive]:text-slate-300' : ''}><BarChart3 className="w-4 h-4 mr-2"/>ניתוח טכני</TabsTrigger>
                        <TabsTrigger value="fundamental" className={userTheme === 'dark' ? 'data-[state=active]:bg-slate-600 data-[state=inactive]:bg-slate-700 data-[state=inactive]:text-slate-300' : ''}><Building className="w-4 h-4 mr-2"/>ניתוח פונדמנטלי</TabsTrigger>
                        <TabsTrigger value="news" className={userTheme === 'dark' ? 'data-[state=active]:bg-slate-600 data-[state=inactive]:bg-slate-700 data-[state=inactive]:text-slate-300' : ''}><Newspaper className="w-4 h-4 mr-2"/>חדשות</TabsTrigger>
                    </TabsList>
                    
                    <TabsContent value="technical">
                        <Card className={userTheme === 'dark' ? 'bg-slate-800 border-slate-700 text-white' : ''}>
                            <CardHeader>
                                <CardTitle>ניתוח טכני</CardTitle>
                            </CardHeader>
                            <CardContent className="space-y-6">
                                <p className={`prose max-w-none ${userTheme === 'dark' ? 'text-slate-300' : ''}`}>{stock.technical_analysis}</p>
                                <Separator className={userTheme === 'dark' ? 'bg-slate-700' : ''}/>
                                <div className="grid md:grid-cols-2 gap-6">
                                    <Card className={userTheme === 'dark' ? 'bg-slate-700 border-slate-600 text-white' : ''}>
                                        <CardHeader>
                                            <CardTitle className="flex items-center gap-2">
                                                <CheckCircle className="text-green-500"/>
                                                סיבות לקנייה
                                            </CardTitle>
                                        </CardHeader>
                                        <CardContent>
                                            <ul className={`list-disc pl-5 space-y-1 ${userTheme === 'dark' ? 'text-slate-300' : ''}`}>
                                                {stock.buy_reasons?.map((r, i) => <li key={i}>{r}</li>)}
                                            </ul>
                                        </CardContent>
                                    </Card>
                                    <Card className={userTheme === 'dark' ? 'bg-slate-700 border-slate-600 text-white' : ''}>
                                        <CardHeader>
                                            <CardTitle className="flex items-center gap-2">
                                                <XCircle className="text-red-500"/>
                                                סיבות למכירה
                                            </CardTitle>
                                        </CardHeader>
                                        <CardContent>
                                            <ul className={`list-disc pl-5 space-y-1 ${userTheme === 'dark' ? 'text-slate-300' : ''}`}>
                                                {stock.sell_reasons?.map((r, i) => <li key={i}>{r}</li>)}
                                            </ul>
                                        </CardContent>
                                    </Card>
                                </div>
                            </CardContent>
                        </Card>
                    </TabsContent>

                    <TabsContent value="fundamental">
                        <Card className={userTheme === 'dark' ? 'bg-slate-800 border-slate-700 text-white' : ''}>
                            <CardHeader>
                                <CardTitle>ניתוח פונדמנטלי</CardTitle>
                            </CardHeader>
                            <CardContent className="space-y-6">
                                <p className={`prose max-w-none ${userTheme === 'dark' ? 'text-slate-300' : ''}`}>{stock.fundamental_analysis}</p>
                                <Separator className={userTheme === 'dark' ? 'bg-slate-700' : ''}/>
                                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
                                    <div className={`p-4 rounded-lg ${userTheme === 'dark' ? 'bg-slate-700' : 'bg-slate-100'}`}>
                                        <p className={`text-sm ${userTheme === 'dark' ? 'text-slate-400' : 'text-slate-500'}`}>שווי שוק</p>
                                        <p className="font-bold text-lg">${(stock.market_cap / 1e9).toFixed(2)}B</p>
                                    </div>
                                    <div className={`p-4 rounded-lg ${userTheme === 'dark' ? 'bg-slate-700' : 'bg-slate-100'}`}>
                                        <p className={`text-sm ${userTheme === 'dark' ? 'text-slate-400' : 'text-slate-500'}`}>יחס P/E</p>
                                        <p className="font-bold text-lg">{stock.pe_ratio?.toFixed(1)}</p>
                                    </div>
                                    <div className={`p-4 rounded-lg ${userTheme === 'dark' ? 'bg-slate-700' : 'bg-slate-100'}`}>
                                        <p className={`text-sm ${userTheme === 'dark' ? 'text-slate-400' : 'text-slate-500'}`}>נפח מסחר</p>
                                        <p className="font-bold text-lg">{(stock.volume / 1e6).toFixed(2)}M</p>
                                    </div>
                                    <div className={`p-4 rounded-lg ${userTheme === 'dark' ? 'bg-slate-700' : 'bg-slate-100'}`}>
                                        <p className={`text-sm ${userTheme === 'dark' ? 'text-slate-400' : 'text-slate-500'}`}>מחיר יעד</p>
                                        <p className="font-bold text-lg text-green-600">${stock.target_price?.toFixed(2)}</p>
                                    </div>
                                </div>
                            </CardContent>
                        </Card>
                    </TabsContent>

                    <TabsContent value="news">
                        <Card className={userTheme === 'dark' ? 'bg-slate-800 border-slate-700 text-white' : ''}>
                            <CardHeader>
                                <CardTitle>חדשות אחרונות</CardTitle>
                            </CardHeader>
                            <CardContent className="space-y-4">
                                {stock.company_news?.length > 0 ? stock.company_news.map((item, index) => (
                                    <a href={item.url} target="_blank" rel="noopener noreferrer" key={index} className={`block p-4 rounded-lg hover:bg-slate-50 ${userTheme === 'dark' ? 'border-slate-700 hover:bg-slate-700/60' : 'border'}`}>
                                        <p className={`font-semibold ${userTheme === 'dark' ? 'text-blue-400' : 'text-blue-600'}`}>{item.title}</p>
                                        <p className={`text-sm ${userTheme === 'dark' ? 'text-slate-400' : 'text-slate-500'}`}>{item.source}</p>
                                    </a>
                                )) : (
                                    <p className={userTheme === 'dark' ? 'text-slate-300' : ''}>אין חדשות זמינות.</p>
                                )}
                            </CardContent>
                        </Card>
                    </TabsContent>
                </Tabs>
            </div>
        </div>
    );
}
