import React, useState, useEffect, useCallback from "react"
import Stock, Favorites, User from Entities
import { InvokeLLM } from "@/integrations/Core";
import { motion, AnimatePresence } from "framer-motion";
import { TrendingUp, Zap, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { useNavigate } from "react-router-dom";
import { createPageUrl } from "@/utils";

import StockCard from "../components/stocks/StockCard";
import StockFilters from "../components/stocks/StockFilters";

export default function Dashboard() {
  const [stocks, setStocks] = useState([]);
  const [filteredStocks, setFilteredStocks] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [filters, setFilters] = useState({
    riskLevel: 'הכל',
    recommendation: 'הכל',
    sortBy: 'return_desc'
  });
  const [isGenerating, setIsGenerating] = useState(false);
  const navigate = useNavigate();

  const loadStocks = useCallback(async () => {
    setIsLoading(true);
    try {
      const stocksData = await Stock.list('-expected_return');
      setStocks(stocksData);
    } catch (error) {
      console.error("Error loading stocks:", error);
    }
    setIsLoading(false);
  }, []);

  useEffect(() => {
    loadStocks();
    
    const interval = setInterval(() => {
      loadStocks();
    }, 300000);

    return () => clearInterval(interval);
  }, [loadStocks]);

  useEffect(() => {
    applyFilters();
  }, [stocks, filters]);

  const generateNewRecommendations = async () => {
    setIsGenerating(true);
    try {
      const user = await User.me();
      const analysisPreference = user.analysis_preference || 'משולב';
      
      let analysisInstructions = '';
      if (analysisPreference === 'טכני') {
        analysisInstructions = "התמקד בניתוח טכני בלבד.";
      } else if (analysisPreference === 'פונדמנטלי') {
        analysisInstructions = "התמקד בניתוח פונדמנטלי בלבד, כולל יחסים פיננסיים, חדשות ודוחות כספיים.";
      } else {
        analysisInstructions = "שלב ניתוח טכני ופונדמנטלי.";
      }
      
      const prompt = `
      אתה מנתח מניות מקצועי. צור 50 המלצות מניות אמריקאיות מגוונות.
      ${analysisInstructions}
      
      כלול מניות מכל רמות הסיכון.
      
      לכל מניה תן:
      - ניתוח טכני מפורט (אם רלוונטי)
      - ניתוח פונדמנטלי מפורט (אם רלוונטי), כולל חדשות אחרונות
      - 3-4 סיבות לקנייה, 2-3 סיבות למכירה
      - המלצה ברורה (קנה/מכור/החזק)
      - מחיר נוכחי ומחיר יעד מציאותיים
      - נתונים פיננסיים כמו שווי שוק, יחס P/E, נפח מסחר.
      
      השתמש בנתונים עדכניים וסבירים לשוק האמריקאי.
      `;

      const result = await InvokeLLM({
        prompt,
        add_context_from_internet: true,
        response_json_schema: {
          type: "object",
          properties: {
            stocks: {
              type: "array",
              items: {
                type: "object",
                properties: {
                  symbol: { type: "string" },
                  name: { type: "string" },
                  current_price: { type: "number" },
                  expected_return: { type: "number" },
                  risk_level: { type: "string", enum: ["סולידי", "בינוני", "סיכון גבוה"] },
                  recommendation: { type: "string", enum: ["קנה", "מכור", "החזק"] },
                  target_price: { type: "number" },
                  technical_analysis: { type: "string" },
                  fundamental_analysis: { type: "string" },
                  company_news: { type: "array", items: {type: "object", properties: {title: {type:"string"}, url: {type:"string"}, source:{type:"string"}}}},
                  buy_reasons: { type: "array", items: { type: "string" } },
                  sell_reasons: { type: "array", items: { type: "string" } },
                  sector: { type: "string" },
                  market_cap: { type: "number" },
                  pe_ratio: { type: "number" },
                  volume: { type: "number" },
                  change_percent: { type: "number" }
                }
              }
            }
          }
        }
      });

      const existingStocks = await Stock.list();
      for (const stock of existingStocks) {
        await Stock.delete(stock.id);
      }

      if (result.stocks && result.stocks.length > 0) {
        await Stock.bulkCreate(result.stocks);
        await loadStocks();
      }
    } catch (error) {
      console.error("Error generating recommendations:", error);
    }
    setIsGenerating(false);
  };

  const applyFilters = () => {
    let filtered = [...stocks];

    if (filters.riskLevel !== 'הכל') {
      filtered = filtered.filter(stock => stock.risk_level === filters.riskLevel);
    }

    if (filters.recommendation !== 'הכל') {
      filtered = filtered.filter(stock => stock.recommendation === filters.recommendation);
    }

    switch (filters.sortBy) {
      case 'name':
        filtered.sort((a, b) => a.name.localeCompare(b.name, 'he'));
        break;
      case 'price_asc':
        filtered.sort((a, b) => a.current_price - b.current_price);
        break;
      case 'price_desc':
        filtered.sort((a, b) => b.current_price - a.current_price);
        break;
      case 'return_desc':
        filtered.sort((a, b) => b.expected_return - a.expected_return);
        break;
      case 'return_asc':
        filtered.sort((a, b) => a.expected_return - b.expected_return);
        break;
      default:
        break;
    }

    setFilteredStocks(filtered);
  };

  const handleAddToFavorites = async (stock) => {
    try {
      await Favorites.create({
        stock_symbol: stock.symbol,
        stock_name: stock.name
      });
    } catch (error) {
      console.error("Error adding to favorites:", error);
    }
  };

  const handleAddToPortfolio = (stock) => {
    navigate(createPageUrl(`Portfolio?stock=${stock.symbol}`));
  };

  const handleAnalyze = (stock) => {
    navigate(createPageUrl(`StockDetails?symbol=${stock.symbol}`));
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 p-6">
      <div className="max-w-7xl mx-auto">
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-8 gap-4">
          <div>
            <h1 className="text-3xl md:text-4xl font-bold text-slate-800 mb-2">
              המלצות מניות חכמות
            </h1>
            <p className="text-slate-600 text-lg">
              ניתוח מתקדם עם בינה מלאכותית לקבלת החלטות השקעה מושכלות
            </p>
          </div>
          
          <div className="flex gap-3">
            <Button
              onClick={loadStocks}
              variant="outline"
              disabled={isLoading}
              className="border-slate-300 hover:bg-slate-50"
            >
              <RefreshCw className={`w-4 h-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
              רענן
            </Button>
            <Button
              onClick={generateNewRecommendations}
              disabled={isGenerating}
              className="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white font-medium"
            >
              <Zap className={`w-4 h-4 mr-2 ${isGenerating ? 'animate-pulse' : ''}`} />
              {isGenerating ? 'מייצר המלצות...' : 'המלצות חדשות'}
            </Button>
          </div>
        </div>

        <div className="grid md:grid-cols-4 gap-6 mb-8">
          <div className="bg-white/90 backdrop-blur-sm rounded-xl p-6 border border-slate-200/60">
            <div className="flex items-center gap-3">
              <div className="w-12 h-12 bg-green-100 rounded-xl flex items-center justify-center">
                <TrendingUp className="w-6 h-6 text-green-600" />
              </div>
              <div>
                <p className="text-sm text-slate-600">סה"כ המלצות</p>
                <p className="text-2xl font-bold text-slate-800">{stocks.length}</p>
              </div>
            </div>
          </div>
          
          <div className="bg-white/90 backdrop-blur-sm rounded-xl p-6 border border-slate-200/60">
            <div className="flex items-center gap-3">
              <div className="w-12 h-12 bg-blue-100 rounded-xl flex items-center justify-center">
                <span className="text-blue-600 font-bold">קנה</span>
              </div>
              <div>
                <p className="text-sm text-slate-600">המלצות קנייה</p>
                <p className="text-2xl font-bold text-slate-800">
                  {stocks.filter(s => s.recommendation === 'קנה').length}
                </p>
              </div>
            </div>
          </div>

          <div className="bg-white/90 backdrop-blur-sm rounded-xl p-6 border border-slate-200/60">
            <div className="flex items-center gap-3">
              <div className="w-12 h-12 bg-yellow-100 rounded-xl flex items-center justify-center">
                <span className="text-yellow-600 font-bold text-xs">בינוני</span>
              </div>
              <div>
                <p className="text-sm text-slate-600">סיכון בינוני</p>
                <p className="text-2xl font-bold text-slate-800">
                  {stocks.filter(s => s.risk_level === 'בינוני').length}
                </p>
              </div>
            </div>
          </div>

          <div className="bg-white/90 backdrop-blur-sm rounded-xl p-6 border border-slate-200/60">
            <div className="flex items-center gap-3">
              <div className="w-12 h-12 bg-emerald-100 rounded-xl flex items-center justify-center">
                <span className="text-emerald-600 font-bold text-lg">%</span>
              </div>
              <div>
                <p className="text-sm text-slate-600">תשואה ממוצעת</p>
                <p className="text-2xl font-bold text-slate-800">
                  {stocks.length > 0 ? 
                    (stocks.reduce((sum, s) => sum + s.expected_return, 0) / stocks.length).toFixed(1) : 
                    '0'
                  }%
                </p>
              </div>
            </div>
          </div>
        </div>

        <div className="mb-8">
          <StockFilters filters={filters} onFiltersChange={setFilters} />
        </div>

        {isLoading ? (
          <div className="grid md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {Array(8).fill(0).map((_, i) => (
              <div key={i} className="bg-white/90 rounded-xl p-6 border border-slate-200/60">
                <Skeleton className="h-6 w-20 mb-2" />
                <Skeleton className="h-4 w-32 mb-4" />
                <Skeleton className="h-8 w-24 mb-4" />
                <div className="flex gap-2 mb-4">
                  <Skeleton className="h-6 w-16" />
                  <Skeleton className="h-6 w-16" />
                </div>
                <Skeleton className="h-10 w-full" />
              </div>
            ))}
          </div>
        ) : (
          <AnimatePresence>
            <div className="grid md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
              {filteredStocks.map((stock, index) => (
                <motion.div
                  key={stock.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.05 }}
                >
                  <StockCard
                    stock={stock}
                    onAnalyze={handleAnalyze}
                    onAddToFavorites={handleAddToFavorites}
                    onAddToPortfolio={handleAddToPortfolio}
                  />
                </motion.div>
              ))}
            </div>
          </AnimatePresence>
        )}
      </div>
    </div>
  );
}