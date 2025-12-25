
import React, { useState, useEffect } from "react";
import { Favorites, Stock } from "@/entities/all";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Heart, Trash2, TrendingUp, TrendingDown, BarChart3 } from "lucide-react";
import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import { createPageUrl } from "@/utils";

export default function FavoritesPage() {
  const [favorites, setFavorites] = useState([]);
  const [stocksData, setStocksData] = useState({});
  const [isLoading, setIsLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    loadFavorites();
  }, []);

  const loadFavorites = async () => {
    setIsLoading(true);
    try {
      const favoritesData = await Favorites.list('-created_date');
      setFavorites(favoritesData);
      
      // טעינת נתוני מניות מפורטים
      const stocksInfo = {};
      for (const fav of favoritesData) {
        const stocks = await Stock.filter({ symbol: fav.stock_symbol });
        if (stocks.length > 0) {
          stocksInfo[fav.stock_symbol] = stocks[0];
        }
      }
      setStocksData(stocksInfo);
    } catch (error) {
      console.error("Error loading favorites:", error);
    }
    setIsLoading(false);
  };

  const handleRemoveFromFavorites = async (id) => {
    try {
      await Favorites.delete(id);
      loadFavorites();
    } catch (error) {
      console.error("Error removing from favorites:", error);
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
      <div className="max-w-6xl mx-auto">
        {/* כותרת */}
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-8 gap-4">
          <div>
            <h1 className="text-3xl md:text-4xl font-bold text-slate-800 mb-2">
              המניות המועדפות שלי
            </h1>
            <p className="text-slate-600 text-lg">
              מעקב אחר המניות שמעניינות אותך הכי הרבה
            </p>
          </div>
          
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 text-slate-600">
              <Heart className="w-5 h-5 text-red-500" />
              <span className="font-medium">{favorites.length} מניות מועדפות</span>
            </div>
          </div>
        </div>

        {/* רשימת מועדפים */}
        {favorites.length === 0 ? (
          <Card className="bg-white/90 backdrop-blur-sm border-slate-200/60">
            <CardContent className="text-center py-16">
              <Heart className="w-16 h-16 text-slate-300 mx-auto mb-4" />
              <h3 className="text-xl font-semibold text-slate-600 mb-2">
                אין מועדפים עדיין
              </h3>
              <p className="text-slate-500 mb-6">
                התחל להוסיף מניות למועדפ
                ים מדף ההמלצות
              </p>
              <Button
                onClick={() => window.location.href = '/Dashboard'}
                className="bg-blue-600 hover:bg-blue-700"
              >
                לעבור להמלצות מניות
              </Button>
            </CardContent>
          </Card>
        ) : (
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {favorites.map((favorite, index) => {
              const stockData = stocksData[favorite.stock_symbol];
              const isPositive = stockData?.change_percent >= 0;
              
              return (
                <motion.div
                  key={favorite.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                >
                  <Card className="bg-white/90 backdrop-blur-sm border-slate-200/60 hover:shadow-xl transition-all duration-300 group h-full">
                    <CardHeader className="pb-4">
                      <div className="flex justify-between items-start">
                        <div>
                          <h3 className="font-bold text-lg text-slate-800">{favorite.stock_symbol}</h3>
                          <p className="text-sm text-slate-600 font-medium">{favorite.stock_name}</p>
                          {stockData && (
                            <p className="text-xs text-slate-500 mt-1">{stockData.sector}</p>
                          )}
                        </div>
                        <Button
                          variant="ghost"
                          size="icon"
                          onClick={() => handleRemoveFromFavorites(favorite.id)}
                          className="text-red-500 hover:text-red-700 hover:bg-red-50"
                        >
                          <Trash2 className="w-4 h-4" />
                        </Button>
                      </div>
                    </CardHeader>
                    
                    <CardContent className="space-y-4">
                      {stockData ? (
                        <>
                          <div className="flex justify-between items-center">
                            <div>
                              <p className="text-2xl font-bold text-slate-800">
                                ${stockData.current_price?.toFixed(2)}
                              </p>
                              <div className={`flex items-center gap-1 text-sm font-medium ${
                                isPositive ? 'text-green-600' : 'text-red-600'
                              }`}>
                                {isPositive ? 
                                  <TrendingUp className="w-4 h-4" /> : 
                                  <TrendingDown className="w-4 h-4" />
                                }
                                {isPositive ? '+' : ''}{stockData.change_percent?.toFixed(2)}%
                              </div>
                            </div>
                            <div className="text-left">
                              <p className="text-lg font-bold text-emerald-600">
                                +{stockData.expected_return?.toFixed(1)}% צפוי
                              </p>
                            </div>
                          </div>

                          <div className="flex gap-2 flex-wrap">
                            <Badge className={`${
                              stockData.risk_level === 'סולידי' ? 'bg-green-100 text-green-800 border-green-200' :
                              stockData.risk_level === 'בינוני' ? 'bg-yellow-100 text-yellow-800 border-yellow-200' :
                              'bg-red-100 text-red-800 border-red-200'
                            } border font-medium`}>
                              {stockData.risk_level}
                            </Badge>
                            <Badge className={`${
                              stockData.recommendation === 'קנה' ? 'bg-green-500' :
                              stockData.recommendation === 'מכור' ? 'bg-red-500' :
                              'bg-blue-500'
                            } text-white font-medium`}>
                              {stockData.recommendation}
                            </Badge>
                          </div>

                          <div className="grid grid-cols-2 gap-4 text-sm">
                            <div>
                              <p className="text-slate-500">יחס P/E</p>
                              <p className="font-semibold text-slate-700">
                                {stockData.pe_ratio?.toFixed(1)}
                              </p>
                            </div>
                            <div>
                              <p className="text-slate-500">מחיר יעד</p>
                              <p className="font-semibold text-slate-700">
                                ${stockData.target_price?.toFixed(2)}
                              </p>
                            </div>
                          </div>

                          <div className="flex gap-2 pt-2">
                            <Button 
                              size="sm"
                              className="flex-1 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white font-medium"
                              onClick={() => handleAnalyze(stockData)}
                            >
                              <BarChart3 className="w-4 h-4 mr-2" />
                              ניתוח מפורט
                            </Button>
                            <Button 
                              size="sm"
                              variant="outline"
                              onClick={() => handleAddToPortfolio(stockData)}
                              className="border-slate-300 hover:bg-slate-50"
                            >
                              הוסף לתיק
                            </Button>
                          </div>
                        </>
                      ) : (
                        <div className="text-center py-8">
                          <p className="text-slate-500">טוען נתונים...</p>
                        </div>
                      )}
                    </CardContent>
                  </Card>
                </motion.div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
