import React from "react";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { TrendingUp, TrendingDown, Target, Heart, BarChart3 } from "lucide-react";
import { motion } from "framer-motion";

const riskColors = {
  "סולידי": "bg-green-100 text-green-800 border-green-200",
  "בינוני": "bg-yellow-100 text-yellow-800 border-yellow-200", 
  "סיכון גבוה": "bg-red-100 text-red-800 border-red-200"
};

const recommendationColors = {
  "קנה": "bg-green-500 text-white",
  "מכור": "bg-red-500 text-white",
  "החזק": "bg-blue-500 text-white"
};

export default function StockCard({ stock, onAnalyze, onAddToFavorites, onAddToPortfolio }) {
  const isPositive = stock.change_percent >= 0;
  
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="h-full"
    >
      <Card className="h-full bg-white/90 backdrop-blur-sm border-slate-200/60 hover:shadow-xl transition-all duration-300 group">
        <CardHeader className="pb-4">
          <div className="flex justify-between items-start">
            <div>
              <h3 className="font-bold text-lg text-slate-800">{stock.symbol}</h3>
              <p className="text-sm text-slate-600 font-medium">{stock.name}</p>
              <p className="text-xs text-slate-500 mt-1">{stock.sector}</p>
            </div>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => onAddToFavorites(stock)}
              className="text-slate-400 hover:text-red-500 transition-colors"
            >
              <Heart className="w-4 h-4" />
            </Button>
          </div>
        </CardHeader>
        
        <CardContent className="space-y-4">
          <div className="flex justify-between items-center">
            <div>
              <p className="text-2xl font-bold text-slate-800">${stock.current_price?.toFixed(2)}</p>
              <div className={`flex items-center gap-1 text-sm font-medium ${
                isPositive ? 'text-green-600' : 'text-red-600'
              }`}>
                {isPositive ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
                {isPositive ? '+' : ''}{stock.change_percent?.toFixed(2)}%
              </div>
            </div>
            <div className="text-left">
              <div className="flex items-center gap-1 text-sm text-slate-600 mb-1">
                <Target className="w-4 h-4" />
                <span>יעד: ${stock.target_price?.toFixed(2)}</span>
              </div>
              <p className="text-lg font-bold text-emerald-600">+{stock.expected_return?.toFixed(1)}%</p>
            </div>
          </div>

          <div className="flex gap-2 flex-wrap">
            <Badge className={`${riskColors[stock.risk_level]} border font-medium`}>
              {stock.risk_level}
            </Badge>
            <Badge className={`${recommendationColors[stock.recommendation]} font-medium`}>
              {stock.recommendation}
            </Badge>
          </div>

          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <p className="text-slate-500">יחס P/E</p>
              <p className="font-semibold text-slate-700">{stock.pe_ratio?.toFixed(1)}</p>
            </div>
            <div>
              <p className="text-slate-500">נפח מסחר</p>
              <p className="font-semibold text-slate-700">{(stock.volume / 1000000)?.toFixed(1)}M</p>
            </div>
          </div>

          <div className="flex gap-2 pt-2">
            <Button 
              onClick={() => onAnalyze(stock)}
              className="flex-1 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white font-medium"
            >
              <BarChart3 className="w-4 h-4 mr-2" />
              ניתוח מפורט
            </Button>
            <Button 
              variant="outline"
              onClick={() => onAddToPortfolio(stock)}
              className="border-slate-300 hover:bg-slate-50"
            >
              הוסף לתיק
            </Button>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}