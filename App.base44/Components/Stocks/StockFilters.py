import React from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Filter, SortAsc, SortDesc } from "lucide-react";

export default function StockFilters({ filters, onFiltersChange }) {
  const handleRiskFilter = (riskLevel) => {
    onFiltersChange({ ...filters, riskLevel });
  };

  const handleSortChange = (sortBy) => {
    onFiltersChange({ ...filters, sortBy });
  };

  const handleRecommendationFilter = (recommendation) => {
    onFiltersChange({ ...filters, recommendation });
  };

  return (
    <Card className="bg-white/90 backdrop-blur-sm border-slate-200/60">
      <CardContent className="p-6">
        <div className="flex flex-wrap gap-4 items-center justify-between">
          <div className="flex items-center gap-2">
            <Filter className="w-5 h-5 text-slate-600" />
            <span className="font-medium text-slate-700">סינון והדרכה:</span>
          </div>
          
          <div className="flex flex-wrap gap-3">
            {/* סינון לפי רמת סיכון */}
            <div className="flex gap-2">
              {['הכל', 'סולידי', 'בינוני', 'סיכון גבוה'].map((risk) => (
                <Button
                  key={risk}
                  variant={filters.riskLevel === risk ? "default" : "outline"}
                  size="sm"
                  onClick={() => handleRiskFilter(risk)}
                  className={filters.riskLevel === risk ? 
                    "bg-blue-600 hover:bg-blue-700" : 
                    "border-slate-300 hover:bg-slate-50"
                  }
                >
                  {risk}
                </Button>
              ))}
            </div>

            {/* סינון לפי המלצה */}
            <Select value={filters.recommendation} onValueChange={handleRecommendationFilter}>
              <SelectTrigger className="w-32">
                <SelectValue placeholder="המלצה" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="הכל">כל ההמלצות</SelectItem>
                <SelectItem value="קנה">קנה</SelectItem>
                <SelectItem value="מכור">מכור</SelectItem>
                <SelectItem value="החזק">החזק</SelectItem>
              </SelectContent>
            </Select>

            {/* מיון */}
            <Select value={filters.sortBy} onValueChange={handleSortChange}>
              <SelectTrigger className="w-40">
                <SelectValue placeholder="מיון לפי" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="name">שם (א-ת)</SelectItem>
                <SelectItem value="price_asc">מחיר (נמוך לגבוה)</SelectItem>
                <SelectItem value="price_desc">מחיר (גבוה לנמוך)</SelectItem>
                <SelectItem value="return_desc">תשואה (גבוהה לנמוכה)</SelectItem>
                <SelectItem value="return_asc">תשואה (נמוכה לגבוהה)</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}