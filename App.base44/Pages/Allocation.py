
import React, { useState, useEffect } from "react";
import { User, Stock } from "@/entities/all";
import { InvokeLLM } from "@/integrations/Core";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from "recharts";
import { Loader2, Zap } from "lucide-react";
import { createPageUrl } from "@/utils";

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#82ca9d', '#ffc658'];

export default function AllocationPage() {
  const [investmentAmount, setInvestmentAmount] = useState(0);
  const [allocation, setAllocation] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isGenerating, setIsGenerating] = useState(false);

  useEffect(() => {
    loadUserData();
  }, []);

  const loadUserData = async () => {
    setIsLoading(true);
    try {
      const user = await User.me();
      setInvestmentAmount(user.investment_amount || 0);
    } catch (error) {
      console.error("Error loading user data:", error);
    }
    setIsLoading(false);
  };

  const generateAllocation = async () => {
    setIsGenerating(true);
    setAllocation(null);
    try {
      const stocks = await Stock.list('-expected_return', 20); // Get top 20 recommendations
      const stockInfo = stocks.map(s => `סמל: ${s.symbol}, שם: ${s.name}, רמת סיכון: ${s.risk_level}, תשואה צפויה: ${s.expected_return}%`).join('\n');
      
      const prompt = `
      בהתבסס על סכום השקעה של $${investmentAmount} ורשימת המניות המומלצות הבאות:
      ${stockInfo}
      
      צור תוכנית חלוקת השקעה (פורטפוליו) מגוונת. 
      הקצה את כל סכום ההשקעה על פני 5-7 מניות מתוך הרשימה.
      ודא שיש פיזור בין רמות סיכון שונות (סולידי, בינוני, גבוה).
      
      לכל מניה בתוכנית, ציין:
      - symbol: סימבול המניה
      - name: שם המניה
      - allocated_amount: הסכום ב-$ להשקעה במניה
      - percentage: אחוז ההקצאה מסך התיק
      
      הקפד שהסכום הכולל של 'allocated_amount' יהיה שווה ל-$${investmentAmount}.
      `;

      const result = await InvokeLLM({
        prompt,
        add_context_from_internet: false,
        response_json_schema: {
          type: "object",
          properties: {
            portfolio: {
              type: "array",
              items: {
                type: "object",
                properties: {
                  symbol: { type: "string" },
                  name: { type: "string" },
                  allocated_amount: { type: "number" },
                  percentage: { type: "number" }
                }
              }
            },
            summary: { type: "string" }
          }
        }
      });
      
      setAllocation(result);
      
    } catch (error) {
      console.error("Error generating allocation:", error);
    }
    setIsGenerating(false);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 p-6">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-3xl md:text-4xl font-bold text-slate-800 mb-2">
          חלוקת השקעה חכמה
        </h1>
        <p className="text-slate-600 text-lg mb-8">
          קבל תוכנית השקעה מותאמת אישית לסכום ההשקעה שלך.
        </p>

        <Card className="mb-8">
          <CardContent className="p-6 flex flex-col md:flex-row justify-between items-center gap-4">
            <div>
              <p className="text-slate-600">סכום ההשקעה שהוגדר</p>
              <p className="text-4xl font-bold text-blue-600">${investmentAmount.toLocaleString()}</p>
              <p className="text-sm text-slate-500 mt-2">
                ניתן לשנות סכום זה בדף <a href={createPageUrl("Settings")} className="text-blue-500 hover:underline">ההגדרות</a>.
              </p>
            </div>
            <Button 
              onClick={generateAllocation} 
              disabled={isGenerating || investmentAmount === 0}
              size="lg"
              className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white"
            >
              {isGenerating ? (
                <>
                  <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                  מייצר תוכנית...
                </>
              ) : (
                <>
                  <Zap className="mr-2 h-5 w-5" />
                  צור תוכנית השקעה
                </>
              )}
            </Button>
          </CardContent>
        </Card>

        {isGenerating && (
          <div className="text-center py-16">
            <Loader2 className="w-12 h-12 text-blue-600 animate-spin mx-auto" />
            <p className="mt-4 text-slate-600 text-lg">המערכת בונה עבורך את תוכנית ההשקעה האופטימלית...</p>
          </div>
        )}

        {allocation && (
          <div className="grid lg:grid-cols-2 gap-8">
            <Card>
              <CardHeader>
                <CardTitle>תוכנית ההשקעה המוצעת</CardTitle>
                <CardDescription>{allocation.summary}</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {allocation.portfolio.map((item, index) => (
                    <div key={item.symbol} className="flex items-center justify-between p-3 rounded-lg border">
                      <div className="flex items-center gap-3">
                        <div style={{backgroundColor: COLORS[index % COLORS.length]}} className="w-3 h-8 rounded"></div>
                        <div>
                          <p className="font-bold text-slate-800">{item.symbol}</p>
                          <p className="text-sm text-slate-600">{item.name}</p>
                        </div>
                      </div>
                      <div>
                        <p className="font-bold text-slate-800 text-right">${item.allocated_amount.toLocaleString()}</p>
                        <p className="text-sm text-slate-600 text-right">{item.percentage.toFixed(1)}%</p>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle>פיזור התיק</CardTitle>
              </CardHeader>
              <CardContent className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={allocation.portfolio}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      outerRadius={120}
                      fill="#8884d8"
                      dataKey="allocated_amount"
                      nameKey="symbol"
                      label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                    >
                      {allocation.portfolio.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip formatter={(value) => `$${value.toLocaleString()}`} />
                  </PieChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    </div>
  );
}
