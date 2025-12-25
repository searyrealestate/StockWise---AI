
import React, { useState, useEffect } from "react";
import { Portfolio, Stock } from "@/entities/all";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { TrendingUp, TrendingDown, Plus, Edit, Trash2, DollarSign, Calendar, Briefcase } from "lucide-react";
import { format } from "date-fns";
import { he } from "date-fns/locale";

export default function PortfolioPage() {
  const [portfolioItems, setPortfolioItems] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [showAddDialog, setShowAddDialog] = useState(false);
  const [editingItem, setEditingItem] = useState(null);
  const [formData, setFormData] = useState({
    stock_symbol: '',
    stock_name: '',
    quantity: '',
    purchase_price: '',
    purchase_date: '',
    sale_price: '',
    sale_date: '',
    status: 'פעיל'
  });

  useEffect(() => {
    loadPortfolio();
    // בדיקה אם יש מניה נבחרת מהדף הראשי
    checkForSelectedStock();
  }, []);

  const checkForSelectedStock = () => {
    const urlParams = new URLSearchParams(window.location.search);
    const stockSymbol = urlParams.get('stock');
    if (stockSymbol) {
      setFormData(prev => ({
        ...prev,
        stock_symbol: stockSymbol
      }));
      setShowAddDialog(true);
    }
  };

  const loadPortfolio = async () => {
    setIsLoading(true);
    try {
      const items = await Portfolio.list('-created_date');
      // עדכון מחירים נוכחיים
      await updateCurrentPrices(items);
      setPortfolioItems(items);
    } catch (error) {
      console.error("Error loading portfolio:", error);
    }
    setIsLoading(false);
  };

  const updateCurrentPrices = async (items) => {
    try {
      for (const item of items) {
        if (item.status === 'פעיל') {
          const stocks = await Stock.filter({ symbol: item.stock_symbol });
          if (stocks.length > 0) {
            const currentPrice = stocks[0].current_price;
            await Portfolio.update(item.id, { current_price: currentPrice });
            item.current_price = currentPrice;
          }
        }
      }
    } catch (error) {
      console.error("Error updating prices:", error);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const portfolioData = {
        ...formData,
        quantity: parseFloat(formData.quantity),
        purchase_price: parseFloat(formData.purchase_price),
        sale_price: formData.sale_price ? parseFloat(formData.sale_price) : null,
      };

      if (editingItem) {
        await Portfolio.update(editingItem.id, portfolioData);
      } else {
        await Portfolio.create(portfolioData);
      }

      setShowAddDialog(false);
      setEditingItem(null);
      setFormData({
        stock_symbol: '',
        stock_name: '',
        quantity: '',
        purchase_price: '',
        purchase_date: '',
        sale_price: '',
        sale_date: '',
        status: 'פעיל'
      });
      loadPortfolio();
    } catch (error) {
      console.error("Error saving portfolio item:", error);
    }
  };

  const handleEdit = (item) => {
    setEditingItem(item);
    setFormData({
      stock_symbol: item.stock_symbol,
      stock_name: item.stock_name,
      quantity: item.quantity.toString(),
      purchase_price: item.purchase_price.toString(),
      purchase_date: item.purchase_date,
      sale_price: item.sale_price?.toString() || '',
      sale_date: item.sale_date || '',
      status: item.status
    });
    setShowAddDialog(true);
  };

  const handleDelete = async (id) => {
    if (window.confirm('האם אתה בטוח שברצונך למחוק פוזיציה זו?')) {
      try {
        await Portfolio.delete(id);
        loadPortfolio();
      } catch (error) {
        console.error("Error deleting portfolio item:", error);
      }
    }
  };

  const calculateProfitLoss = (item) => {
    const sellPrice = item.status === 'נמכר' ? item.sale_price : item.current_price;
    if (!sellPrice) return { amount: 0, percentage: 0 };
    
    const profit = (sellPrice - item.purchase_price) * item.quantity;
    const percentage = ((sellPrice - item.purchase_price) / item.purchase_price) * 100;
    
    return { amount: profit, percentage };
  };

  const getTotalValue = () => {
    return portfolioItems.reduce((total, item) => {
      if (item.status === 'פעיל') {
        return total + (item.current_price || item.purchase_price) * item.quantity;
      }
      return total + (item.sale_price * item.quantity);
    }, 0);
  };

  const getTotalProfitLoss = () => {
    return portfolioItems.reduce((total, item) => {
      const { amount } = calculateProfitLoss(item);
      return total + amount;
    }, 0);
  };

  const getTotalInvestment = () => {
    return portfolioItems.reduce((total, item) => {
      return total + (item.purchase_price * item.quantity);
    }, 0);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 p-6">
      <div className="max-w-6xl mx-auto">
        {/* כותרת */}
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-8 gap-4">
          <div>
            <h1 className="text-3xl md:text-4xl font-bold text-slate-800 mb-2">
              תיק ההשקעות שלי
            </h1>
            <p className="text-slate-600 text-lg">
              מעקב אחר ההשקעות שלך ומעקב רווחים והפסדים
            </p>
          </div>
          
          <Button
            onClick={() => setShowAddDialog(true)}
            className="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white font-medium"
          >
            <Plus className="w-4 h-4 mr-2" />
            הוסף פוזיציה
          </Button>
        </div>

        {/* סיכום כללי */}
        <div className="grid md:grid-cols-4 gap-6 mb-8">
          <Card className="bg-white/90 backdrop-blur-sm border-slate-200/60">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-slate-600">שווי תיק נוכחי</p>
                  <p className="text-2xl font-bold text-slate-800">${getTotalValue().toFixed(2)}</p>
                </div>
                <DollarSign className="w-8 h-8 text-blue-600" />
              </div>
            </CardContent>
          </Card>

          <Card className="bg-white/90 backdrop-blur-sm border-slate-200/60">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-slate-600">השקעה כוללת</p>
                  <p className="text-2xl font-bold text-slate-800">${getTotalInvestment().toFixed(2)}</p>
                </div>
                <Calendar className="w-8 h-8 text-purple-600" />
              </div>
            </CardContent>
          </Card>

          <Card className="bg-white/90 backdrop-blur-sm border-slate-200/60">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-slate-600">רווח/הפסד כולל</p>
                  <p className={`text-2xl font-bold ${getTotalProfitLoss() >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {getTotalProfitLoss() >= 0 ? '+' : ''}${getTotalProfitLoss().toFixed(2)}
                  </p>
                </div>
                {getTotalProfitLoss() >= 0 ? 
                  <TrendingUp className="w-8 h-8 text-green-600" /> :
                  <TrendingDown className="w-8 h-8 text-red-600" />
                }
              </div>
            </CardContent>
          </Card>

          <Card className="bg-white/90 backdrop-blur-sm border-slate-200/60">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-slate-600">מספר פוזיציות</p>
                  <p className="text-2xl font-bold text-slate-800">{portfolioItems.length}</p>
                </div>
                <div className="text-right">
                  <p className="text-xs text-slate-500">פעילות: {portfolioItems.filter(i => i.status === 'פעיל').length}</p>
                  <p className="text-xs text-slate-500">סגורות: {portfolioItems.filter(i => i.status === 'נמכר').length}</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* רשימת פוזיציות */}
        <Card className="bg-white/90 backdrop-blur-sm border-slate-200/60">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Briefcase className="w-5 h-5" />
              פוזיציות בתיק
            </CardTitle>
          </CardHeader>
          <CardContent>
            {portfolioItems.length === 0 ? (
              <div className="text-center py-12">
                <p className="text-slate-500 text-lg mb-4">התיק שלך ריק</p>
                <Button
                  onClick={() => setShowAddDialog(true)}
                  className="bg-blue-600 hover:bg-blue-700"
                >
                  <Plus className="w-4 h-4 mr-2" />
                  הוסף פוזיציה ראשונה
                </Button>
              </div>
            ) : (
              <div className="space-y-4">
                {portfolioItems.map((item) => {
                  const { amount: profitAmount, percentage: profitPercentage } = calculateProfitLoss(item);
                  const isProfit = profitAmount >= 0;
                  
                  return (
                    <div key={item.id} className="border border-slate-200 rounded-lg p-4 hover:shadow-md transition-shadow">
                      <div className="flex justify-between items-start mb-3">
                        <div>
                          <h3 className="font-bold text-lg text-slate-800">{item.stock_symbol}</h3>
                          <p className="text-slate-600">{item.stock_name}</p>
                        </div>
                        <div className="flex items-center gap-2">
                          <Badge 
                            className={item.status === 'פעיל' ? 
                              'bg-green-100 text-green-800 border-green-200' : 
                              'bg-gray-100 text-gray-800 border-gray-200'
                            }
                          >
                            {item.status}
                          </Badge>
                          <Button
                            variant="ghost"
                            size="icon"
                            onClick={() => handleEdit(item)}
                          >
                            <Edit className="w-4 h-4" />
                          </Button>
                          <Button
                            variant="ghost"
                            size="icon"
                            onClick={() => handleDelete(item.id)}
                            className="text-red-500 hover:text-red-700"
                          >
                            <Trash2 className="w-4 h-4" />
                          </Button>
                        </div>
                      </div>
                      
                      <div className="grid md:grid-cols-6 gap-4 text-sm">
                        <div>
                          <p className="text-slate-500">כמות</p>
                          <p className="font-semibold">{item.quantity}</p>
                        </div>
                        <div>
                          <p className="text-slate-500">מחיר קנייה</p>
                          <p className="font-semibold">${item.purchase_price}</p>
                        </div>
                        <div>
                          <p className="text-slate-500">מחיר נוכחי</p>
                          <p className="font-semibold">
                            ${item.status === 'נמכר' ? item.sale_price : (item.current_price || item.purchase_price)}
                          </p>
                        </div>
                        <div>
                          <p className="text-slate-500">שווי נוכחי</p>
                          <p className="font-semibold">
                            ${((item.status === 'נמכר' ? item.sale_price : (item.current_price || item.purchase_price)) * item.quantity).toFixed(2)}
                          </p>
                        </div>
                        <div>
                          <p className="text-slate-500">רווח/הפסד</p>
                          <p className={`font-semibold ${isProfit ? 'text-green-600' : 'text-red-600'}`}>
                            {isProfit ? '+' : ''}${profitAmount.toFixed(2)}
                          </p>
                        </div>
                        <div>
                          <p className="text-slate-500">אחוז שינוי</p>
                          <p className={`font-semibold ${isProfit ? 'text-green-600' : 'text-red-600'}`}>
                            {isProfit ? '+' : ''}{profitPercentage.toFixed(2)}%
                          </p>
                        </div>
                      </div>
                      
                      <Separator className="my-3" />
                      
                      <div className="flex justify-between items-center text-xs text-slate-500">
                        <span>נקנה: {format(new Date(item.purchase_date), 'dd/MM/yyyy', { locale: he })}</span>
                        {item.sale_date && (
                          <span>נמכר: {format(new Date(item.sale_date), 'dd/MM/yyyy', { locale: he })}</span>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </CardContent>
        </Card>

        {/* דיאלוג הוספה/עריכה */}
        <Dialog open={showAddDialog} onOpenChange={setShowAddDialog}>
          <DialogContent className="max-w-md">
            <DialogHeader>
              <DialogTitle>
                {editingItem ? 'עריכת פוזיציה' : 'הוספת פוזיציה חדשה'}
              </DialogTitle>
            </DialogHeader>
            
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label htmlFor="stock_symbol">סימבול מניה</Label>
                  <Input
                    id="stock_symbol"
                    value={formData.stock_symbol}
                    onChange={(e) => setFormData({...formData, stock_symbol: e.target.value.toUpperCase()})}
                    placeholder="AAPL"
                    required
                  />
                </div>
                <div>
                  <Label htmlFor="stock_name">שם המניה</Label>
                  <Input
                    id="stock_name"
                    value={formData.stock_name}
                    onChange={(e) => setFormData({...formData, stock_name: e.target.value})}
                    placeholder="Apple Inc."
                    required
                  />
                </div>
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label htmlFor="quantity">כמות</Label>
                  <Input
                    id="quantity"
                    type="number"
                    step="0.01"
                    value={formData.quantity}
                    onChange={(e) => setFormData({...formData, quantity: e.target.value})}
                    placeholder="10"
                    required
                  />
                </div>
                <div>
                  <Label htmlFor="purchase_price">מחיר קנייה</Label>
                  <Input
                    id="purchase_price"
                    type="number"
                    step="0.01"
                    value={formData.purchase_price}
                    onChange={(e) => setFormData({...formData, purchase_price: e.target.value})}
                    placeholder="150.50"
                    required
                  />
                </div>
              </div>
              
              <div>
                <Label htmlFor="purchase_date">תאריך קנייה</Label>
                <Input
                  id="purchase_date"
                  type="date"
                  value={formData.purchase_date}
                  onChange={(e) => setFormData({...formData, purchase_date: e.target.value})}
                  required
                />
              </div>

              {formData.status === 'נמכר' && (
                <>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <Label htmlFor="sale_price">מחיר מכירה</Label>
                      <Input
                        id="sale_price"
                        type="number"
                        step="0.01"
                        value={formData.sale_price}
                        onChange={(e) => setFormData({...formData, sale_price: e.target.value})}
                        placeholder="160.00"
                      />
                    </div>
                    <div>
                      <Label htmlFor="sale_date">תאריך מכירה</Label>
                      <Input
                        id="sale_date"
                        type="date"
                        value={formData.sale_date}
                        onChange={(e) => setFormData({...formData, sale_date: e.target.value})}
                      />
                    </div>
                  </div>
                </>
              )}

              <div className="flex gap-2 pt-4">
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => setFormData({...formData, status: 'פעיל'})}
                  className={formData.status === 'פעיל' ? 'bg-green-50 border-green-300' : ''}
                >
                  פעיל
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => setFormData({...formData, status: 'נמכר'})}
                  className={formData.status === 'נמכר' ? 'bg-gray-50 border-gray-300' : ''}
                >
                  נמכר
                </Button>
              </div>

              <div className="flex justify-end gap-3 pt-4">
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => {
                    setShowAddDialog(false);
                    setEditingItem(null);
                  }}
                >
                  ביטול
                </Button>
                <Button
                  type="submit"
                  className="bg-blue-600 hover:bg-blue-700"
                >
                  {editingItem ? 'עדכן' : 'הוסף'}
                </Button>
              </div>
            </form>
          </DialogContent>
        </Dialog>
      </div>
    </div>
  );
}
