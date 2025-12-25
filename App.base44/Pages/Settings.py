
import React, { useState, useEffect } from "react";
import { User } from "@/entities/User";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Check, Loader2, Sun, Moon } from "lucide-react";

export default function SettingsPage() {
  const [settings, setSettings] = useState({
    analysis_preference: 'משולב',
    investment_amount: 0,
    notifications_enabled: true,
    notification_threshold: 5,
    theme: 'light' // Added: theme setting
  });
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(false);

  useEffect(() => {
    loadUserSettings();
  }, []);

  const loadUserSettings = async () => {
    setIsLoading(true);
    try {
      const user = await User.me();
      setSettings({
        analysis_preference: user.analysis_preference || 'משולב',
        investment_amount: user.investment_amount || 0,
        notifications_enabled: user.notifications_enabled !== false,
        notification_threshold: user.notification_threshold || 5,
        theme: user.theme || 'light' // Added: theme setting from user data
      });
    } catch (error) {
      console.error("Error loading user settings:", error);
    }
    setIsLoading(false);
  };

  const handleInputChange = (field, value) => {
    setSettings(prev => ({ ...prev, [field]: value }));
  };

  const handleSave = async (e) => {
    e.preventDefault();
    setIsSaving(true);
    setSaveSuccess(false);
    try {
      await User.updateMyUserData(settings);
      setSaveSuccess(true);
      setTimeout(() => setSaveSuccess(false), 2000);
    } catch (error) {
      console.error("Error saving settings:", error);
    }
    setIsSaving(false);
  };

  if (isLoading) {
    return (
      <div className="min-h-screen p-6 flex justify-center items-center">
        <Loader2 className="w-8 h-8 animate-spin" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 p-6">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl md:text-4xl font-bold text-slate-800 mb-8">
          הגדרות
        </h1>

        <form onSubmit={handleSave}>
          <div className="space-y-8">
            {/* New Card: Display Preferences */}
            <Card>
              <CardHeader>
                <CardTitle>העדפות תצוגה</CardTitle>
                <CardDescription>התאם את המראה והתחושה של המערכת.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    {settings.theme === 'light' ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
                    <Label htmlFor="theme">מצב תצוגה</Label>
                  </div>
                  <Select
                    value={settings.theme}
                    onValueChange={(value) => handleInputChange('theme', value)}
                  >
                    <SelectTrigger className="w-32">
                      <SelectValue placeholder="בחר מצב" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="light">בהיר</SelectItem>
                      <SelectItem value="dark">כהה</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </CardContent>
            </Card>

            {/* Existing Card: Analysis Preferences (moved after new card) */}
            <Card>
              <CardHeader>
                <CardTitle>העדפות ניתוח</CardTitle>
                <CardDescription>בחר כיצד המערכת תנתח ותציג המלצות מניות.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <Label htmlFor="analysis_preference">סוג ניתוח מועדף</Label>
                  <Select
                    value={settings.analysis_preference}
                    onValueChange={(value) => handleInputChange('analysis_preference', value)}
                  >
                    <SelectTrigger id="analysis_preference">
                      <SelectValue placeholder="בחר סוג ניתוח" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="טכני">ניתוח טכני</SelectItem>
                      <SelectItem value="פונדמנטלי">ניתוח פונדמנטלי</SelectItem>
                      <SelectItem value="משולב">ניתוח משולב</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </CardContent>
            </Card>

            {/* Existing Card: Investment Settings */}
            <Card>
              <CardHeader>
                <CardTitle>הגדרות השקעה</CardTitle>
                <CardDescription>מידע זה יעזור למערכת להתאים לך המלצות.</CardDescription>
              </CardHeader>
              <CardContent>
                <div>
                  <Label htmlFor="investment_amount">סכום זמין להשקעה ($)</Label>
                  <Input
                    id="investment_amount"
                    type="number"
                    value={settings.investment_amount}
                    onChange={(e) => handleInputChange('investment_amount', parseFloat(e.target.value))}
                    placeholder="לדוגמה: 10000"
                  />
                </div>
              </CardContent>
            </Card>

            {/* Existing Card: Notifications */}
            <Card>
              <CardHeader>
                <CardTitle>התראות</CardTitle>
                <CardDescription>נהל כיצד ומתי תקבל התראות על המניות שלך.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="flex items-center justify-between">
                  <Label htmlFor="notifications_enabled">הפעל התראות</Label>
                  <Switch
                    id="notifications_enabled"
                    checked={settings.notifications_enabled}
                    onCheckedChange={(value) => handleInputChange('notifications_enabled', value)}
                  />
                </div>
                {settings.notifications_enabled && (
                  <div>
                    <Label htmlFor="notification_threshold">
                      סף התראה מוקדמת (% סטייה ממחיר יעד)
                    </Label>
                    <Input
                      id="notification_threshold"
                      type="number"
                      value={settings.notification_threshold}
                      onChange={(e) => handleInputChange('notification_threshold', parseInt(e.target.value))}
                      placeholder="לדוגמה: 5"
                    />
                    <p className="text-sm text-slate-500 mt-2">
                      תקבל התראות כאשר מחיר המניה נמצא במרחק של אחוז זה ממחיר היעד (קנייה או מכירה).
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>

            <div className="flex justify-end">
              <Button type="submit" disabled={isSaving}>
                {isSaving ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    שומר...
                  </>
                ) : saveSuccess ? (
                  <>
                    <Check className="mr-2 h-4 w-4" />
                    נשמר בהצלחה!
                  </>
                ) : (
                  'שמור שינויים'
                )}
              </Button>
            </div>
          </div>
        </form>
      </div>
    </div>
  );
}
