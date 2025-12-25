{
  "name": "Stock",
  "type": "object",
  "properties": {
    "symbol": {
      "type": "string",
      "description": "סימבול המניה (למשל: AAPL)"
    },
    "name": {
      "type": "string",
      "description": "שם החברה"
    },
    "current_price": {
      "type": "number",
      "description": "מחיר נוכחי"
    },
    "expected_return": {
      "type": "number",
      "description": "תשואה צפויה באחוזים"
    },
    "risk_level": {
      "type": "string",
      "enum": [
        "סולידי",
        "בינוני",
        "סיכון גבוה"
      ],
      "description": "רמת סיכון"
    },
    "recommendation": {
      "type": "string",
      "enum": [
        "קנה",
        "מכור",
        "החזק"
      ],
      "description": "המלצה"
    },
    "target_price": {
      "type": "number",
      "description": "מחיר יעד"
    },
    "technical_analysis": {
      "type": "string",
      "description": "ניתוח טכני מילולי"
    },
    "fundamental_analysis": {
      "type": "string",
      "description": "ניתוח פונדמנטלי מילולי"
    },
    "company_news": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "title": {
            "type": "string"
          },
          "url": {
            "type": "string"
          },
          "source": {
            "type": "string"
          }
        }
      },
      "description": "חדשות אחרונות על החברה"
    },
    "buy_reasons": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "סיבות לקנייה"
    },
    "sell_reasons": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "סיבות למכירה"
    },
    "sector": {
      "type": "string",
      "description": "מגזר כלכלי"
    },
    "market_cap": {
      "type": "number",
      "description": "שווי שוק"
    },
    "pe_ratio": {
      "type": "number",
      "description": "יחס מחיר לרווח"
    },
    "volume": {
      "type": "number",
      "description": "נפח מסחר יומי"
    },
    "change_percent": {
      "type": "number",
      "description": "שינוי אחוזי יומי"
    }
  },
  "required": [
    "symbol",
    "name",
    "current_price",
    "risk_level",
    "recommendation"
  ]
}