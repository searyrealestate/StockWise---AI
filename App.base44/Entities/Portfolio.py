{
  "name": "Portfolio",
  "type": "object",
  "properties": {
    "stock_symbol": {
      "type": "string",
      "description": "סימבול המניה"
    },
    "stock_name": {
      "type": "string",
      "description": "שם המניה"
    },
    "quantity": {
      "type": "number",
      "description": "כמות מניות"
    },
    "purchase_price": {
      "type": "number",
      "description": "מחיר קנייה"
    },
    "purchase_date": {
      "type": "string",
      "format": "date",
      "description": "תאריך קנייה"
    },
    "sale_price": {
      "type": "number",
      "description": "מחיר מכירה (אם נמכרה)"
    },
    "sale_date": {
      "type": "string",
      "format": "date",
      "description": "תאריך מכירה (אם נמכרה)"
    },
    "current_price": {
      "type": "number",
      "description": "מחיר נוכחי"
    },
    "status": {
      "type": "string",
      "enum": [
        "פעיל",
        "נמכר"
      ],
      "default": "פעיל",
      "description": "סטטוס הפוזיציה"
    },
    "notifications_muted": {
      "type": "boolean",
      "default": false,
      "description": "האם התראות עבור מניה זו מושתקות"
    }
  },
  "required": [
    "stock_symbol",
    "stock_name",
    "quantity",
    "purchase_price",
    "purchase_date"
  ]
}