"""
🇮🇱 ISRAELI TRADING COSTS CALCULATOR
═══════════════════════════════════════════════════════════════

Add this to your testing script to calculate real Israeli trading costs
Based on Inter broker fees and Israeli tax law (25% capital gains tax)
"""

class IsraeliTradingCosts:
    def __init__(self):
        # Inter broker fees (from the PDF)
        self.commission_rate = 0.003  # 0.3% commission
        self.min_commission_usd = 10  # Minimum $10 USD
        self.min_commission_ils = 35  # Minimum ₪35 ILS

        # Israeli taxes
        self.capital_gains_tax = 0.25  # 25% tax on profits

        # Exchange rates (approximate - should be updated with real rates)
        self.usd_to_ils = 3.7  # Updated exchange rate

        # Additional fees
        self.foreign_stock_fee = 0.0001  # 0.01% for foreign stocks
        self.settlement_fee_usd = 1  # $1 per transaction

    def calculate_commission(self, trade_value_usd, currency='USD'):
        """Calculate Inter broker commission"""

        # Base commission (0.3% of trade value)
        commission = trade_value_usd * self.commission_rate

        # Apply minimum commission
        if currency == 'USD':
            commission = max(commission, self.min_commission_usd)
        else:  # ILS
            min_commission_usd = self.min_commission_ils / self.usd_to_ils
            commission = max(commission, min_commission_usd)

        return commission

    def calculate_additional_fees(self, trade_value_usd):
        """Calculate additional Inter fees"""

        # Foreign stock fee (0.01%)
        foreign_fee = trade_value_usd * self.foreign_stock_fee

        # Settlement fee
        settlement_fee = self.settlement_fee_usd

        return foreign_fee + settlement_fee

    def calculate_total_trading_costs(self, entry_price, exit_price, shares, holding_days=7):
        """
        Calculate complete Israeli trading costs including taxes

        Args:
            entry_price: Buy price per share (USD)
            exit_price: Sell price per share (USD)
            shares: Number of shares
            holding_days: Days held (affects tax calculation)

        Returns:
            dict with detailed cost breakdown
        """

        # Trade values
        entry_value = entry_price * shares
        exit_value = exit_price * shares
        gross_profit = exit_value - entry_value

        # Entry costs (buying)
        entry_commission = self.calculate_commission(entry_value)
        entry_additional = self.calculate_additional_fees(entry_value)
        entry_total_fees = entry_commission + entry_additional

        # Exit costs (selling)
        exit_commission = self.calculate_commission(exit_value)
        exit_additional = self.calculate_additional_fees(exit_value)
        exit_total_fees = exit_commission + exit_additional

        # Total broker fees
        total_broker_fees = entry_total_fees + exit_total_fees

        # Net profit before tax
        net_profit_before_tax = gross_profit - total_broker_fees

        # Israeli capital gains tax (25% on profits only)
        if net_profit_before_tax > 0:
            capital_gains_tax = net_profit_before_tax * self.capital_gains_tax
        else:
            capital_gains_tax = 0  # No tax on losses

        # Final net profit
        final_net_profit = net_profit_before_tax - capital_gains_tax

        # Calculate percentages
        gross_return_pct = (gross_profit / entry_value) * 100 if entry_value > 0 else 0
        net_return_pct = (final_net_profit / entry_value) * 100 if entry_value > 0 else 0

        # Cost breakdown
        return {
            'entry_value_usd': round(entry_value, 2),
            'exit_value_usd': round(exit_value, 2),
            'shares': shares,
            'gross_profit_usd': round(gross_profit, 2),
            'gross_return_pct': round(gross_return_pct, 2),

            # Broker fees breakdown
            'entry_commission': round(entry_commission, 2),
            'entry_additional_fees': round(entry_additional, 2),
            'exit_commission': round(exit_commission, 2),
            'exit_additional_fees': round(exit_additional, 2),
            'total_broker_fees': round(total_broker_fees, 2),
            'broker_fees_pct': round((total_broker_fees / entry_value) * 100, 2),

            # Tax calculation
            'taxable_profit': round(max(net_profit_before_tax, 0), 2),
            'capital_gains_tax': round(capital_gains_tax, 2),
            'tax_rate_applied': 25.0 if net_profit_before_tax > 0 else 0.0,

            # Final results
            'net_profit_after_all_costs': round(final_net_profit, 2),
            'net_return_pct': round(net_return_pct, 2),
            'total_cost_impact_pct': round(gross_return_pct - net_return_pct, 2),

            # Cost efficiency metrics
            'break_even_price': round(entry_price * (1 + (total_broker_fees + capital_gains_tax) / entry_value), 2),
            'min_profit_for_positive_return': round(total_broker_fees / (1 - self.capital_gains_tax), 2)
        }

# Integration with your existing testing script
def integrate_israeli_costs_to_testing():
    """
    ADD THESE FUNCTIONS TO YOUR EXISTING test_algorithm.py FILE:
    """

    # Add this to your test_single_prediction method
    def enhanced_test_single_prediction_with_costs(self, symbol, test_date, holding_days=7, investment_amount_usd=10000):
        """Enhanced version with Israeli trading costs"""

        # Your existing test logic here...
        # [Keep all your existing code until you get actual_profit_pct]

        # NEW: Calculate with Israeli trading costs
        cost_calculator = IsraeliTradingCosts()

        if actual_analysis_price and actual_future_price:
            # Calculate shares based on investment amount
            shares = int(investment_amount_usd / actual_analysis_price)
            actual_investment = shares * actual_analysis_price

            # Calculate costs with Israeli taxes and fees
            cost_breakdown = cost_calculator.calculate_total_trading_costs(
                entry_price=actual_analysis_price,
                exit_price=actual_future_price,
                shares=shares,
                holding_days=holding_days
            )

            # Update profit calculations
            gross_profit_pct = cost_breakdown['gross_return_pct']
            net_profit_pct = cost_breakdown['net_return_pct']
            total_costs_pct = cost_breakdown['total_cost_impact_pct']

            # Enhanced test result
            test_record = {
                # ... your existing fields ...

                # NEW: Israeli cost fields
                'investment_amount_usd': actual_investment,
                'shares_traded': shares,
                'gross_profit_pct': gross_profit_pct,
                'net_profit_pct_after_costs': net_profit_pct,
                'broker_fees_usd': cost_breakdown['total_broker_fees'],
                'broker_fees_pct': cost_breakdown['broker_fees_pct'],
                'capital_gains_tax_usd': cost_breakdown['capital_gains_tax'],
                'total_cost_impact_pct': total_costs_pct,
                'break_even_price': cost_breakdown['break_even_price'],

                # Update test result based on net profit
                'test_result_gross': self.evaluate_prediction(action, confidence, predicted_profit, gross_profit_pct,
                                                            buy_price, sell_price, actual_analysis_price, actual_future_price),
                'test_result_net': self.evaluate_prediction(action, confidence, predicted_profit, net_profit_pct,
                                                          buy_price, sell_price, actual_analysis_price, actual_future_price)
            }

            return test_record

def update_existing_testing_script():
    """
    EXACT CHANGES FOR YOUR EXISTING test_algorithm.py:
    """

    changes = '''
    # 1. ADD AT THE TOP OF YOUR FILE (after imports):
    
    from israeli_trading_costs import IsraeliTradingCosts  # Add this import
    
    # 2. UPDATE YOUR test_single_prediction METHOD:
    # Find this section in your existing method:
    
    # Calculate actual profit
    actual_profit_pct = ((future_price / analysis_price) - 1) * 100
    
    # REPLACE WITH:
    
    # Calculate actual profit (gross)
    gross_profit_pct = ((future_price / analysis_price) - 1) * 100
    
    # NEW: Calculate with Israeli costs
    cost_calculator = IsraeliTradingCosts()
    investment_amount = 10000  # $10,000 default investment
    shares = int(investment_amount / analysis_price)
    
    cost_breakdown = cost_calculator.calculate_total_trading_costs(
        entry_price=analysis_price,
        exit_price=future_price, 
        shares=shares,
        holding_days=holding_days
    )
    
    net_profit_pct = cost_breakdown['net_return_pct']
    
    # 3. UPDATE YOUR RESULTS DICTIONARY:
    # Add these new fields to your results.append() call:
    
    results.append({
        'Stock': stock,
        'Date': test_date.strftime('%Y-%m-%d'),
        'Action': action,
        'Confidence': f"{confidence:.1f}%",
        'Buy_Price': f"${buy_price:.2f}" if buy_price else "N/A",
        'Sell_Price': f"${sell_price:.2f}" if sell_price else "N/A", 
        'Predicted_Profit': f"{predicted_profit:.1f}%",
        'Actual_Price_Analysis': f"${analysis_price:.2f}",
        'Actual_Price_Future': f"${future_price:.2f}",
        
        # UPDATED PROFIT CALCULATIONS:
        'Gross_Profit': f"{gross_profit_pct:.1f}%",
        'Net_Profit_After_Costs': f"{net_profit_pct:.1f}%",
        'Broker_Fees': f"${cost_breakdown['total_broker_fees']:.2f}",
        'Taxes_Paid': f"${cost_breakdown['capital_gains_tax']:.2f}",
        'Total_Cost_Impact': f"{cost_breakdown['total_cost_impact_pct']:.1f}%",
        
        # TEST RESULTS (use net profit for realistic assessment):
        'Test_Result_Gross': test_result_gross,
        'Test_Result_Net': determine_test_result(action, net_profit_pct),
        'Investment_Amount': f"${investment_amount:.0f}",
        'Shares': shares
    })
    
    # 4. UPDATE YOUR SUMMARY CALCULATIONS:
    # Change your summary to use net profits:
    
    # Quick summary  
    total_tests = len(df)
    pass_count_net = len(df[df['Test_Result_Net'] == 'PASS'])
    
    print(f"📊 REALISTIC SUMMARY (After Israeli Costs):")
    print(f"   🟢 NET PASS: {pass_count_net} ({pass_count_net/total_tests*100:.1f}%)")
    
    # Show cost impact
    df['Net_Profit_Numeric'] = df['Net_Profit_After_Costs'].str.replace('%', '').astype(float)
    avg_cost_impact = df['Total_Cost_Impact'].str.replace('%', '').astype(float).mean()
    print(f"   💸 Average Cost Impact: {avg_cost_impact:.1f}%")
    '''

    return changes

# Helper function for realistic test evaluation
def determine_test_result(action, net_profit_pct):
    """Determine test result based on net profit after all costs"""
    if action == 'BUY':
        if net_profit_pct > 1.0:  # At least 1% net profit
            return "PASS"
        elif net_profit_pct > 0:
            return "PARTIAL"
        else:
            return "FAIL"
    elif action == 'SELL/AVOID':
        if net_profit_pct <= 0:
            return "PASS"
        else:
            return "FAIL"
    else:  # WAIT
        if -1.0 <= net_profit_pct <= 1.0:
            return "PASS"
        else:
            return "PARTIAL"

# Example usage and testing
def test_israeli_costs():
    """Test the Israeli cost calculator"""

    calc = IsraeliTradingCosts()

    # Example trade: Buy AAPL at $150, sell at $160, 66 shares ($10,000 investment)
    costs = calc.calculate_total_trading_costs(
        entry_price=150.0,
        exit_price=160.0,
        shares=66,
        holding_days=7
    )

    print("🧪 ISRAELI COST CALCULATION TEST:")
    print("=" * 50)
    print(f"📊 Trade: 66 shares AAPL, $150 → $160")
    print(f"💰 Investment: ${costs['entry_value_usd']:,.2f}")
    print(f"📈 Gross Profit: ${costs['gross_profit_usd']:,.2f} ({costs['gross_return_pct']:.1f}%)")
    print(f"💸 Broker Fees: ${costs['total_broker_fees']:.2f} ({costs['broker_fees_pct']:.2f}%)")
    print(f"🏛️ Capital Gains Tax: ${costs['capital_gains_tax']:.2f}")
    print(f"💵 Net Profit: ${costs['net_profit_after_all_costs']:.2f} ({costs['net_return_pct']:.1f}%)")
    print(f"📉 Total Cost Impact: {costs['total_cost_impact_pct']:.1f}%")
    print(f"⚖️ Break-even Price: ${costs['break_even_price']:.2f}")


if __name__ == "__main__":
    test_israeli_costs()