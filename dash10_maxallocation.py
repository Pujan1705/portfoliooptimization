import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf
import plotly.graph_objects as go
import sqlite3
import ast

# Function to fetch data from Yahoo Finance
def FetchingDataFromYahoo(ticker_names, start_date, end_date):
    data = yf.download(ticker_names, start=start_date, end=end_date)
    df = data['Close'].copy()
    df.reset_index(inplace=True)
    return df

# Function to run portfolio optimization
def run_portfolio_optimization(ticker_names, start_date, end_date):
    # Fetch historical stock data using Yahoo Finance API
    data = FetchingDataFromYahoo(ticker_names, start_date, end_date)
    data.set_index('Date', inplace=True)
    stocks_data = data
    
    # Calculate daily and annual returns
    ret = stocks_data.pct_change()
    Return_daily = ret.mean()
    Return_Annual = (((1 + Return_daily) ** 252) - 1) * 100
    
    # Calculate daily and annual standard deviations
    daily_std = ret.std()
    annual_std = daily_std * np.sqrt(252) * 100
    
    # Create a DataFrame to store annual returns and standard deviations
    stocks_result = pd.DataFrame(Return_Annual)
    stocks_result.rename({0: 'Annual_Return'}, axis=1, inplace=True)
    stocks_result['Annual_sd'] = round(annual_std, 4)
    
    # Calculate stock log returns and covariance matrix
    stocks_returns = stocks_data / stocks_data.shift(1)
    stocks_returns = stocks_returns[1:]
    stocks_logReturns = np.log(stocks_returns)
    meanLogReturn = stocks_logReturns.mean()
    Sigma = stocks_logReturns.cov()
    
    # Number of portfolios for simulation
    noOfPortfolios = 1000
    
    # Number of stocks in the portfolio
    num_stocks = len(stocks_returns.columns)
    weight = np.zeros((noOfPortfolios, num_stocks))
    
    # Initialize arrays to store portfolio metrics
    expectedReturn_annual = np.zeros(noOfPortfolios)
    expectedReturnLog = np.zeros(noOfPortfolios)
    expectedVolatility_annual = np.zeros(noOfPortfolios)
    expectedVolatilityLog = np.zeros(noOfPortfolios)
    sharpeRatio_annual_value = np.zeros(noOfPortfolios)
    sharpeRatioLog = np.zeros(noOfPortfolios)
    
    # Extract annual return and covariance values
    A_Return = stocks_result.Annual_Return
    Sigma = stocks_returns.cov()
    
    # Calculate covariance matrix for log returns
    Sigma_log = stocks_logReturns.cov()
    
    # Create an empty DataFrame to store simulation details
    sim_details = pd.DataFrame(columns=['expectedReturn', 'expectedRisk', 'sharpeRatio_annual_value',
                                        'sharpeRatioLog', 'weight'])
    
    # Iterate over each portfolio
    for k in range(noOfPortfolios):
        # Generate random weights for stocks in the portfolio
        w = np.array(np.random.random(num_stocks))
        w = w / np.sum(w)
        weight[k, :] = w
        
        # Calculate portfolio metrics based on weights
        expectedReturn_annual[k] = np.sum(A_Return * w)
        expectedReturnLog[k] = np.sum(meanLogReturn * w)
        expectedVolatility_annual[k] = np.sqrt(np.dot(w.T, np.dot(Sigma, w))) * np.sqrt(252) * 100
        expectedVolatilityLog[k] = np.sqrt(np.dot(w.T, np.dot(Sigma_log, w)))
        sharpeRatio_annual_value[k] = expectedReturn_annual[k] / expectedVolatility_annual[k]
        sharpeRatioLog[k] = expectedReturnLog[k] / expectedVolatilityLog[k]
        
        # Store the simulation details in the DataFrame
        sim_details.loc[k] = [expectedReturn_annual[k], expectedVolatility_annual[k], sharpeRatio_annual_value[k],
                              sharpeRatioLog[k], w]
    
    # Sort simulation details by Sharpe ratio (log)
    sim_details.sort_values('sharpeRatioLog', axis=0, ascending=False, inplace=True)
    sim_details = round(sim_details, 4)
    
    return sim_details

# Function to get the best portfolio based on risk
def get_best_portfolio_by_risk(ticker_names, start_date, end_date, risk_level):
    # Run portfolio optimization to get simulation details
    sim_details = run_portfolio_optimization(ticker_names, start_date, end_date)

    # Print the maximum and minimum risk
    max_risk = sim_details['expectedRisk'].max()
    min_risk = sim_details['expectedRisk'].min()
    print("Maximum Risk:", max_risk)
    print("Minimum Risk:", min_risk)

    # Filter portfolios based on the desired risk level
    filtered_portfolios = sim_details[sim_details['expectedRisk'] <= risk_level]

    # Check if any portfolios meet the risk level criteria
    if filtered_portfolios.empty:
        print("No portfolios found within the specified risk level.")
    else:
        # Find the best portfolio within the filtered set
        best_portfolio_index = filtered_portfolios['sharpeRatioLog'].idxmax()
        best_portfolio_weights = filtered_portfolios.loc[best_portfolio_index, 'weight']
        best_portfolio_return = filtered_portfolios.loc[best_portfolio_index, 'expectedReturn']
        best_portfolio_sharpe = filtered_portfolios.loc[best_portfolio_index, 'sharpeRatioLog']

        # Print weights for the best portfolio
        print("Weights for the Best Portfolio:")
        for stock, weight in zip(ticker_names, best_portfolio_weights):
            weight_percentage = weight * 100
            print(f"{stock}: {weight_percentage:.2f}%")

        # Print the best portfolio's return and Sharpe ratio
        print("Best Portfolio Return (Annual):", best_portfolio_return)
        print("Best Portfolio Sharpe Ratio (Annual):", best_portfolio_sharpe)

        # Print the portfolio return for 3 years
        portfolio_return_3_years = (best_portfolio_return / 100) * 3
        print("Portfolio Return for 3 Years:", portfolio_return_3_years*100, "%")


# Function to plot the efficient frontier
def plot_efficient_frontier(sim_details):
    # Plot the efficient frontier using Plotly
    fig = go.Figure(data=go.Scatter(
        x=sim_details['expectedRisk'],
        y=sim_details['expectedReturn'],
        mode='markers',
        marker=dict(
            size=8,
            color=sim_details['sharpeRatioLog'],
            colorscale='viridis',
            showscale=True
        ),
        hovertext=sim_details.apply(lambda row: f"Sharpe Ratio (Log): {row['sharpeRatioLog']:.4f}", axis=1),
        hovertemplate='Expected Risk: %{x:.4f}<br>Expected Return: %{y:.4f}<br>%{hovertext}',
    ))

    # Customize the plot layout
    fig.update_layout(
        title='Efficient Frontier',
        xaxis=dict(title='Expected Risk (Annual Volatility)'),
        yaxis=dict(title='Expected Return (Annual Return)'),
    )

    # Show the plot
    fig.show()

# Set the random seed for consistent results
np.random.seed(42)

# Set the start and end dates for historical data
start_date = dt.datetime(2021, 6, 29)
end_date = dt.datetime(2023, 7, 13)

# Prompt the user to enter ticker names
ticker_input = input("Enter the ticker names (comma-separated): ")
ticker_names = [ticker.strip() for ticker in ticker_input.split(",")]

# Ensure at least 10 stock tickers are provided
if len(ticker_names) < 10:
    print("Please provide at least 10 stock tickers.")
    exit()

# Prompt the user to enter the maximum stock allocation (0 to 1)
max_allocation = float(input("Enter the maximum stock allocation (0 to 1): "))

# Create a dictionary to store maximum allocations for each stock
max_allocations = {}
for stock in ticker_names:
    max_allocations[stock] = max_allocation

# Calculate and print the maximum and minimum risk after the user enters the stocks
sim_details = run_portfolio_optimization(ticker_names, start_date, end_date)
max_risk = sim_details['expectedRisk'].max()
min_risk = sim_details['expectedRisk'].min()
print("Maximum Risk:", max_risk)
print("Minimum Risk:", min_risk)

# Prompt the user to enter the desired risk level
risk_level = float(input("Enter the desired risk level: "))

# Run the function to find the best portfolio based on risk
get_best_portfolio_by_risk(ticker_names, start_date, end_date, risk_level)

# Plot the efficient frontier using simulation details
plot_efficient_frontier(sim_details)

# Prompt the user to add more stocks if desired
add_more_stocks = input("Do you want to add more stocks? (Yes/No): ")

while add_more_stocks.lower() == "yes":
    additional_tickers = input("Enter additional ticker names (comma-separated): ")
    additional_tickers = [ticker.strip() for ticker in additional_tickers.split(",")]

    # Extend the list of ticker names with additional ones
    ticker_names.extend(additional_tickers)

    # Run portfolio optimization and display results
    sim_details = run_portfolio_optimization(ticker_names, start_date, end_date)
    get_best_portfolio_by_risk(ticker_names, start_date, end_date, risk_level)
    plot_efficient_frontier(sim_details)

    # Prompt the user again
    add_more_stocks = input("Do you want to add more stocks? (Yes/No): ")

# Thank the user for using the tool
print("Thank you for using the portfolio optimization tool!")

# Store the simulation details in a SQLite database
connection = sqlite3.connect('portfolio_optimization.db')
sim_details.to_sql('simulation_details', connection, if_exists='replace', index=False)
connection.close()
