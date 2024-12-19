import yfinance as yf
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
#  import pandas as pd

# Define the stock tickers and time period
tickers = ["AAPL", "MSFT", "TSLA"]
start_date = "2020-01-01"
end_date = "2023-12-31"

# Download historical data
data = yf.download(tickers, start=start_date, end=end_date)["Adj Close"]

# Calculate daily returns
returns = data.pct_change().dropna()

# Print summary of the data
print(returns.head())

# Calculate the mean returns and covariance matrix
mean_returns = returns.mean()
cov_matrix = returns.cov()

print("Mean Returns:\n", mean_returns)
print("Covariance Matrix:\n", cov_matrix)


# Portfolio optimization functions
def portfolio_performance(weights, mean_returns, cov_matrix):
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_risk


def minimize_risk(weights, mean_returns, cov_matrix, target_return):
    _, portfolio_risk = portfolio_performance(
            weights, mean_returns, cov_matrix)
    return portfolio_risk


# Constraints: Weights sum to 1, and target return
def constraint(weights):
    return np.sum(weights) - 1


# Bounds: Each weight is between 0 and 1
bounds = tuple((0, 1) for _ in range(len(mean_returns)))

# Target return
target_return = 0.001  # Adjust based on your desired return

# Initial weights
initial_weights = np.array([1/len(mean_returns)] * len(mean_returns))

# Optimization
constraints = [{'type': 'eq', 'fun': constraint},
               {'type': 'eq',
                'fun': lambda weights:
                np.dot(weights, mean_returns) - target_return}]

optimized = minimize(
        minimize_risk, initial_weights,
        args=(mean_returns, cov_matrix, target_return),
        method='SLSQP', bounds=bounds, constraints=constraints)

optimized_weights = optimized.x
print("Optimized Weights:", optimized_weights)


def efficient_frontier(mean_returns, cov_matrix, num_portfolios=100):
    results = {"returns": [], "risks": [], "weights": []}
    for _ in range(num_portfolios):
        weights = np.random.dirichlet(
                np.ones(len(mean_returns)), size=1).flatten()
        portfolio_return, portfolio_risk = portfolio_performance(
                weights, mean_returns, cov_matrix)
        results["returns"].append(portfolio_return)
        results["risks"].append(portfolio_risk)
        results["weights"].append(weights)
    return results


# Generate random portfolios
results = efficient_frontier(mean_returns, cov_matrix)

# Plot efficient frontier
plt.scatter(results["risks"], results["returns"],
            c=np.array(results["returns"])/np.array(results["risks"]),
            cmap="viridis")
plt.colorbar(label="Sharpe Ratio")
plt.xlabel("Risk (Standard Deviation)")
plt.ylabel("Return")
plt.title("Efficient Frontier")
plt.show()

