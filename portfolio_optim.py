import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Simulate or Load Data
np.random.seed(42)
num_assets = 5
num_days = 252  # Trading days in a year
returns = np.random.normal(0.0005, 0.02, (num_days, num_assets))  # Simulated daily returns

# Convert to DataFrame
assets = ['Asset_' + str(i) for i in range(1, num_assets + 1)]
returns_df = pd.DataFrame(returns, columns=assets)
print(returns_df)

# Step 2: Portfolio Metrics
weights = np.random.random(num_assets)
weights /= np.sum(weights)  # Normalize to 1

portfolio_return = np.dot(weights, returns_df.mean()) * num_days
portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns_df.cov() * num_days, weights)))
sharpe_ratio = portfolio_return / portfolio_volatility

print(f"Return: {portfolio_return:.4f}, Volatility: {portfolio_volatility:.4f}, Sharpe Ratio: {sharpe_ratio:.4f}")

# Step 3: Optimize Portfolio
num_portfolios = 10000
results = np.zeros((3, num_portfolios))
all_weights = []

for i in range(num_portfolios):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    all_weights.append(weights)
    
    # Portfolio Metrics
    port_return = np.dot(weights, returns_df.mean()) * num_days
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(returns_df.cov() * num_days, weights)))
    sharpe = port_return / port_volatility
    
    # Store results
    results[0, i] = port_return
    results[1, i] = port_volatility
    results[2, i] = sharpe

# Step 4: Efficient Frontier Plot
plt.scatter(results[1, :], results[0, :], c=results[2, :], cmap='viridis', marker='o')
plt.title('Efficient Frontier')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.colorbar(label='Sharpe Ratio')

# Highlight Max Sharpe Portfolio
max_sharpe_idx = np.argmax(results[2, :])
plt.scatter(results[1, max_sharpe_idx], results[0, max_sharpe_idx], c='red', marker='*', s=200, label='Max Sharpe')
plt.legend()
plt.show()
