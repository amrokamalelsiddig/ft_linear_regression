import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train import load_data,compute_r2,normalization
from predict import load_model_data

def plot_data_only(mileages, prices):
    plt.scatter(mileages, prices, label="Original Data")
    plt.xlabel("Mileage (km)")
    plt.ylabel("Price (€)")
    plt.title("Car Price vs Mileage")
    plt.legend()
    plt.savefig("plot.png") 
    print("Plot saved as plot.png")

def plot_regression_line(mileages, prices,theta0, theta1,min_mile, max_mile,min_price, max_price):
    plt.scatter(mileages, prices, label="Data Points")
    plt.xlabel("Mileage")
    plt.ylabel("Price")
    x_line = np.linspace(min(mileages), max(mileages), 100)
    y_line = []
    for x in x_line:
        x_scaled = (x - min_mile) / (max_mile - min_mile)
        y_scaled = theta0 + theta1 * x_scaled
        y_unscaled = y_scaled * (max_price - min_price) + min_price
        y_line.append(y_unscaled)
    plt.plot(x_line, y_line, color='red', label="Regression Line")
    plt.title("Mileage vs. Price (with Regression Line)")
    plt.legend()
    plt.savefig("plot_line.png") 
    print("Plot saved as plot.png")

def mean_absolute_percentage_error(mileages, prices, theta0, theta1):
    m = len(mileages)
    total_percentage_error = 0.0
    for i in range(m):
        actual = prices[i]
        predicted = theta0 + theta1 * mileages[i]
        diff = abs((actual - predicted) / actual)
        total_percentage_error += diff
    return (total_percentage_error / m) 


if __name__ == "__main__":
    mileage, prices = load_data("data/data.csv")
    theta0, theta1, min_mile, max_mile, min_price, max_price = load_model_data("thetas.csv")
    args = sys.argv
    if len(args) != 2 or args[1] not in ["-p_data", "-p_line", "-p"] :
        print(f"error : unknown input: {input}")
        print("Usage:")
        print("  python3 bonus.py -p_data       # execute bonus point 1")
        print("  python3 bonus.py -p_line       # execute bonus point 2")
        print("  python3 bonus.py -p            # execute bonus point 3")
        sys.exit(1)

    input = args[1]
    if input == "-p_data":
        plot_data_only(mileage, prices)
    elif input == "-p_line":
        plot_regression_line(mileage, prices,theta0, theta1,min_mile, max_mile,min_price, max_price)
    elif input == "-p":        
        mape_value = mean_absolute_percentage_error(mileage, prices, theta0, theta1)
        scaled_mileage, min_mile,max_mile = normalization(mileage)
        scaled_prices,min_price,max_price = normalization(prices)

        r2 = compute_r2(scaled_mileage,scaled_prices,theta0,theta1)
        print(f"MAPE: {mape_value:.2f}%                # Lower MAPE = Better accuracy.")
        print(f"R²  : {r2:.2f}                  # R²: ≥0.7 → Strong fit | 1.0 → Overfit | ≤0.3 → Weak fit")  
    else:
        print(f"error : unknown input: {input}")
        print("use one of the following flags : -p_data, -p_line, or -p.")
