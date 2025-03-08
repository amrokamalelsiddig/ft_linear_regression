import csv
import os

global gradient_descent 

def load_data(file_path):
    mileage = []
    prices = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            miles_holder = float(row[0])
            price_holder   = float(row[1])
            mileage.append(miles_holder)
            prices.append(price_holder)
    return mileage, prices

def normalization(values):
    min_value = min(values)
    max_value = max(values)
    scale = [(value - min_value)/(max_value - min_value) for value in values]
    return scale ,min_value, max_value

def compute_MSE(mileages, prices, t0, t1):
    m = len(mileages)
    total_error = 0
    for i in range(m):
        predication = t0 + (t1 * mileages[i])
        residual = predication - prices[i]
        total_error += residual ** 2
    return total_error / m

def compute_r2(mileages, prices, theta0, theta1):
    m = len(mileages)
    if m == 0:
        return 0.0
    mean_price = sum(prices) / m
    residual_sum_squares = 0 
    total_sum_squares = 0
    for i in range(m):
        predication = theta0 + (theta1 * mileages[i])
        residual_sum_squares +=  (predication - prices[i])**2
        total_sum_squares += ( prices[i] - mean_price )**2
    r2 = 1.0 - ( residual_sum_squares / total_sum_squares)
    return r2

def gradient_descent(mileages, prices, theta0, theta1, learning_rate):
    m = len(mileages)
    sum_theata0 = 0.0
    sum_theata1 = 0.0
    for i in range(m):
        predication = theta0 + (theta1 * mileages[i])
        sum_theata0 += predication - prices[i]
        sum_theata1 += (predication - prices[i]) * mileages[i]
    theta0_holder = learning_rate * (1.0 / m) * sum_theata0
    theta1_holder = learning_rate * (1.0 / m) * sum_theata1
    theta0_holder = theta0 - theta0_holder
    theta1_holder = theta1 - theta1_holder
    return theta0_holder, theta1_holder

def train_linear_model(mileages, prices, learning_rate=0, iterations=0):
    theta0 = 0.0
    theta1 = 0.0
    print("======================================================================================")
    for i in range(iterations):
        theta0, theta1 = gradient_descent(mileages, prices, theta0, theta1, learning_rate)
        if i % 1000 == 0:
            MSE = compute_MSE(mileages, prices, theta0, theta1)
            print(f"Iteration {i:5d} | Cost={MSE:.6f} | "f"theta0 = {theta0:.6f}, theta1 = {theta1:.6f}")
    print("======================================================================================")
    overall_MSE = compute_MSE(mileages, prices, theta0, theta1)
    r2 = compute_r2(mileages, prices, theta0, theta1)
    print(f"\ntraining completed after {iterations} iterations:")
    print(f"  MSE             = {overall_MSE:.6f}")
    print(f"  R²              = {r2:.6f} ")
    print(f"  theta0          = {theta0:.6f}")
    print(f"  theta1          = {theta1:.6f}")
    return theta0, theta1
        
def save_model(theta0, theta1, min_mile, max_mile,min_price, max_price,out_file="thetas.csv"):
    with open(out_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([theta0, theta1, min_mile, max_mile, min_price, max_price])

if __name__ == "__main__":
    raw_mileages, raw_prices = load_data("data/data.csv")
    scaled_mileages, min_mile, max_mile = normalization(raw_mileages)
    scaled_prices, min_price, max_price = normalization(raw_prices)
    learning_rate = 0.009
    iterations = 10000
    final_theta0, final_theta1 = train_linear_model(scaled_mileages, scaled_prices,learning_rate, iterations)
    save_model(final_theta0, final_theta1,min_mile, max_mile,min_price, max_price,out_file="thetas.csv")
    print("model saved to thetas.csv")
