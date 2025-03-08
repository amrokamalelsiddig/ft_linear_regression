import csv
from scikit import sklearn_prediction


def load_model_data(file_path="thetas.csv"):
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        row = next(reader)
        theta0     = float(row[0])
        theta1     = float(row[1])
        min_mile   = float(row[2])
        max_mile   = float(row[3])
        min_price  = float(row[4])
        max_price  = float(row[5])
    return (theta0, theta1, min_mile, max_mile, min_price, max_price)

def predict_price(mileage, theta0, theta1,min_mile, max_mile,min_price, max_price):
    scaled_m = (mileage - min_mile) / (max_mile - min_mile)
    scaled_price_guess = theta0 + theta1 * scaled_m
    real_price_guess = scaled_price_guess * (max_price - min_price) + min_price
    return real_price_guess

if __name__ == "__main__":
    (theta0, theta1, min_mile, max_mile, min_price, max_price) = load_model_data("thetas.csv")
    user_input = input("Enter a mileage: ")
    try:
        mileage_val = float(user_input)
    except ValueError:
        print("Please enter a valid number for mileage.")
        exit(1)
    estimated_price = predict_price(mileage_val, theta0, theta1,min_mile, max_mile,min_price, max_price)
    print("my model :")
    print(f"Estimated price for mileage {mileage_val:.0f} is about {estimated_price:.2f}")
    print("sklearn model :")
    estimated_price_sk = sklearn_prediction(float(user_input))
    print(f"Estimated price for mileage {mileage_val:.0f} is about {estimated_price_sk:.2f}")
    delta = round(((estimated_price/estimated_price_sk * 100.00 ) - 100),2)
    print(f"diffrenct between my model and sklearn = {delta} %")
